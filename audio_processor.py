"""
音频处理模块：MFCC特征提取、Whisper语音识别、语义规则评分。

三层处理：
1. 声学特征 (MFCC)：始终可用，捕捉语速/清晰度/停顿等谵妄特征语音
2. ASR识别 (faster-whisper)：提取患者回答文字
3. 语义评分 (规则匹配)：判断患者回答是否符合谵妄特征

设计说明：
- 不做说话人分离，处理完整音频，模型自动学习关注患者部分
- DEBUG_MODE下跳过Whisper，语义分数设为中性值0.5
"""
import os
import json
import warnings
import numpy as np
import librosa
import config

warnings.filterwarnings("ignore", category=UserWarning)


# ============ 诊断性问题的语义规则 ============
# 基于PDF中的CAM-ICU评估范式，建立已知问题的答案规则
# 结构: {关键词片段: (正确答案极性, 错误答案关键词, 正确答案关键词)}
#   - 极性 1 = 应该肯定, -1 = 应该否定
#   - 若检测到"错误答案关键词"则 score=1.0（谵妄指标）
#   - 若检测到"正确答案关键词"则 score=0.0（非谵妄）
#   - 否则 score=0.5（中性）

SEMANTIC_RULES = {
    # 思维测试（clip_index=2，视频片段_2）
    "石头": {
        "correct_polarity": -1,  # 正确答案：否（石头沉）
        "wrong_keywords": ["能浮", "会浮", "浮起", "浮上", "对", "是的", "能", "可以"],
        "right_keywords": ["不", "沉", "不能", "不会", "不浮"],
    },
    "一斤": {
        "correct_polarity": -1,  # 正确答案：否（一斤<两斤，所以一斤不重于两斤）
        "wrong_keywords": ["是的", "对", "重", "一斤重", "是"],
        "right_keywords": ["不", "不是", "不重", "两斤重"],
    },
    "两斤": {  # 兜底，与"一斤"规则相同
        "correct_polarity": -1,
        "wrong_keywords": ["一斤重", "是的", "对"],
        "right_keywords": ["两斤重", "不", "不是"],
    },
    "海里": {
        "correct_polarity": 1,   # 正确答案：是（海里有鱼）
        "wrong_keywords": ["没有", "没", "无", "不", "没鱼"],
        "right_keywords": ["有", "是", "对", "有鱼"],
    },
    "鱼": {
        "correct_polarity": 1,
        "wrong_keywords": ["没有鱼", "没鱼", "不"],
        "right_keywords": ["有鱼", "有", "是"],
    },
    "铁锤": {
        "correct_polarity": 1,   # 正确答案：是（铁锤能钉钉子）
        "wrong_keywords": ["不能", "不可以", "锯木头", "不"],
        "right_keywords": ["能", "可以", "对", "是"],
    },
    "树叶": {
        "correct_polarity": 1,   # 正确答案：是（树叶能浮在水面）
        "wrong_keywords": ["不能", "沉", "不"],
        "right_keywords": ["能浮", "能", "是", "对"],
    },
    "大象": {
        "correct_polarity": -1,  # 正确答案：否（海里没有大象）
        "wrong_keywords": ["有", "是", "对"],
        "right_keywords": ["没有", "没", "不"],
    },
}


def extract_audio_from_video(video_path, target_sr=16000):
    """
    从视频文件中提取音频。

    Args:
        video_path: 视频文件路径
        target_sr: 目标采样率（Hz）

    Returns:
        tuple(np.ndarray, int): 音频数组和采样率，失败时返回 (None, 0)
    """
    try:
        audio, sr = librosa.load(video_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        print(f"  [音频] 提取失败 {os.path.basename(video_path)}: {e}")
        return None, 0


def compute_mfcc_features(audio, sr, n_mfcc=None, max_frames=None):
    """
    计算MFCC特征序列（含delta和delta-delta）。

    Args:
        audio: 音频数组
        sr: 采样率
        n_mfcc: MFCC系数数量（默认使用config）
        max_frames: 最大帧数（默认使用config）

    Returns:
        np.ndarray: [T, n_mfcc*3] 特征矩阵（MFCC + delta + delta-delta）
        None: 失败时
    """
    n_mfcc = n_mfcc or config.N_MFCC
    max_frames = max_frames or config.MAX_AUDIO_FRAMES

    if audio is None or len(audio) == 0:
        return None

    try:
        # MFCC: [n_mfcc, T]
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                     hop_length=160, n_fft=512)
        # delta特征（速度）
        delta = librosa.feature.delta(mfcc)
        # delta-delta特征（加速度）
        delta2 = librosa.feature.delta(mfcc, order=2)

        # 拼接: [n_mfcc*3, T]
        features = np.concatenate([mfcc, delta, delta2], axis=0)
        # 转置: [T, n_mfcc*3]
        features = features.T.astype(np.float32)

        # 截断或保留
        if features.shape[0] > max_frames:
            features = features[:max_frames]

        # 归一化（按特征维度）
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        return features

    except Exception as e:
        print(f"  [音频] MFCC计算失败: {e}")
        return None


def get_whisper_model(model_size=None):
    """懒加载Whisper模型（全局单例）"""
    if not hasattr(get_whisper_model, '_model') or get_whisper_model._model is None:
        from faster_whisper import WhisperModel
        import ssl
        import os as _os
        # 绕过SSL验证以支持离线/受限网络环境下的模型加载
        ssl._create_default_https_context = ssl._create_unverified_context
        _os.environ.setdefault("CURL_CA_BUNDLE", "")
        _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        model_size = model_size or config.WHISPER_MODEL
        print(f"  [ASR] 加载 faster-whisper {model_size} 模型...")
        get_whisper_model._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"  # CPU上使用int8量化加速
        )
    return get_whisper_model._model


get_whisper_model._model = None


def transcribe_audio(audio, sr, use_whisper=True):
    """
    使用Whisper对音频进行语音识别。

    Args:
        audio: 音频数组（16kHz）
        sr: 采样率
        use_whisper: 是否实际运行Whisper（DEBUG模式下为False）

    Returns:
        dict: {
            "text": 识别文字,
            "no_speech_prob": 无语音概率(0-1),
            "segments": [段落列表]
        }
    """
    if not use_whisper or audio is None:
        return {"text": "", "no_speech_prob": 1.0, "segments": []}

    try:
        model = get_whisper_model()
        # faster-whisper需要float32音频
        audio_f32 = audio.astype(np.float32)
        segments, info = model.transcribe(
            audio_f32,
            language="zh",
            beam_size=3,
            vad_filter=True,         # 语音活动检测，过滤静音
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        segment_list = list(segments)
        full_text = "".join(s.text for s in segment_list)
        # 计算平均无语音概率
        lang_probs = getattr(info, 'all_language_probs', None)
        avg_no_speech = lang_probs.get("zh", 0.0) if isinstance(lang_probs, dict) else 0.0
        # 简单估计：如果文字很短则认为无语音概率高
        no_speech_prob = 0.2 if len(full_text.strip()) > 2 else 0.8

        return {
            "text": full_text.strip(),
            "no_speech_prob": no_speech_prob,
            "segments": [{"start": s.start, "end": s.end, "text": s.text}
                         for s in segment_list]
        }
    except Exception as e:
        print(f"  [ASR] 识别失败: {e}")
        return {"text": "", "no_speech_prob": 1.0, "segments": []}


def semantic_score(text, clip_index=None):
    """
    基于规则匹配判断患者回答是否含谵妄指标。

    通过检测已知诊断性问题的关键词和回答词判断：
    - score=1.0: 回答错误（谵妄指标）
    - score=0.5: 无法判断（中性）
    - score=0.0: 回答正确（非谵妄）

    Args:
        text: ASR识别出的文字
        clip_index: 片段序号（0=意识, 1=注意力, 2=思维）

    Returns:
        tuple(float, float): (score, confidence)
            - score: 谵妄语义得分 [0, 1]
            - confidence: 匹配置信度 [0, 1]
    """
    if not text or len(text.strip()) < 2:
        return 0.5, 0.0  # 无文字，中性低置信度

    text = text.strip()

    # 在文字中查找匹配的诊断性问题关键词
    for trigger_word, rule in SEMANTIC_RULES.items():
        if trigger_word in text:
            # 找到相关问题，检查回答
            for wrong_kw in rule["wrong_keywords"]:
                if wrong_kw in text:
                    return 1.0, 0.9  # 回答错误，高置信度谵妄指标
            for right_kw in rule["right_keywords"]:
                if right_kw in text:
                    return 0.0, 0.9  # 回答正确，高置信度非谵妄

            # 找到问题但无法判断回答
            return 0.5, 0.3

    # 未找到已知问题关键词，返回中性
    return 0.5, 0.1


def process_clip_audio(video_path, clip_index=0, use_whisper=True, debug=False):
    """
    处理单个视频片段的完整音频流程。

    Args:
        video_path: 视频文件路径
        clip_index: 片段序号（影响语义评分规则）
        use_whisper: 是否运行ASR
        debug: 调试模式（跳过Whisper，减少帧数）

    Returns:
        dict: {
            "mfcc": np.ndarray [T, N_MFCC*3] 或 None,
            "asr": dict（ASR结果）,
            "semantic_score": float,
            "semantic_confidence": float
        }
    """
    max_frames = config.MAX_AUDIO_FRAMES // 4 if debug else config.MAX_AUDIO_FRAMES

    # 1. 提取音频
    audio, sr = extract_audio_from_video(video_path)

    # 2. 计算MFCC
    mfcc = compute_mfcc_features(audio, sr, max_frames=max_frames) if audio is not None else None

    # 3. ASR（调试模式跳过）
    run_whisper = use_whisper and not debug and audio is not None
    asr_result = transcribe_audio(audio, sr, use_whisper=run_whisper)

    # 4. 语义评分
    sc, conf = semantic_score(asr_result["text"], clip_index)
    # 当无语音概率高时，降低置信度
    if asr_result["no_speech_prob"] > 0.7:
        conf = 0.0
        sc = 0.5

    return {
        "mfcc": mfcc,
        "asr": asr_result,
        "semantic_score": sc,
        "semantic_confidence": conf,
    }


def save_audio_features(result, patient_out_dir, clip_idx):
    """
    保存音频特征到文件。

    保存路径:
        {patient_out_dir}/clip_{clip_idx}_mfcc.npy
        {patient_out_dir}/clip_{clip_idx}_asr.json

    Returns:
        str: mfcc 文件路径，或 None 如果没有特征
    """
    mfcc_path = None

    if result["mfcc"] is not None:
        mfcc_path = os.path.join(patient_out_dir, f"clip_{clip_idx}_mfcc.npy")
        np.save(mfcc_path, result["mfcc"])

    asr_path = os.path.join(patient_out_dir, f"clip_{clip_idx}_asr.json")
    asr_data = {
        "text": result["asr"]["text"],
        "no_speech_prob": result["asr"]["no_speech_prob"],
        "semantic_score": result["semantic_score"],
        "semantic_confidence": result["semantic_confidence"],
        "clip_index": clip_idx,
    }
    with open(asr_path, "w", encoding="utf-8") as f:
        json.dump(asr_data, f, ensure_ascii=False, indent=2)

    return mfcc_path


def load_audio_features(patient_dir, clip_idx, max_frames=None):
    """
    加载保存的音频特征（dataset.py中调用）。

    Returns:
        tuple(np.ndarray or None, dict): (mfcc特征, asr结果)
    """
    max_frames = max_frames or config.MAX_AUDIO_FRAMES

    mfcc_path = os.path.join(patient_dir, f"clip_{clip_idx}_mfcc.npy")
    asr_path = os.path.join(patient_dir, f"clip_{clip_idx}_asr.json")

    mfcc = None
    if os.path.isfile(mfcc_path):
        mfcc = np.load(mfcc_path)
        if mfcc.shape[0] > max_frames:
            mfcc = mfcc[:max_frames]

    asr_data = {"text": "", "no_speech_prob": 1.0, "semantic_score": 0.5,
                "semantic_confidence": 0.0, "clip_index": clip_idx}
    if os.path.isfile(asr_path):
        with open(asr_path, "r", encoding="utf-8") as f:
            asr_data = json.load(f)

    return mfcc, asr_data
