"""
对已预处理的样本运行 Whisper ASR，更新 ASR JSON 文件中的转录文字和语义得分。

preprocess.py 在调试模式下跳过了 Whisper，导致所有 ASR JSON 中 text="" 且
semantic_score=0.5。本脚本读取 manifest.csv，找到原始视频，重新运行 Whisper，
并覆写 clip_{i}_asr.json。

用法:
    python run_whisper.py                   # 处理 manifest 中所有患者
    python run_whisper.py --patients 5      # 只处理前5个患者（快速验证）
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import argparse
from tqdm import tqdm

import config
from audio_processor import (
    extract_audio_from_video, transcribe_audio,
    semantic_score, get_whisper_model
)


def find_original_videos(patient_id, label):
    """
    在原始数据目录中查找患者的所有视频文件（按文件名排序）。
    """
    source_dir = config.LUCID_DIR if label == 0 else config.DELIRIUM_DIR
    patient_dir = os.path.join(source_dir, patient_id)
    if not os.path.isdir(patient_dir):
        return []
    videos = sorted([
        os.path.join(patient_dir, f)
        for f in os.listdir(patient_dir)
        if f.lower().endswith('.mp4')
    ])
    return videos[:config.MAX_CLIPS]


def update_asr_for_patient(patient_id, label, audio_dirs):
    """
    对单个患者的每个片段运行 Whisper，更新 clip_{i}_asr.json。

    Returns:
        list of dict: 每个片段的结果摘要
    """
    videos = find_original_videos(patient_id, label)
    if not videos:
        return []

    results = []
    for clip_idx, (video_path, audio_dir) in enumerate(zip(videos, audio_dirs)):
        if not audio_dir:
            continue

        asr_path = os.path.join(audio_dir, f"clip_{clip_idx}_asr.json")

        # 提取音频并转录
        audio, sr = extract_audio_from_video(video_path)
        if audio is None:
            results.append({"clip": clip_idx, "text": "", "score": 0.5, "error": "audio_extract_failed"})
            continue

        asr = transcribe_audio(audio, sr, use_whisper=True)
        sc, conf = semantic_score(asr["text"], clip_idx)

        # 如果无语音概率高，中性
        if asr["no_speech_prob"] > 0.7:
            conf = 0.0
            sc = 0.5

        # 覆写 ASR JSON
        asr_data = {
            "text": asr["text"],
            "no_speech_prob": asr["no_speech_prob"],
            "semantic_score": sc,
            "semantic_confidence": conf,
            "clip_index": clip_idx,
        }
        with open(asr_path, "w", encoding="utf-8") as f:
            json.dump(asr_data, f, ensure_ascii=False, indent=2)

        results.append({
            "clip": clip_idx,
            "text": asr["text"][:60] + ("..." if len(asr["text"]) > 60 else ""),
            "score": sc,
            "confidence": conf,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="对预处理样本运行 Whisper ASR")
    parser.add_argument("--patients", type=int, default=None,
                        help="只处理前 N 个患者（默认全部）")
    args = parser.parse_args()

    manifest_path = os.path.join(config.PROCESSED_DIR, "manifest.csv")
    if not os.path.isfile(manifest_path):
        print(f"错误: 找不到 manifest.csv，请先运行 preprocess.py")
        return

    # 读取 manifest
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "patient_id": row["patient_id"],
                "label": int(row["label"]),
                "audio_dirs": [d for d in row.get("audio_dirs", "").split("|")],
            })

    if args.patients:
        records = records[:args.patients]

    print(f"处理 {len(records)} 个患者，加载 Whisper 模型...")
    get_whisper_model()  # 预加载（避免每次单独计时）
    print("Whisper 模型已加载\n")

    # 汇总统计
    total_clips = 0
    transcribed = 0
    semantic_fired = 0  # 触发了非中性语义规则的片段数
    all_summaries = []

    for rec in tqdm(records, desc="ASR转录"):
        clip_results = update_asr_for_patient(
            rec["patient_id"], rec["label"], rec["audio_dirs"]
        )
        for cr in clip_results:
            total_clips += 1
            if cr.get("text"):
                transcribed += 1
            if cr.get("score", 0.5) != 0.5:
                semantic_fired += 1
                all_summaries.append({
                    "patient": rec["patient_id"][:30],
                    "label": "谵妄" if rec["label"] == 1 else "清醒",
                    "clip": cr["clip"],
                    "text": cr.get("text", ""),
                    "score": cr.get("score", 0.5),
                })

    print(f"\n{'='*60}")
    print(f"ASR 完成:")
    print(f"  总片段数:       {total_clips}")
    print(f"  成功转录:       {transcribed} ({transcribed/max(total_clips,1)*100:.0f}%)")
    print(f"  触发语义规则:   {semantic_fired}")
    print(f"{'='*60}")

    if all_summaries:
        print(f"\n触发语义规则的片段（谵妄诊断信号）:")
        print(f"{'患者':<30} {'标签':<6} {'片段':<4} {'语义分':<6} 转录文字")
        print("-" * 80)
        for s in all_summaries:
            print(f"{s['patient']:<30} {s['label']:<6} {s['clip']:<4} {s['score']:<6.1f} {s['text']}")
    else:
        print("\n未触发语义规则（可能视频中无匹配关键词，或语音识别结果为空）")
        print("提示：部分患者语音较轻/模糊，Whisper可能无法识别，属于正常现象")


if __name__ == "__main__":
    main()
