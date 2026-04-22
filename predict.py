"""
单样本推理脚本：输入患者视频文件夹，输出谵妄概率和预测结果。

用法:
    python predict.py --patient_dir "path/to/patient_folder"
    python predict.py --patient_dir "path/to/patient_folder" --checkpoint "best_model.pth"
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import cv2
import numpy as np
import torch
import mediapipe as mp

import config
from model import DeliriumNet
from audio_processor import process_clip_audio


def _ckpt_uses_audio(state_dict):
    return any(k.startswith("audio_stream.") for k in state_dict.keys())


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 恢复训练时的 max_frames 配置（避免 pos_embedding 尺寸不匹配）
    if "max_frames_per_clip" in ckpt:
        config.MAX_FRAMES_PER_CLIP = ckpt["max_frames_per_clip"]
    ckpt_use_audio = ckpt.get("use_audio")
    if ckpt_use_audio is None:
        ckpt_use_audio = _ckpt_uses_audio(ckpt["model_state_dict"])
    model = DeliriumNet(pretrained=False, use_audio=ckpt_use_audio).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _probe_best_model(cap, det_close, det_far, n_probe=8):
    """采样若干帧，返回检测率更高的模型（平局时选近距离模型）。"""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // n_probe)
    score_close = score_far = 0
    for i in range(n_probe):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if det_close.process(rgb).detections:
            score_close += 1
        if det_far.process(rgb).detections:
            score_far += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return det_close if score_close >= score_far else det_far


def extract_faces_from_video(video_path, face_detectors, frame_rate=2, face_size=112, max_frames=80):
    """从单个视频中提取人脸帧，自动选择最佳检测模型。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    det_close, det_far = face_detectors
    primary = _probe_best_model(cap, det_close, det_far)
    fallback = det_far if primary is det_close else det_close

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    frame_interval = max(1, int(video_fps / frame_rate))

    faces = []
    frame_idx = 0
    last_bbox = None

    while len(faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = primary.process(rgb_frame)
        bbox = None
        if res.detections:
            best = max(res.detections,
                       key=lambda d: d.location_data.relative_bounding_box.width *
                                     d.location_data.relative_bounding_box.height)
            rb = best.location_data.relative_bounding_box
            bbox = (rb.xmin, rb.ymin, rb.width, rb.height)
        else:
            res2 = fallback.process(rgb_frame)
            if res2.detections:
                best = max(res2.detections,
                           key=lambda d: d.location_data.relative_bounding_box.width *
                                         d.location_data.relative_bounding_box.height)
                rb = best.location_data.relative_bounding_box
                bbox = (rb.xmin, rb.ymin, rb.width, rb.height)

        if bbox is not None:
            last_bbox = bbox
        elif last_bbox is not None:
            bbox = last_bbox

        if bbox is None:
            continue

        xmin_rel, ymin_rel, w_rel, h_rel = bbox
        pad = 0.2
        cx = xmin_rel + w_rel / 2
        cy = ymin_rel + h_rel / 2
        side = max(w_rel, h_rel) * (1 + pad)
        x1 = max(0, int((cx - side / 2) * w))
        y1 = max(0, int((cy - side / 2) * h))
        x2 = min(w, int((cx + side / 2) * w))
        y2 = min(h, int((cy + side / 2) * h))

        if x2 - x1 < 20 or y2 - y1 < 20:
            continue

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (face_size, face_size))
        faces.append(face)

    cap.release()
    return faces


def preprocess_faces(faces):
    """将人脸图像列表转为模型输入tensor"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def to_tensor_norm(face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
        return (t - mean) / std

    tensors = [to_tensor_norm(f) for f in faces]
    return torch.stack(tensors)  # [T, 3, H, W]


@torch.no_grad()
def predict(model, patient_dir, device):
    """
    对单个患者进行谵妄预测。

    Args:
        model: 加载好的模型
        patient_dir: 患者视频文件夹路径

    Returns:
        dict: 包含预测概率和各片段信息
    """
    # 找到所有视频文件
    video_files = sorted([
        f for f in os.listdir(patient_dir)
        if f.lower().endswith('.mp4')
    ])

    if not video_files:
        return {"error": f"未找到视频文件: {patient_dir}"}

    print(f"找到 {len(video_files)} 个视频片段")

    # 初始化双模型人脸检测器（近距离 + 远距离，自动选择最佳）
    mp_face_detection = mp.solutions.face_detection
    conf = config.MIN_FACE_CONFIDENCE
    face_detectors = (
        mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=conf),
        mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=conf),
    )

    all_clips = []
    clip_lengths = []
    clip_mask = []
    all_audio = []
    all_semantic = []
    all_semantic_conf = []
    audio_mask = []
    clip_info = []

    n_mfcc_total = config.N_MFCC * 3

    for clip_idx, vf in enumerate(video_files[:config.MAX_CLIPS]):
        video_path = os.path.join(patient_dir, vf)
        print(f"  处理片段 {clip_idx + 1}: {vf}")

        faces = extract_faces_from_video(
            video_path, face_detectors,
            frame_rate=config.FRAME_RATE,
            face_size=config.FACE_SIZE,
            max_frames=config.MAX_FRAMES_PER_CLIP
        )

        if not faces:
            print(f"    [警告] 未检测到人脸，跳过")
            continue

        print(f"    提取到 {len(faces)} 帧人脸")
        clip_tensor = preprocess_faces(faces)
        T = clip_tensor.shape[0]
        if T > config.MAX_FRAMES_PER_CLIP:
            clip_tensor = clip_tensor[:config.MAX_FRAMES_PER_CLIP]
            T = config.MAX_FRAMES_PER_CLIP

        clip_lengths.append(T)
        if T < config.MAX_FRAMES_PER_CLIP:
            pad = torch.zeros(config.MAX_FRAMES_PER_CLIP - T, 3,
                              config.FACE_SIZE, config.FACE_SIZE)
            clip_tensor = torch.cat([clip_tensor, pad], dim=0)
        all_clips.append(clip_tensor)
        clip_mask.append(1)
        clip_info.append({"filename": vf, "frames": T})

        # 音频处理
        if config.USE_AUDIO:
            audio_result = process_clip_audio(
                video_path,
                clip_index=clip_idx,
                use_whisper=config.USE_WHISPER and config.USE_AUDIO
            )
            mfcc = audio_result.get("mfcc")
            sem_score = audio_result.get("semantic_score", 0.5)
            sem_conf = audio_result.get("semantic_confidence", 0.0)
            if mfcc is not None:
                mfcc_t = torch.tensor(mfcc, dtype=torch.float32)
                Ta = mfcc_t.shape[0]
                if Ta < config.MAX_AUDIO_FRAMES:
                    pad = torch.zeros(config.MAX_AUDIO_FRAMES - Ta, n_mfcc_total)
                    mfcc_t = torch.cat([mfcc_t, pad], dim=0)
                else:
                    mfcc_t = mfcc_t[:config.MAX_AUDIO_FRAMES]
                all_audio.append(mfcc_t)
                audio_mask.append(1)
            else:
                all_audio.append(torch.zeros(config.MAX_AUDIO_FRAMES, n_mfcc_total))
                audio_mask.append(0)
            all_semantic.append(sem_score)
            all_semantic_conf.append(sem_conf)

    for det in face_detectors:
        det.close()

    if not all_clips:
        return {"error": "所有片段均无法检测到人脸"}

    # 填充到 max_clips
    while len(all_clips) < config.MAX_CLIPS:
        all_clips.append(torch.zeros(config.MAX_FRAMES_PER_CLIP, 3,
                                     config.FACE_SIZE, config.FACE_SIZE))
        clip_lengths.append(0)
        clip_mask.append(0)
        if config.USE_AUDIO:
            all_audio.append(torch.zeros(config.MAX_AUDIO_FRAMES, n_mfcc_total))
            all_semantic.append(0.5)
            all_semantic_conf.append(0.0)
            audio_mask.append(0)

    # 构建batch (batch_size=1)
    clips_tensor = torch.stack(all_clips).unsqueeze(0).to(device)
    lengths_tensor = torch.tensor([clip_lengths], dtype=torch.long).to(device)
    mask_tensor = torch.tensor([clip_mask], dtype=torch.float).to(device)

    audio_tensor = semantic_tensor = semantic_conf_tensor = audio_mask_tensor = None
    if config.USE_AUDIO and all_audio:
        audio_tensor = torch.stack(all_audio).unsqueeze(0).to(device)           # [1, max_clips, T_audio, N_MFCC*3]
        semantic_tensor = torch.tensor(all_semantic, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)  # [1, max_clips, 1]
        semantic_conf_tensor = torch.tensor(all_semantic_conf, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)  # [1, max_clips, 1]
        audio_mask_tensor = torch.tensor([audio_mask], dtype=torch.float).to(device)

    # 推理
    logits = model(clips_tensor, lengths_tensor, mask_tensor,
                   audio_tensor, semantic_tensor,
                   semantic_conf_tensor, audio_mask_tensor).squeeze()
    prob = torch.sigmoid(logits).item()

    return {
        "delirium_probability": prob,
        "prediction": "谵妄" if prob >= 0.5 else "清醒",
        "confidence": prob if prob >= 0.5 else 1 - prob,
        "num_clips": len(clip_info),
        "clip_info": clip_info
    }


def main():
    parser = argparse.ArgumentParser(description="谵妄识别推理")
    parser.add_argument("--patient_dir", type=str, required=True,
                        help="患者视频文件夹路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型检查点路径")
    args = parser.parse_args()

    device = config.DEVICE
    checkpoint_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.isfile(checkpoint_path):
        print(f"错误: 找不到模型检查点 {checkpoint_path}")
        print("请先运行 train.py 训练模型")
        return

    if not os.path.isdir(args.patient_dir):
        print(f"错误: 患者目录不存在 {args.patient_dir}")
        return

    print(f"患者目录: {args.patient_dir}")
    print(f"设备: {device}")

    # 加载模型
    model = load_model(checkpoint_path, device)
    print(f"模型已加载: {checkpoint_path}\n")

    # 预测
    result = predict(model, args.patient_dir, device)

    if "error" in result:
        print(f"\n预测失败: {result['error']}")
        return

    print(f"\n{'='*40}")
    print(f"预测结果: {result['prediction']}")
    print(f"谵妄概率: {result['delirium_probability']:.4f} ({result['delirium_probability']*100:.1f}%)")
    print(f"置信度:   {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
    print(f"使用片段: {result['num_clips']} 个")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
