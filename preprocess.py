"""
数据预处理：从视频中抽帧、检测人脸、裁剪对齐、保存处理后图像。
生成数据清单CSV供训练使用。

用法:
    python preprocess.py                # 处理全部数据（或DEBUG_MODE下少量数据）
    python preprocess.py --debug        # 强制调试模式
    python preprocess.py --full         # 强制全量模式
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import csv
import argparse
import warnings
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import config
from audio_processor import process_clip_audio, save_audio_features


warnings.filterwarnings("ignore", category=FutureWarning)


def create_face_detectors():
    """创建双模型MediaPipe人脸检测器（近距离 + 远距离）"""
    mp_face_detection = mp.solutions.face_detection
    conf = config.MIN_FACE_CONFIDENCE
    det_close = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=conf)
    det_far = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=conf)
    return det_close, det_far


def _probe_best_model(cap, det_close, det_far, n_probe=8):
    """
    采样视频开头若干帧，判断哪个模型检测率更高，返回胜出的检测器。
    两者平局时优先用近距离模型（实测对ICU床边视频效果更好）。
    """
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


def _best_detection(detections):
    """从检测结果中取面积最大的人脸，返回相对坐标 bbox 或 None。"""
    if not detections:
        return None
    best = max(detections,
               key=lambda d: d.location_data.relative_bounding_box.width *
                             d.location_data.relative_bounding_box.height)
    rb = best.location_data.relative_bounding_box
    return (rb.xmin, rb.ymin, rb.width, rb.height)


def extract_and_crop_faces(video_path, face_detectors, frame_rate=2, face_size=112):
    """
    从视频中抽帧并裁剪人脸。自动探测使用近距离或远距离模型。

    Args:
        video_path: 视频文件路径
        face_detectors: (det_close, det_far) 元组
        frame_rate: 抽帧率(fps)
        face_size: 裁剪后人脸尺寸

    Returns:
        list of np.ndarray: 裁剪后的人脸图像列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [警告] 无法打开视频: {video_path}")
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

    while len(faces) < config.MAX_FRAMES_PER_CLIP:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 先用主模型，失败再用备用模型
        res = primary.process(rgb_frame)
        bbox = _best_detection(res.detections)
        if bbox is None:
            res2 = fallback.process(rgb_frame)
            bbox = _best_detection(res2.detections)

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

        face_crop = frame[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (face_size, face_size))
        faces.append(face_crop)

    cap.release()
    return faces


def get_patient_folders(data_dir, label):
    """获取指定目录下的所有患者文件夹"""
    folders = []
    if not os.path.isdir(data_dir):
        return folders
    for name in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, name)
        if os.path.isdir(full_path) and not name.startswith('.'):
            folders.append((full_path, name, label))
    return folders


def process_patient(patient_dir, patient_name, label, face_detectors, debug=False):
    """
    处理单个患者的所有视频片段（视频帧 + 音频特征）。

    Returns:
        dict: {patient_id, label, clip_dirs, audio_dirs, total_frames}
        或 None（如果无有效数据）
    """
    video_files = sorted([
        f for f in os.listdir(patient_dir)
        if f.lower().endswith('.mp4')
    ])

    if not video_files:
        return None

    label_name = "delirium" if label == 1 else "lucid"
    patient_id = patient_name.replace(" ", "_")
    patient_out_dir = os.path.join(config.PROCESSED_DIR, label_name, patient_id)
    os.makedirs(patient_out_dir, exist_ok=True)

    clip_dirs = []
    audio_dirs = []
    total_frames = 0

    for clip_idx, vf in enumerate(video_files[:config.MAX_CLIPS]):
        video_path = os.path.join(patient_dir, vf)
        clip_out_dir = os.path.join(patient_out_dir, f"clip_{clip_idx}")

        # --- 视频帧处理 ---
        if os.path.isdir(clip_out_dir) and len(os.listdir(clip_out_dir)) > 0:
            n = len([f for f in os.listdir(clip_out_dir) if f.endswith('.jpg')])
            if n > 0:
                clip_dirs.append(clip_out_dir)
                total_frames += n
            else:
                clip_dirs.append(None)
        else:
            os.makedirs(clip_out_dir, exist_ok=True)
            faces = extract_and_crop_faces(
                video_path, face_detectors,
                frame_rate=config.FRAME_RATE,
                face_size=config.FACE_SIZE
            )
            if faces:
                for i, face in enumerate(faces):
                    cv2.imwrite(os.path.join(clip_out_dir, f"frame_{i:04d}.jpg"), face)
                clip_dirs.append(clip_out_dir)
                total_frames += len(faces)
            else:
                clip_dirs.append(None)

        # --- 音频处理 ---
        # 视觉-only模式下跳过音频提取，可显著加速全量预处理。
        if not config.USE_AUDIO:
            audio_dirs.append(patient_out_dir)
        else:
            mfcc_path = os.path.join(patient_out_dir, f"clip_{clip_idx}_mfcc.npy")
            if os.path.isfile(mfcc_path):
                audio_dirs.append(patient_out_dir)
            else:
                audio_result = process_clip_audio(
                    video_path, clip_index=clip_idx,
                    use_whisper=config.USE_WHISPER and config.USE_AUDIO and not debug,
                    debug=debug
                )
                save_audio_features(audio_result, patient_out_dir, clip_idx)
                audio_dirs.append(patient_out_dir)

    if not any(d for d in clip_dirs if d is not None):
        return None

    # 保持 clip_dirs 与 audio_dirs 长度对齐（用 "" 标记无人脸的片段）
    return {
        "patient_id": patient_id,
        "label": label,
        "clip_dirs": [d if d is not None else "" for d in clip_dirs],
        "audio_dirs": audio_dirs,
        "total_frames": total_frames
    }


def main():
    parser = argparse.ArgumentParser(description="谵妄识别数据预处理")
    parser.add_argument("--debug", action="store_true", help="强制调试模式（少量数据）")
    parser.add_argument("--full", action="store_true", help="强制全量模式")
    args = parser.parse_args()

    debug_mode = config.DEBUG_MODE
    if args.debug:
        debug_mode = True
    elif args.full:
        debug_mode = False

    print(f"预处理模式: {'调试（少量数据）' if debug_mode else '全量'}")
    print(f"抽帧率: {config.FRAME_RATE} fps")
    print(f"人脸尺寸: {config.FACE_SIZE}x{config.FACE_SIZE}")

    # 收集所有患者文件夹
    lucid_folders = get_patient_folders(config.LUCID_DIR, label=0)
    delirium_folders = get_patient_folders(config.DELIRIUM_DIR, label=1)

    print(f"清醒样本文件夹: {len(lucid_folders)}")
    print(f"谵妄样本文件夹: {len(delirium_folders)}")

    if debug_mode:
        n = config.DEBUG_SAMPLES_PER_CLASS
        lucid_folders = lucid_folders[:n]
        delirium_folders = delirium_folders[:n]
        print(f"调试模式：每类使用 {n} 个样本")

    all_folders = lucid_folders + delirium_folders

    # 处理所有患者
    face_detectors = create_face_detectors()
    manifest = []
    failed = 0

    for patient_dir, patient_name, label in tqdm(all_folders, desc="处理患者视频"):
        result = process_patient(patient_dir, patient_name, label, face_detectors, debug_mode)
        if result is None:
            failed += 1
            continue
        manifest.append(result)

    for det in face_detectors:
        det.close()

    # 保存数据清单CSV
    csv_path = os.path.join(config.PROCESSED_DIR, "manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "label", "num_clips", "total_frames", "clip_dirs", "audio_dirs"])
        for item in manifest:
            writer.writerow([
                item["patient_id"],
                item["label"],
                len(item["clip_dirs"]),
                item["total_frames"],
                "|".join(item["clip_dirs"]),
                "|".join(item.get("audio_dirs", []))
            ])

    # 统计
    lucid_count = sum(1 for m in manifest if m["label"] == 0)
    delirium_count = sum(1 for m in manifest if m["label"] == 1)
    total_frames = sum(m["total_frames"] for m in manifest)

    print(f"\n预处理完成:")
    print(f"  成功: {len(manifest)} 个患者 (清醒={lucid_count}, 谵妄={delirium_count})")
    print(f"  失败: {failed} 个患者")
    print(f"  总帧数: {total_frames}")
    print(f"  清单文件: {csv_path}")


if __name__ == "__main__":
    main()
