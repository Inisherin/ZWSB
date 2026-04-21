"""
PyTorch数据集：加载预处理后的人脸帧序列，支持多片段、数据增强、加权采样。

关键设计：
- 患者级划分，防止数据泄漏
- 每个样本 = 一个患者的所有片段帧序列
- 支持类别不平衡的加权采样
"""
import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import config
from audio_processor import load_audio_features


class DeliriumDataset(Dataset):
    """谵妄识别数据集"""

    def __init__(self, patient_records, transform=None, max_clips=4, max_frames=80,
                 max_audio_frames=None, use_audio=True):
        """
        Args:
            patient_records: list of dict, 每个dict包含patient_id, label, clip_dirs, audio_dirs
            transform: 图像变换
            max_clips: 每个患者最多使用的片段数
            max_frames: 每个片段最多使用的帧数
            max_audio_frames: 每个片段最多音频帧数
            use_audio: 是否加载音频特征
        """
        self.records = patient_records
        self.transform = transform
        self.max_clips = max_clips
        self.max_frames = max_frames
        self.max_audio_frames = max_audio_frames or config.MAX_AUDIO_FRAMES
        self.use_audio = use_audio and config.USE_AUDIO
        self.n_mfcc_total = config.N_MFCC * 3  # MFCC + delta + delta-delta

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        label = record["label"]
        clip_dirs = record["clip_dirs"][:self.max_clips]
        audio_dirs = record.get("audio_dirs", [])[:self.max_clips]

        all_clips = []
        clip_lengths = []
        clip_mask = []
        all_audio = []
        semantic_scores = []
        audio_mask = []

        # clip_dirs and audio_dirs are aligned (same length, "" means no face detected)
        for slot_idx, (clip_dir, audio_dir) in enumerate(zip(clip_dirs, audio_dirs)):
            has_visual = bool(clip_dir)
            frames = self._load_clip_frames(clip_dir) if has_visual else []

            if len(frames) > 0:
                clip_tensor = torch.stack(frames)
                T = clip_tensor.shape[0]
                if T > self.max_frames:
                    clip_tensor = clip_tensor[:self.max_frames]
                    T = self.max_frames
                clip_lengths.append(T)
                if T < self.max_frames:
                    pad = torch.zeros(self.max_frames - T, *clip_tensor.shape[1:])
                    clip_tensor = torch.cat([clip_tensor, pad], dim=0)
                all_clips.append(clip_tensor)
                clip_mask.append(1)
            else:
                C, H, W = 3, config.FACE_SIZE, config.FACE_SIZE
                all_clips.append(torch.zeros(self.max_frames, C, H, W))
                clip_lengths.append(0)
                clip_mask.append(0)

            # 音频特征（与视觉对齐，用 slot_idx 索引 audio 文件）
            if self.use_audio and audio_dir:
                mfcc, asr_data = load_audio_features(
                    audio_dir, slot_idx, max_frames=self.max_audio_frames
                )
            else:
                mfcc, asr_data = None, {"semantic_score": 0.5, "no_speech_prob": 1.0}

            if mfcc is not None:
                audio_t = torch.tensor(mfcc, dtype=torch.float32)
                Ta = audio_t.shape[0]
                if Ta < self.max_audio_frames:
                    pad = torch.zeros(self.max_audio_frames - Ta, self.n_mfcc_total)
                    audio_t = torch.cat([audio_t, pad], dim=0)
                all_audio.append(audio_t)
                audio_mask.append(1)
            else:
                all_audio.append(torch.zeros(self.max_audio_frames, self.n_mfcc_total))
                audio_mask.append(0)

            semantic_scores.append(float(asr_data.get("semantic_score", 0.5)))

        # 填充到 max_clips（当视频总片段数不足时）
        C, H, W = 3, config.FACE_SIZE, config.FACE_SIZE
        while len(all_clips) < self.max_clips:
            all_clips.append(torch.zeros(self.max_frames, C, H, W))
            clip_lengths.append(0)
            clip_mask.append(0)
            all_audio.append(torch.zeros(self.max_audio_frames, self.n_mfcc_total))
            semantic_scores.append(0.5)
            audio_mask.append(0)

        clips_tensor = torch.stack(all_clips)
        lengths_tensor = torch.tensor(clip_lengths, dtype=torch.long)
        mask_tensor = torch.tensor(clip_mask, dtype=torch.float)
        audio_tensor = torch.stack(all_audio)                           # [max_clips, T_audio, N_MFCC*3]
        semantic_tensor = torch.tensor(semantic_scores, dtype=torch.float).unsqueeze(-1)  # [max_clips, 1]
        audio_mask_tensor = torch.tensor(audio_mask, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return {
            "clips": clips_tensor,
            "clip_lengths": lengths_tensor,
            "clip_mask": mask_tensor,
            "audio_features": audio_tensor,
            "semantic_scores": semantic_tensor,
            "audio_mask": audio_mask_tensor,
            "label": label_tensor,
            "patient_id": record["patient_id"]
        }

    def _load_clip_frames(self, clip_dir):
        """加载一个片段目录下的所有帧图像"""
        if not os.path.isdir(clip_dir):
            return []
        frame_files = sorted([
            f for f in os.listdir(clip_dir) if f.endswith('.jpg')
        ])
        frames = []
        for ff in frame_files:
            img_path = os.path.join(clip_dir, ff)
            try:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
                frames.append(img)
            except Exception:
                continue
        return frames


def load_manifest(csv_path):
    """从CSV清单加载患者记录"""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "patient_id": row["patient_id"],
                "label": int(row["label"]),
                "clip_dirs": [d for d in row["clip_dirs"].split("|")] if row.get("clip_dirs") else [],
                "audio_dirs": row["audio_dirs"].split("|") if row.get("audio_dirs") else [],
                "num_clips": int(row["num_clips"]),
                "total_frames": int(row["total_frames"])
            })
    return records


def split_dataset(records, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    患者级分层划分数据集。
    确保同一患者不会出现在不同集合中。
    """
    labels = [r["label"] for r in records]
    indices = list(range(len(records)))

    # 先分出测试集
    train_val_idx, test_idx = train_test_split(
        indices, test_size=1.0 - train_ratio - val_ratio,
        stratify=labels, random_state=seed
    )
    # 再从训练+验证中分出验证集
    train_val_labels = [labels[i] for i in train_val_idx]
    val_frac = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_frac,
        stratify=train_val_labels, random_state=seed
    )

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    return train_records, val_records, test_records


def get_transforms(is_train=True):
    """获取数据增强变换"""
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_weighted_sampler(records):
    """为类别不平衡创建加权采样器"""
    labels = [r["label"] for r in records]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(records),
        replacement=True
    )


def create_dataloaders(manifest_csv=None):
    """
    创建训练、验证、测试DataLoader。

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    if manifest_csv is None:
        manifest_csv = os.path.join(config.PROCESSED_DIR, "manifest.csv")

    records = load_manifest(manifest_csv)
    print(f"加载 {len(records)} 个患者记录")

    train_records, val_records, test_records = split_dataset(
        records,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        seed=config.RANDOM_SEED
    )

    train_labels = [r["label"] for r in train_records]
    n_lucid = train_labels.count(0)
    n_delirium = train_labels.count(1)
    print(f"训练集: {len(train_records)} (清醒={n_lucid}, 谵妄={n_delirium})")
    print(f"验证集: {len(val_records)}")
    print(f"测试集: {len(test_records)}")

    # 计算类别权重用于损失函数
    total = n_lucid + n_delirium
    class_weights = torch.tensor([total / (2 * n_lucid), total / (2 * n_delirium)],
                                 dtype=torch.float)

    train_dataset = DeliriumDataset(
        train_records,
        transform=get_transforms(is_train=True),
        max_clips=config.MAX_CLIPS,
        max_frames=config.MAX_FRAMES_PER_CLIP,
        max_audio_frames=config.MAX_AUDIO_FRAMES,
        use_audio=config.USE_AUDIO
    )
    val_dataset = DeliriumDataset(
        val_records,
        transform=get_transforms(is_train=False),
        max_clips=config.MAX_CLIPS,
        max_frames=config.MAX_FRAMES_PER_CLIP,
        max_audio_frames=config.MAX_AUDIO_FRAMES,
        use_audio=config.USE_AUDIO
    )
    test_dataset = DeliriumDataset(
        test_records,
        transform=get_transforms(is_train=False),
        max_clips=config.MAX_CLIPS,
        max_frames=config.MAX_FRAMES_PER_CLIP,
        max_audio_frames=config.MAX_AUDIO_FRAMES,
        use_audio=config.USE_AUDIO
    )

    sampler = get_weighted_sampler(train_records)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        sampler=sampler, num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_weights
