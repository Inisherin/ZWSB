"""
训练脚本：支持GPU/CPU、加权损失、早停、断点续训。

用法:
    python train.py                # 自动检测GPU/CPU
    python train.py --debug        # 强制调试模式
    python train.py --resume checkpoint.pth  # 断点续训
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config
from dataset import create_dataloaders
from model import DeliriumNet, count_parameters, freeze_backbone, unfreeze_backbone


class SmoothedBCELoss(nn.Module):
    """BCEWithLogitsLoss + label smoothing，防止模型输出极端 logit。

    将标签从 {0, 1} 软化为 {s/2, 1-s/2}，使得最优 logit 有界（约 ±log((1-s)/s)）。
    """
    def __init__(self, pos_weight=None, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def mixup_batch(clips, labels, alpha):
    """在 batch 内对 clips 和 labels 做 Mixup 插值。"""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(clips.shape[0], device=clips.device)
    mixed_clips = lam * clips + (1.0 - lam) * clips[idx]
    mixed_labels = lam * labels + (1.0 - lam) * labels[idx]
    return mixed_clips, mixed_labels


def _parse_bool_flag(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {v}")


def _sanitize_tag(tag):
    if not tag:
        return ""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag)


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=False):
    """训练一个epoch，支持 Mixup 增强。"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="  训练", leave=False):
        clips = batch["clips"].to(device)
        clip_lengths = batch["clip_lengths"].to(device)
        clip_mask = batch["clip_mask"].to(device)
        labels = batch["label"].to(device)
        audio_features = batch.get("audio_features")
        semantic_scores = batch.get("semantic_scores")
        semantic_confidences = batch.get("semantic_confidences")
        audio_mask = batch.get("audio_mask")
        if audio_features is not None:
            audio_features = audio_features.to(device)
            semantic_scores = semantic_scores.to(device)
            semantic_confidences = semantic_confidences.to(device)
            audio_mask = audio_mask.to(device)

        # Mixup：仅对 clips 和 labels 插值（batch_size > 1 才有意义）
        if use_mixup and clips.shape[0] > 1 and config.MIXUP_ALPHA > 0:
            clips, labels = mixup_batch(clips, labels, config.MIXUP_ALPHA)

        optimizer.zero_grad()
        logits = model(clips, clip_lengths, clip_mask,
                       audio_features, semantic_scores,
                       semantic_confidences, audio_mask).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        # 记录 sigmoid 概率用于 AUC（mixup label 可能是小数，不影响排序）
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    try:
        auc = roc_auc_score((np.array(all_labels) >= 0.5).astype(int), all_preds)
    except ValueError:
        auc = 0.0

    return avg_loss, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="  评估", leave=False):
        clips = batch["clips"].to(device)
        clip_lengths = batch["clip_lengths"].to(device)
        clip_mask = batch["clip_mask"].to(device)
        labels = batch["label"].to(device)
        audio_features = batch.get("audio_features")
        semantic_scores = batch.get("semantic_scores")
        semantic_confidences = batch.get("semantic_confidences")
        audio_mask = batch.get("audio_mask")
        if audio_features is not None:
            audio_features = audio_features.to(device)
            semantic_scores = semantic_scores.to(device)
            semantic_confidences = semantic_confidences.to(device)
            audio_mask = audio_mask.to(device)

        logits = model(clips, clip_lengths, clip_mask,
                       audio_features, semantic_scores,
                       semantic_confidences, audio_mask).squeeze(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.0

    return avg_loss, auc, np.array(all_preds), np.array(all_labels)


def save_checkpoint(model, optimizer, scheduler, epoch, best_auc, path):
    """保存检查点（包含模型配置，避免加载时尺寸不匹配）"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_auc": best_auc,
        "max_frames_per_clip": config.MAX_FRAMES_PER_CLIP,
        "use_audio": bool(getattr(model, "use_audio", config.USE_AUDIO)),
        "dropout": float(config.DROPOUT),
    }, path)


def _ckpt_uses_audio(state_dict):
    """通过state_dict判断检查点是否包含音频分支"""
    return any(k.startswith("audio_stream.") for k in state_dict.keys())


def main():
    parser = argparse.ArgumentParser(description="谵妄识别模型训练")
    parser.add_argument("--debug", action="store_true", help="强制调试模式")
    parser.add_argument("--resume", type=str, default=None, help="断点续训的检查点路径")
    parser.add_argument("--tag", type=str, default="", help="实验标签，用于区分日志与检查点")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖批大小")
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--weight-decay", type=float, default=None, help="覆盖权重衰减")
    parser.add_argument("--num-epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--patience", type=int, default=None, help="覆盖早停耐心值")
    parser.add_argument("--dropout", type=float, default=None, help="覆盖模型dropout")
    parser.add_argument("--max-frames", type=int, default=None, help="覆盖每片段最大帧数")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument("--use-audio", type=_parse_bool_flag, default=None,
                        help="是否使用音频分支(true/false)")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="覆盖 label smoothing 系数 (0=关闭)")
    parser.add_argument("--freeze-epochs", type=int, default=None,
                        help="覆盖 backbone 冻结 epoch 数 (0=不冻结)")
    parser.add_argument("--no-mixup", action="store_true",
                        help="关闭 Mixup 增强")
    args = parser.parse_args()

    tag = _sanitize_tag(args.tag)
    suffix = f"_{tag}" if tag else ""

    if args.debug:
        config.DEBUG_MODE = True
        config.NUM_EPOCHS = 3
        config.BATCH_SIZE = 2
        config.MAX_FRAMES_PER_CLIP = 20  # CPU测试时减少帧数加快速度

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.weight_decay is not None:
        config.WEIGHT_DECAY = args.weight_decay
    if args.num_epochs is not None:
        config.NUM_EPOCHS = args.num_epochs
    if args.patience is not None:
        config.PATIENCE = args.patience
    if args.dropout is not None:
        config.DROPOUT = args.dropout
    if args.max_frames is not None:
        config.MAX_FRAMES_PER_CLIP = args.max_frames
    if args.seed is not None:
        config.RANDOM_SEED = args.seed
    if args.use_audio is not None:
        config.USE_AUDIO = args.use_audio
    if args.label_smoothing is not None:
        config.LABEL_SMOOTHING = args.label_smoothing
    if args.freeze_epochs is not None:
        config.FREEZE_BACKBONE_EPOCHS = args.freeze_epochs
    if args.no_mixup:
        config.USE_MIXUP = False

    device = config.DEVICE
    print(f"设备: {device}")
    print(f"模式: {'调试' if config.DEBUG_MODE else '正式训练'}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"LR: {config.LEARNING_RATE}")
    print(f"Weight Decay: {config.WEIGHT_DECAY}")
    print(f"Dropout: {config.DROPOUT}")
    print(f"Patience: {config.PATIENCE}")
    print(f"Use Audio: {config.USE_AUDIO}")
    print(f"Label Smoothing: {config.LABEL_SMOOTHING}")
    print(f"Freeze Backbone Epochs: {config.FREEZE_BACKBONE_EPOCHS}")
    print(f"Mixup: {config.USE_MIXUP} (alpha={config.MIXUP_ALPHA})")
    if tag:
        print(f"Tag: {tag}")

    # 创建数据加载器
    print("\n加载数据...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders()

    # 创建模型
    print("\n创建模型...")
    model = DeliriumNet(pretrained=True).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 损失函数：Label Smoothing + 类别权重
    pos_weight = class_weights[1] / class_weights[0]
    criterion = SmoothedBCELoss(
        pos_weight=pos_weight.to(device),
        smoothing=config.LABEL_SMOOTHING
    )
    print(f"正样本权重: {pos_weight:.2f}, Label Smoothing: {config.LABEL_SMOOTHING}")

    # Backbone 冻结（前 FREEZE_BACKBONE_EPOCHS epoch 只训练非 ResNet 部分）
    freeze_epochs = config.FREEZE_BACKBONE_EPOCHS
    if freeze_epochs > 0:
        freeze_backbone(model)
        print(f"Backbone 已冻结（将在 epoch {freeze_epochs+1} 解冻）")

    # 优化器：backbone 和其余部分使用不同学习率
    backbone_params = list(model.appearance_stream.features.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = AdamW([
        {"params": backbone_params, "lr": config.LEARNING_RATE * config.BACKBONE_LR_SCALE},
        {"params": other_params,    "lr": config.LEARNING_RATE},
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    # 断点续训
    start_epoch = 0
    best_auc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"\n从检查点恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt["best_auc"]
        print(f"从 epoch {start_epoch} 继续, best AUC = {best_auc:.4f}")
        # 如果续训时还在冻结阶段，重新冻结 backbone
        if freeze_epochs > 0 and start_epoch < freeze_epochs:
            freeze_backbone(model)
            print(f"  续训：Backbone 仍处于冻结阶段（将在 epoch {freeze_epochs+1} 解冻）")

    # 训练循环
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": [], "lr": []}

    print(f"\n开始训练...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        t0 = time.time()

        # Backbone 解冻：到达指定 epoch 后恢复 backbone 参数的训练
        if freeze_epochs > 0 and epoch == freeze_epochs:
            unfreeze_backbone(model)
            # 重置 backbone 参数组的学习率（已在优化器中配置为低倍率）
            print(f"  ★ Epoch {epoch+1}: Backbone 已解冻 (LR × {config.BACKBONE_LR_SCALE})")

        current_lr = optimizer.param_groups[1]["lr"]  # 主参数组 LR

        # 训练（backbone 冻结阶段不用 mixup，解冻后才有意义）
        use_mixup = config.USE_MIXUP and (epoch >= freeze_epochs)
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=use_mixup
        )

        # 验证
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS} ({elapsed:.1f}s) | "
              f"LR={current_lr:.6f} | "
              f"Train Loss={train_loss:.4f} AUC={train_auc:.4f} | "
              f"Val Loss={val_loss:.4f} AUC={val_auc:.4f}")

        # 保存最优模型
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_path = os.path.join(config.CHECKPOINT_DIR, f"best_model{suffix}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc, best_path)
            print(f"  ★ 新最优模型已保存 (AUC={best_auc:.4f})")
        else:
            patience_counter += 1

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch{epoch + 1}{suffix}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc, ckpt_path)

        # 早停
        if patience_counter >= config.PATIENCE:
            print(f"\n早停触发 (验证AUC连续{config.PATIENCE}个epoch未提升)")
            break

    # 保存训练历史
    history_path = os.path.join(config.LOG_DIR, f"training_history{suffix}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n训练历史已保存: {history_path}")

    # 在测试集上评估最优模型
    print(f"\n加载最优模型评估测试集...")
    best_path = os.path.join(config.CHECKPOINT_DIR, f"best_model{suffix}.pth")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        ckpt_use_audio = ckpt.get("use_audio", _ckpt_uses_audio(ckpt["model_state_dict"]))
        if getattr(model, "use_audio", None) != ckpt_use_audio:
            model = DeliriumNet(pretrained=False, use_audio=ckpt_use_audio).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_auc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    # 计算最优阈值下的指标
    best_threshold = 0.5
    pred_binary = (test_preds >= best_threshold).astype(int)
    tp = ((pred_binary == 1) & (test_labels == 1)).sum()
    tn = ((pred_binary == 0) & (test_labels == 0)).sum()
    fp = ((pred_binary == 1) & (test_labels == 0)).sum()
    fn = ((pred_binary == 0) & (test_labels == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n{'='*50}")
    print(f"测试集结果:")
    print(f"  AUC:       {test_auc:.4f}")
    print(f"  准确率:    {accuracy:.4f}")
    print(f"  灵敏度:    {sensitivity:.4f}")
    print(f"  特异度:    {specificity:.4f}")
    print(f"{'='*50}")

    # 保存测试结果
    results = {
        "test_auc": float(test_auc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "threshold": best_threshold,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "tag": tag,
        "learning_rate": float(config.LEARNING_RATE),
        "weight_decay": float(config.WEIGHT_DECAY),
        "batch_size": int(config.BATCH_SIZE),
        "dropout": float(config.DROPOUT),
        "patience": int(config.PATIENCE),
        "use_audio": bool(config.USE_AUDIO),
        "seed": int(config.RANDOM_SEED),
    }
    results_path = os.path.join(config.LOG_DIR, f"test_results{suffix}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"测试结果已保存: {results_path}")


if __name__ == "__main__":
    main()
