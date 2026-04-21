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
from model import DeliriumNet, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
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
        audio_mask = batch.get("audio_mask")
        if audio_features is not None:
            audio_features = audio_features.to(device)
            semantic_scores = semantic_scores.to(device)
            audio_mask = audio_mask.to(device)

        optimizer.zero_grad()
        logits = model(clips, clip_lengths, clip_mask,
                       audio_features, semantic_scores, audio_mask).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
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
        audio_mask = batch.get("audio_mask")
        if audio_features is not None:
            audio_features = audio_features.to(device)
            semantic_scores = semantic_scores.to(device)
            audio_mask = audio_mask.to(device)

        logits = model(clips, clip_lengths, clip_mask,
                       audio_features, semantic_scores, audio_mask).squeeze(-1)
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
    }, path)


def main():
    parser = argparse.ArgumentParser(description="谵妄识别模型训练")
    parser.add_argument("--debug", action="store_true", help="强制调试模式")
    parser.add_argument("--resume", type=str, default=None, help="断点续训的检查点路径")
    args = parser.parse_args()

    if args.debug:
        config.DEBUG_MODE = True
        config.NUM_EPOCHS = 3
        config.BATCH_SIZE = 2
        config.MAX_FRAMES_PER_CLIP = 20  # CPU测试时减少帧数加快速度

    device = config.DEVICE
    print(f"设备: {device}")
    print(f"模式: {'调试' if config.DEBUG_MODE else '正式训练'}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")

    # 创建数据加载器
    print("\n加载数据...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders()

    # 创建模型
    print("\n创建模型...")
    model = DeliriumNet(pretrained=True).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 损失函数：使用类别权重应对不平衡
    # BCEWithLogitsLoss的pos_weight = 正样本权重/负样本权重
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    print(f"正样本权重: {pos_weight:.2f}")

    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    # 断点续训
    start_epoch = 0
    best_auc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"\n从检查点恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt["best_auc"]
        print(f"从 epoch {start_epoch} 继续, best AUC = {best_auc:.4f}")

    # 训练循环
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": [], "lr": []}

    print(f"\n开始训练...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # 训练
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
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
            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc, best_path)
            print(f"  ★ 新最优模型已保存 (AUC={best_auc:.4f})")
        else:
            patience_counter += 1

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch{epoch + 1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc, ckpt_path)

        # 早停
        if patience_counter >= config.PATIENCE:
            print(f"\n早停触发 (验证AUC连续{config.PATIENCE}个epoch未提升)")
            break

    # 保存训练历史
    history_path = os.path.join(config.LOG_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n训练历史已保存: {history_path}")

    # 在测试集上评估最优模型
    print(f"\n加载最优模型评估测试集...")
    best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
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
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }
    results_path = os.path.join(config.LOG_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"测试结果已保存: {results_path}")


if __name__ == "__main__":
    main()
