"""
评估脚本：加载最优模型，在测试集上计算完整评估指标并生成可视化。

用法:
    python evaluate.py                          # 使用默认最优模型
    python evaluate.py --checkpoint path.pth    # 指定检查点
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, f1_score
)
from tqdm import tqdm

import config
from dataset import create_dataloaders
from model import DeliriumNet


def _ckpt_uses_audio(state_dict):
    return any(k.startswith("audio_stream.") for k in state_dict.keys())


def _sanitize_tag(tag):
    if not tag:
        return ""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag)


@torch.no_grad()
def predict_all(model, loader, device):
    """对所有样本进行预测"""
    model.eval()
    all_preds = []
    all_labels = []
    all_patient_ids = []

    for batch in tqdm(loader, desc="预测"):
        clips = batch["clips"].to(device)
        clip_lengths = batch["clip_lengths"].to(device)
        clip_mask = batch["clip_mask"].to(device)
        labels = batch["label"]
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
        probs = torch.sigmoid(logits).cpu().numpy()

        all_preds.extend(probs)
        all_labels.extend(labels.numpy())
        all_patient_ids.extend(batch["patient_id"])

    return np.array(all_preds), np.array(all_labels), all_patient_ids


def find_optimal_threshold(labels, preds):
    """通过Youden指数找最优分类阈值"""
    fpr, tpr, thresholds = roc_curve(labels, preds)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return thresholds[best_idx]


def compute_metrics(labels, preds, threshold=0.5):
    """计算完整评估指标"""
    pred_binary = (preds >= threshold).astype(int)

    tp = ((pred_binary == 1) & (labels == 1)).sum()
    tn = ((pred_binary == 0) & (labels == 0)).sum()
    fp = ((pred_binary == 1) & (labels == 0)).sum()
    fn = ((pred_binary == 0) & (labels == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率/灵敏度
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.0

    return {
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "threshold": threshold,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum())
    }


def plot_roc_curve(labels, preds, save_path):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - Delirium Detection', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC曲线已保存: {save_path}")


def plot_confusion_matrix(labels, preds, threshold, save_path):
    """绘制混淆矩阵"""
    pred_binary = (preds >= threshold).astype(int)
    cm = confusion_matrix(labels, pred_binary)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()

    classes = ['Lucid (0)', 'Delirium (1)']
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    # 在格子中显示数值
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center", color=color, fontsize=16)

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_prediction_distribution(labels, preds, save_path):
    """绘制预测概率分布"""
    plt.figure(figsize=(8, 5))
    plt.hist(preds[labels == 0], bins=30, alpha=0.6, color='#4CAF50', label='Lucid', density=True)
    plt.hist(preds[labels == 1], bins=30, alpha=0.6, color='#F44336', label='Delirium', density=True)
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Prediction Distribution', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"预测分布图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="谵妄识别模型评估")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--tag", type=str, default="", help="实验标签，用于读取/保存对应结果")
    args = parser.parse_args()

    tag = _sanitize_tag(args.tag)
    suffix = f"_{tag}" if tag else ""

    device = config.DEVICE
    print(f"设备: {device}")

    # 加载模型
    checkpoint_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, f"best_model{suffix}.pth")
    if not os.path.isfile(checkpoint_path):
        print(f"错误: 找不到模型检查点 {checkpoint_path}")
        print("请先运行 train.py 训练模型")
        return

    print(f"加载模型: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 恢复训练时的 max_frames 配置（避免 pos_embedding 尺寸不匹配）
    if "max_frames_per_clip" in ckpt:
        config.MAX_FRAMES_PER_CLIP = ckpt["max_frames_per_clip"]
    ckpt_use_audio = ckpt.get("use_audio")
    if ckpt_use_audio is None:
        ckpt_use_audio = _ckpt_uses_audio(ckpt["model_state_dict"])
    model = DeliriumNet(pretrained=False, use_audio=ckpt_use_audio).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"模型加载成功 (训练至epoch {ckpt['epoch'] + 1}, best AUC={ckpt['best_auc']:.4f}, max_frames={config.MAX_FRAMES_PER_CLIP})")

    # 加载测试数据
    print("\n加载数据...")
    _, _, test_loader, _ = create_dataloaders()

    # 预测
    print("\n在测试集上预测...")
    preds, labels, patient_ids = predict_all(model, test_loader, device)

    if len(labels) == 0:
        print("错误: 测试集为空")
        return

    # 找最优阈值
    optimal_threshold = find_optimal_threshold(labels, preds)
    print(f"\nYouden最优阈值: {optimal_threshold:.4f}")

    # 计算两个阈值下的指标
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")

    for thresh_name, thresh in [("0.5 (默认)", 0.5), (f"{optimal_threshold:.3f} (Youden最优)", optimal_threshold)]:
        metrics = compute_metrics(labels, preds, threshold=float(thresh_name.split()[0]) if thresh_name.startswith("0.5") else optimal_threshold)
        print(f"\n--- 阈值 = {thresh_name} ---")
        print(f"  AUC:         {metrics['auc']:.4f}")
        print(f"  准确率:      {metrics['accuracy']:.4f}")
        print(f"  灵敏度(召回): {metrics['sensitivity']:.4f}")
        print(f"  特异度:      {metrics['specificity']:.4f}")
        print(f"  精确率:      {metrics['precision']:.4f}")
        print(f"  F1:          {metrics['f1']:.4f}")
        print(f"  假阳性率:    {metrics['false_positive_rate']:.4f}")
        print(f"  假阴性率:    {metrics['false_negative_rate']:.4f}")
        print(f"  TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")

    # 保存评估结果
    eval_metrics = compute_metrics(labels, preds, optimal_threshold)
    eval_metrics["optimal_threshold"] = float(optimal_threshold)
    eval_metrics["tag"] = tag

    # 也保存默认阈值的指标
    default_metrics = compute_metrics(labels, preds, 0.5)
    eval_metrics["default_threshold_metrics"] = default_metrics

    eval_path = os.path.join(config.LOG_DIR, f"evaluation_results{suffix}.json")
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f, indent=2, default=lambda x: float(x))
    print(f"\n评估结果已保存: {eval_path}")

    # 生成可视化
    print("\n生成可视化...")
    vis_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    try:
        plot_roc_curve(labels, preds, os.path.join(vis_dir, f"roc_curve{suffix}.png"))
    except Exception as e:
        print(f"  ROC曲线绘制失败: {e}")

    try:
        plot_confusion_matrix(labels, preds, optimal_threshold,
                              os.path.join(vis_dir, f"confusion_matrix{suffix}.png"))
    except Exception as e:
        print(f"  混淆矩阵绘制失败: {e}")

    try:
        plot_prediction_distribution(labels, preds,
                                     os.path.join(vis_dir, f"prediction_distribution{suffix}.png"))
    except Exception as e:
        print(f"  预测分布图绘制失败: {e}")

    # 逐患者结果
    patient_results_path = os.path.join(config.LOG_DIR, f"patient_predictions{suffix}.csv")
    with open(patient_results_path, "w") as f:
        f.write("patient_id,true_label,predicted_prob,predicted_label\n")
        for pid, true_l, pred_p in zip(patient_ids, labels, preds):
            pred_l = 1 if pred_p >= optimal_threshold else 0
            f.write(f"{pid},{int(true_l)},{pred_p:.4f},{pred_l}\n")
    print(f"逐患者预测结果已保存: {patient_results_path}")

    print(f"\n{'='*60}")
    print("评估完成！")


if __name__ == "__main__":
    main()
