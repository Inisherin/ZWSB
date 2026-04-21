# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

研究项目：基于面部视频的神经重症患者谵妄识别（北京天坛医院，项目编号PX2023021）。使用双流时序深度网络（Dual-Stream Temporal Network）从患者面部视频自动识别谵妄状态。

## 运行环境

使用独立conda环境（`delirium`），避免与系统base环境的cv2冲突：
```bash
conda activate delirium
# 或用全路径: ~/anaconda3/envs/delirium/bin/python
```

## 常用命令

```bash
# 数据预处理（调试模式：每类10个样本）
python preprocess.py --debug

# 数据预处理（全量）
python preprocess.py --full

# 训练（CPU调试模式：3个epoch，20帧/clip）
python train.py --debug

# 训练（GPU全量）
python train.py

# 断点续训
python train.py --resume output/checkpoints/checkpoint_epoch5.pth

# 评估（生成ROC曲线、混淆矩阵等）
python evaluate.py

# 单患者推理
python predict.py --patient_dir "2025年新谵妄数据/谵妄视频新/患者文件夹/"
```

## 代码架构

```
config.py       全局配置：路径、超参数、DEBUG_MODE（无GPU时自动开启）
preprocess.py   预处理：抽帧(2fps) → MediaPipe人脸检测裁剪 → 保存112x112图像 + manifest.csv
dataset.py      PyTorch Dataset：患者级分层划分、数据增强、WeightedRandomSampler
model.py        双流时序网络：AppearanceStream(ResNet18) + MotionStream(帧差CNN) →
                TemporalTransformer → MultiClipFusion → 分类头
train.py        训练循环：加权BCE损失、AdamW+CosineScheduler、早停、断点续训
evaluate.py     评估：AUC、准确率、灵敏度、特异度 + ROC曲线/混淆矩阵可视化
predict.py      单样本推理：输入患者视频目录，输出谵妄概率
```

## 关键设计决策

- **患者级数据划分**（`dataset.py:split_dataset`）：同一患者的所有片段必须在同一集合，防止数据泄漏
- **多片段融合**（`model.py:MultiClipFusion`）：3-4个测试环节视频通过注意力加权聚合，注意力权重反映各片段的诊断贡献
- **CPU/GPU自适应**：`config.DEBUG_MODE = not torch.cuda.is_available()` 自动切换，CPU调试时减少帧数和样本量
- **类别不平衡处理**：训练时同时使用加权采样（WeightedRandomSampler）和加权损失（BCEWithLogitsLoss with pos_weight）

## 数据集结构

所有数据在 `2025年新谵妄数据/` 下：
- **`清醒/`** — 非谵妄患者（~653个文件夹）
- **`谵妄视频新/`** — 谵妄患者（~246个文件夹）

文件夹命名：`姓名_住院号_导出时间戳_含N次检查/`，内含3-4段MP4（注意力/思维/意识测试）

预处理后数据保存在 `processed_data/{lucid,delirium}/患者ID/clip_N/frame_XXXX.jpg`

## 注意事项

- 数据包含患者身份信息（姓名、住院号），需按医疗数据隐私规范处理
- 视频文件较大（7-34MB/个），不适合git版本控制，需git-lfs
- GPU训练建议：`BATCH_SIZE=8, MAX_FRAMES_PER_CLIP=80, NUM_EPOCHS=50`
- CPU调试：`BATCH_SIZE=2, MAX_FRAMES_PER_CLIP=20, NUM_EPOCHS=3`（通过 `--debug` 自动设置）
