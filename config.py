"""全局配置文件"""
import os
import torch

# numba/llvmlite (loaded by librosa) conflicts with PyTorch MKL via LLVM thread pool.
# Limit OpenMP threads to 1 to prevent segfaults when both are loaded in the same process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "2025年新谵妄数据")
LUCID_DIR = os.path.join(DATA_DIR, "清醒")
DELIRIUM_DIR = os.path.join(DATA_DIR, "谵妄视频新")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ============ 设备配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

# ============ 调试模式 ============
# CPU测试时开启：仅用少量数据跑通流程
DEBUG_MODE = not torch.cuda.is_available()
DEBUG_SAMPLES_PER_CLASS = 50  # 调试模式下每类使用的样本数

# ============ 预处理参数 ============
FRAME_RATE = 2            # 抽帧率 (fps)
FACE_SIZE = 112           # 人脸裁剪后尺寸
MIN_FACE_CONFIDENCE = 0.5 # 人脸检测最低置信度
MAX_CLIPS = 4             # 每个患者最多片段数
MAX_FRAMES_PER_CLIP = 20  # 每个片段最大帧数（加速全量训练）

# ============ 模型参数 ============
# 外观流
APPEARANCE_DIM = 512      # ResNet18输出维度
# 动态流
MOTION_DIM = 128          # 动态流输出维度
# 融合后特征维度
FUSED_DIM = APPEARANCE_DIM + MOTION_DIM  # 640
# Transformer
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.3
# 分类头
HIDDEN_DIM = 256
NUM_CLASSES = 1           # 二分类 (sigmoid)

# ============ 音频参数 ============
AUDIO_SAMPLE_RATE = 16000
N_MFCC = 40
MAX_AUDIO_FRAMES = 200 if not DEBUG_MODE else 50
AUDIO_DIM = 128
SEMANTIC_DIM = 1
FUSED_DIM_MULTIMODAL = FUSED_DIM + AUDIO_DIM + SEMANTIC_DIM  # 769
WHISPER_MODEL = "small"
USE_AUDIO = True
# 是否启用ASR语义特征（开启后会显著增加预处理耗时）
USE_WHISPER = True

# ============ 训练参数 ============
BATCH_SIZE = 8 if torch.cuda.is_available() else 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50 if not DEBUG_MODE else 10
PATIENCE = 10             # 早停耐心值
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============ 正则化参数 ============
# Label smoothing：防止模型输出极端概率（0.0003/0.9997）
# 把 0/1 标签软化为 smoothing/2 和 1-smoothing/2
LABEL_SMOOTHING = 0.1
# Backbone 冻结：前 N epoch 只训练非 ResNet18 部分，防止快速过拟合
# 0 表示不冻结；建议 3~8
FREEZE_BACKBONE_EPOCHS = 5
# 解冻后对 backbone 使用更低的学习率倍数（0.1 = backbone LR = main LR × 0.1）
BACKBONE_LR_SCALE = 0.1
# Mixup 增强：在 batch 内插值样本对，提升泛化
USE_MIXUP = True
MIXUP_ALPHA = 0.3         # Beta 分布参数；越大插值越重

# ============ 创建必要目录 ============
for d in [PROCESSED_DIR, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
