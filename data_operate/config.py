from pathlib import Path
import numpy as np

# ================= 项目统一配置 =================

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATA_DIR = DATASETS_DIR / "original" / "data"
DATA_5FOLD_DIR = DATASETS_DIR / "original" / "data_5fold"
MODEL_BEST_LAST_DIR = PROJECT_ROOT / "model_best_last"
TRAIN_VAL_RESULT_DIR = PROJECT_ROOT / "train_val_result"
TEST_RESULT_DIR = PROJECT_ROOT / "test_result"

# 随机种子配置
RANDOM_SEED = 42
FOLD_SEEDS = [42, 123, 2026, 3407, 777]
SPLIT_SEED = RANDOM_SEED
VAL_SPLIT_SEED_OFFSET = 1000

# 训练配置
EPOCHS_PER_RUN = 50
BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_CLASSES = 2
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
LR_ETA_MIN = 1e-7
USE_GLOBAL_NORM = True
USE_DETERMINISTIC_ALGORITHMS = False
AMP_ENABLED = True

# 数据配置
VOLUME_SHAPE = (296, 37, 37)
RAW_DTYPE = np.uint8
N_SPLITS = 5
CLASS_NAMES = ("normal", "defective")
CLASS_TO_IDX = {"normal": 0, "defective": 1}
DISPLAY_LABELS = ("Normal", "Defective")
SPLIT_NAMES = ("train", "val", "test")
VAL_SPLIT_RATIO = 0.25

# 测试配置
TEST_THRESHOLD = 0.5

# 实验输出配置
RESNET18_CE_EXPERIMENT = "resnet18_5fold"
RESNET18_MULTI_SCALE_2_EXPERIMENT = "resnet18_multi_scale_2_5fold"
RESNET18_NO_MAXPOOL_EXPERIMENT = "resnet18_no_maxpool_5fold"
RESNET18_MAXPOOL_STRIDE1_EXPERIMENT = "resnet18_maxpool_stride1_5fold"
RESNET18_LEARNABLE_DOWNSAMPLE_EXPERIMENT = "resnet18_learnable_downsample_5fold"
RESNET18_LEARNABLE_KEEP148_EXPERIMENT = "resnet18_learnable_keep148_5fold"
RESNET18_LAYER2_KEEP37_EXPERIMENT = "resnet18_layer2_keep37_5fold"
RESNET18_AVG_TOPK_POOL_EXPERIMENT = "resnet18_avg_topk_pool_5fold"
RESNET18_CE_MICRO_DEFECT_SAFE_EXPERIMENT = f"{RESNET18_CE_EXPERIMENT}_micro_defect_safe"
RESNET18_MULTI_SCALE_2_MICRO_DEFECT_SAFE_EXPERIMENT = f"{RESNET18_MULTI_SCALE_2_EXPERIMENT}_micro_defect_safe"
RESNET18_CE_MODEL_FILES = {
    "best_f1": "best_resnet18_3d.pth",
    "best_loss": "best_loss_resnet18_3d.pth",
    "last": "last_resnet18_3d.pth",
}
RESNET18_NO_MAXPOOL_MODEL_FILES = {
    "best_f1": "best_resnet18_no_maxpool_3d.pth",
    "best_loss": "best_loss_resnet18_no_maxpool_3d.pth",
    "last": "last_resnet18_no_maxpool_3d.pth",
}
RESNET18_MAXPOOL_STRIDE1_MODEL_FILES = {
    "best_f1": "best_resnet18_maxpool_stride1_3d.pth",
    "best_loss": "best_loss_resnet18_maxpool_stride1_3d.pth",
    "last": "last_resnet18_maxpool_stride1_3d.pth",
}
RESNET18_LEARNABLE_DOWNSAMPLE_MODEL_FILES = {
    "best_f1": "best_resnet18_learnable_downsample_3d.pth",
    "best_loss": "best_loss_resnet18_learnable_downsample_3d.pth",
    "last": "last_resnet18_learnable_downsample_3d.pth",
}
RESNET18_LEARNABLE_KEEP148_MODEL_FILES = {
    "best_f1": "best_resnet18_learnable_keep148_3d.pth",
    "best_loss": "best_loss_resnet18_learnable_keep148_3d.pth",
    "last": "last_resnet18_learnable_keep148_3d.pth",
}
RESNET18_LAYER2_KEEP37_MODEL_FILES = {
    "best_f1": "best_resnet18_layer2_keep37_3d.pth",
    "best_loss": "best_loss_resnet18_layer2_keep37_3d.pth",
    "last": "last_resnet18_layer2_keep37_3d.pth",
}
RESNET18_AVG_TOPK_POOL_MODEL_FILES = {
    "best_f1": "best_resnet18_avg_topk_pool_3d.pth",
    "best_loss": "best_loss_resnet18_avg_topk_pool_3d.pth",
    "last": "last_resnet18_avg_topk_pool_3d.pth",
}
RESNET18_MULTI_SCALE_2_MODEL_FILES = {
    "best_f1": "best_resnet18_multi_scale_2_3d.pth",
    "best_loss": "best_loss_resnet18_multi_scale_2_3d.pth",
    "last": "last_resnet18_multi_scale_2_3d.pth",
}
