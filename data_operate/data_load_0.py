import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 兼容直接运行和被导入两种情况
try:
    # 当被其他模块导入时（如从test目录导入）
    from data_operate.config import BATCH_SIZE, CLASS_NAMES, CLASS_TO_IDX, DATA_DIR, NUM_WORKERS, RANDOM_SEED, RAW_DTYPE, USE_GLOBAL_NORM, VOLUME_SHAPE
except ImportError:
    # 当直接运行data_load_0.py时
    from config import BATCH_SIZE, CLASS_NAMES, CLASS_TO_IDX, DATA_DIR, NUM_WORKERS, RANDOM_SEED, RAW_DTYPE, USE_GLOBAL_NORM, VOLUME_SHAPE

class RawCTDataset(Dataset):
    def __init__(self, root_dir, shape=VOLUME_SHAPE, dtype=RAW_DTYPE, augment=False, global_min=None, global_max=None):
        self.root_dir = root_dir
        self.shape = shape
        self.dtype = dtype
        self.augment = augment  # 是否启用数据增强
        self.global_min = global_min
        self.global_max = global_max

        self.classes = list(CLASS_NAMES)
        self.class_to_idx = dict(CLASS_TO_IDX)

        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            if not os.path.isdir(class_dir):
                print(f"警告: 文件夹 {class_dir} 不存在")
                continue
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    if fname.lower().endswith('.raw'):
                        self.samples.append((os.path.join(root, fname), self.class_to_idx[target_class]))

        print(f"从 {root_dir} 加载了 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        # 读取数据
        data = np.fromfile(path, dtype=self.dtype).reshape(self.shape)
        data = torch.from_numpy(data).float().unsqueeze(0)

        # --- 1. 几何增强 (翻转 + 旋转) - 在归一化之前 ---
        if self.augment:
            # 策略：统一逻辑，不再区分 target 0 和 1
            # 目的：保证缺陷特征不被破坏，同时增加数据多样性

            # 1. 随机翻转 (仅限水平 / 垂直，严禁深度翻转)
            if torch.rand(1).item() > 0.5:
                # 从 dim=2(高度), dim=3(宽度) 中随机选一个
                # randint(2, 4) 会随机生成 2 或 3
                flip_dim = torch.randint(2, 4, (1,)).item()
                data = torch.flip(data, dims=[flip_dim])

            # 2. 随机旋转 (仅在H-W平面)
            if torch.rand(1).item() > 0.5:
                # 随机旋转 90, 180, 或 270 度 (k=1, 2, 3)
                k = torch.randint(1, 4, (1,)).item()
                data = torch.rot90(data, k=k, dims=[2, 3])
        # 在 import 区加入: import torchvision.transforms.functional as TF

        # --- 在原有的几何增强下方补充 ---
        if self.augment and torch.rand(1).item() > 0.5:
            # 在 X 和 Y 方向随机平移 -3 到 +3 个像素 (约占 37 的 10% 以内)
            shift_y = torch.randint(-3, 4, (1,)).item()
            shift_x = torch.randint(-3, 4, (1,)).item()

            # affine 变换针对的是 (C, H, W)，所以我们需要在深度维度上循环或者reshape
            # 对于 3D 数据，最简单的办法是使用 torch.roll 进行平移 (超出边界的会循环滚回来)
            # 或者用 zero padding 再 crop (更严谨，边缘填 0)

            # 这里提供严谨的 Padding 做法 (边缘补 0)：
            pad_y1 = max(0, shift_y)
            pad_y2 = max(0, -shift_y)
            pad_x1 = max(0, shift_x)
            pad_x2 = max(0, -shift_x)

            # 填充 (左, 右, 上, 下, 前, 后) -> 注意 PyTorch padding 是从最后面的维度开始的
            # 对于 (1, 296, 37, 37)，最后两维是 X(宽) 和 Y(高)
            padded = torch.nn.functional.pad(data, (pad_x1, pad_x2, pad_y1, pad_y2), mode='constant', value=0)
            # 截取回原来的 37x37 尺寸
            start_y = pad_y2
            start_x = pad_x2
            data = padded[:, :, start_y:start_y + 37, start_x:start_x + 37]

        # --- 2. 归一化 ---
        if self.global_min is not None and self.global_max is not None:
            # 使用全局统计量（仅从训练集计算，避免数据泄露）
            data = torch.clamp((data - self.global_min) / (self.global_max - self.global_min + 1e-6), 0.0, 1.0)
        else:
            # 样本级归一化（仅在无全局统计量时使用）
            data = (data - data.min()) / (data.max() - data.min() + 1e-6)
    
        # --- 3. 像素级增强：噪声、亮度变化、对比度调整 - 在归一化之后 ---
        if self.augment:
            # 1. 随机添加高斯噪声 (强度增强)
            if torch.rand(1).item() > 0.5:
                noise_std = 0.01 + torch.rand(1).item() * 0.02  # 随机噪声强度 [0.01, 0.03]
                noise = torch.randn_like(data) * noise_std
                data = data + noise
                data = torch.clamp(data, 0.0, 1.0)

            # 2. 随机亮度调整 (范围扩大)
            if torch.rand(1).item() > 0.5:
                brightness_factor = 0.9 + torch.rand(1).item() * 0.2  # 亮度因子 [0.9, 1.1]
                data = data * brightness_factor
                data = torch.clamp(data, 0.0, 1.0)

            # 3. 随机对比度调整
            if torch.rand(1).item() > 0.5:
                contrast_factor = 0.9 + torch.rand(1).item() * 0.2  # 对比度因子 [0.9, 1.1]
                # 对比度调整：以0.5为中心进行缩放
                data = (data - 0.5) * contrast_factor + 0.5
                data = torch.clamp(data, 0.0, 1.0)

        return data, target


def compute_global_stats(data_root, shape=VOLUME_SHAPE, dtype=RAW_DTYPE):
    """计算全局归一化统计量 (min, max) - 仅使用训练集避免数据泄露"""
    print("\n计算全局归一化统计量...")
    train_dir = os.path.join(data_root, 'train')
    global_min, global_max, total_samples = float('inf'), float('-inf'), 0

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for root, _, fnames in sorted(os.walk(class_dir)):
            for fname in sorted(fnames):
                if fname.lower().endswith('.raw'):
                    data = np.fromfile(os.path.join(root, fname), dtype=dtype).reshape(shape)
                    global_min = min(global_min, data.min())
                    global_max = max(global_max, data.max())
                    total_samples += 1
                    if total_samples % 50 == 0:
                        print(f" 已处理 {total_samples} 个样本, 当前全局范围: [{global_min}, {global_max}]")

    print(f"\n全局归一化统计量计算完成!\n 总样本数: {total_samples}\n 全局范围: [{global_min}, {global_max}]\n")
    return global_min, global_max


def worker_init_fn(worker_id):
    """为每个worker设置随机种子,确保可复现性"""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # 组合基础种子和worker ID
        seed = worker_info.seed % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        # 当 num_workers=0 时,在主进程中也需要设置种子
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)


def get_data_loaders(data_root, shape=VOLUME_SHAPE, dtype=RAW_DTYPE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, use_global_norm=USE_GLOBAL_NORM, seed=RANDOM_SEED):
    """创建训练集和验证集的 DataLoader"""
    train_dir, val_dir = os.path.join(data_root, 'train'), os.path.join(data_root, 'val')
    global_min, global_max = (None, None)

    if use_global_norm:
        global_min, global_max = compute_global_stats(data_root, shape, dtype)
        print(f"使用全局归一化: min={global_min}, max={global_max}")
    else:
        print("使用样本级归一化（每个样本独立归一化）")

    # 训练集开启增强
    train_dataset = RawCTDataset(train_dir, shape, dtype, augment=True, global_min=global_min, global_max=global_max)
    # 验证集关闭增强
    val_dataset = RawCTDataset(val_dir, shape, dtype, augment=False, global_min=global_min, global_max=global_max)

    return (
        DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(seed)  # 关键：固定shuffle的种子
        ),
        DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )

    )


# --- 使用示例 ---
if __name__ == "__main__":
    data_path = str(DATA_DIR)
    print(f"数据路径: {data_path}")
    train_loader, val_loader = get_data_loaders(data_path, batch_size=BATCH_SIZE)
    for loader, name in [(train_loader, "训练集"), (val_loader, "验证集")]:
        print(f"\n{'=' * 50}\n测试{name}:")
        for batch_data, batch_labels in loader:
            print(
                f"形状: {batch_data.shape}, 标签: {batch_labels}, 范围: [{batch_data.min():.4f}, {batch_data.max():.4f}]")
            break
