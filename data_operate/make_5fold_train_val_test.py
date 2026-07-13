import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    from data_operate.config import (
        CLASS_NAMES,
        CLASS_TO_IDX,
        DATA_5FOLD_DIR,
        DATA_DIR,
        N_SPLITS,
        SPLIT_NAMES,
        SPLIT_SEED,
        VAL_SPLIT_RATIO,
        VAL_SPLIT_SEED_OFFSET,
    )
except ImportError:
    from config import (
        CLASS_NAMES,
        CLASS_TO_IDX,
        DATA_5FOLD_DIR,
        DATA_DIR,
        N_SPLITS,
        SPLIT_NAMES,
        SPLIT_SEED,
        VAL_SPLIT_RATIO,
        VAL_SPLIT_SEED_OFFSET,
    )

DEFAULT_DATA_DIR = DATA_DIR
DEFAULT_OUT_DIR = DATA_5FOLD_DIR


def collect_samples(data_dir: Path):
    """Collect .raw files from data/normal and data/defective."""
    samples = []
    labels = []

    for cls in CLASS_NAMES:
        label = CLASS_TO_IDX[cls]
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"找不到类别目录: {cls_dir}")

        files = sorted([p for p in cls_dir.rglob("*.raw") if p.is_file()])
        print(f"{cls}: {len(files)} 个 .raw 文件")
        for p in files:
            samples.append(p)
            labels.append(label)

    if len(samples) == 0:
        raise RuntimeError(f"没有在 {data_dir}/normal 和 {data_dir}/defective 下找到 .raw 文件")

    return np.array(samples, dtype=object), np.array(labels, dtype=int)


def prepare_empty_fold_dir(out_root: Path, fold_idx: int):
    fold_dir = out_root / f"fold_{fold_idx}"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)

    for split in SPLIT_NAMES:
        for cls in CLASS_NAMES:
            (fold_dir / split / cls).mkdir(parents=True, exist_ok=True)

    return fold_dir


def copy_files(paths, labels, target_root: Path, split: str):
    label_to_class = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
    for src, label in zip(paths, labels):
        cls = label_to_class[int(label)]
        dst = target_root / split / cls / Path(src).name
        shutil.copy2(src, dst)


def count_split(fold_dir: Path):
    stats = {}
    for split in SPLIT_NAMES:
        stats[split] = {}
        for cls in CLASS_NAMES:
            stats[split][cls] = len(list((fold_dir / split / cls).glob("*.raw")))
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Create 5-fold stratified train/val/test splits. Each fold uses 20% test, then 25% of remaining 80% as val, giving about 6:2:2."
    )
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="原始数据目录，里面应有 normal/ 和 defective/")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="五折输出目录")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="划分随机种子")
    parser.add_argument("--n_splits", type=int, default=N_SPLITS, help="折数，默认5")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("创建 5-fold stratified train/val/test 划分")
    print(f"原始数据目录: {data_dir}")
    print(f"输出目录: {out_root}")
    print("每折比例: train ≈ 60%, val ≈ 20%, test ≈ 20%")
    print("=" * 80)

    samples, labels = collect_samples(data_dir)
    print(f"总样本数: {len(samples)}")
    print(f"normal: {(labels == CLASS_TO_IDX['normal']).sum()} | defective: {(labels == CLASS_TO_IDX['defective']).sum()}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    all_rows = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(samples, labels)):
        # 从剩余 80% 中再分出 25% 作为 val，最终 val = 0.8 * 0.25 = 0.2
        trainval_labels = labels[trainval_idx]
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=VAL_SPLIT_RATIO,
            random_state=args.seed + VAL_SPLIT_SEED_OFFSET + fold_idx
        )
        inner_train_idx, inner_val_idx = next(sss.split(samples[trainval_idx], trainval_labels))

        train_idx = trainval_idx[inner_train_idx]
        val_idx = trainval_idx[inner_val_idx]

        fold_dir = prepare_empty_fold_dir(out_root, fold_idx)
        copy_files(samples[train_idx], labels[train_idx], fold_dir, "train")
        copy_files(samples[val_idx], labels[val_idx], fold_dir, "val")
        copy_files(samples[test_idx], labels[test_idx], fold_dir, "test")

        stats = count_split(fold_dir)
        print("\n" + "-" * 80)
        print(f"Fold {fold_idx} -> {fold_dir}")
        for split in SPLIT_NAMES:
            n_normal = stats[split]["normal"]
            n_defective = stats[split]["defective"]
            n_total = n_normal + n_defective
            print(f"{split:5s}: total={n_total:4d}, normal={n_normal:4d}, defective={n_defective:4d}")
            all_rows.append({
                "fold": fold_idx,
                "split": split,
                "total": n_total,
                "normal": n_normal,
                "defective": n_defective,
            })

    # 保存划分统计，不依赖 openpyxl，避免环境问题
    summary_path = out_root / "fold_split_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("fold,split,total,normal,defective\n")
        for row in all_rows:
            f.write(f"{row['fold']},{row['split']},{row['total']},{row['normal']},{row['defective']}\n")

    print("\n" + "=" * 80)
    print("5-fold 划分完成")
    print(f"统计表已保存: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
