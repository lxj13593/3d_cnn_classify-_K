"""Generate a read-only, 20-sample preview of the agreed Head40 representation.

Reuse the existing non-augmented loader. No training or source-data changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
import scipy
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
LABEL_CN = {"normal": "正常", "defective": "缺陷"}
DIFFICULTY_CN = {"clear": "清晰", "fuzzy": "模糊"}


def configure_plotting() -> None:
    # Render in a separate, torch-free process to avoid mixed OpenMP runtimes.
    global plt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False})


def board_key(board: str) -> str:
    match = re.search(r"VT-X750-(\d+)_(\d+)", board)
    return f"B{match[1]}-{int(match[2]):04d}" if match else board[:20]


def select_samples(dataset: MultiBoardRawDataset) -> list[int]:
    """Six clear/four fuzzy, with 2/6/2 samples at 29/37/49 for each class."""
    slots = [
        ("clear", 29, False), ("fuzzy", 29, False),
        ("clear", 49, True), ("fuzzy", 49, False),
        ("clear", 37, False), ("clear", 37, False),
        ("clear", 37, False), ("clear", 37, False),
        ("fuzzy", 37, False), ("fuzzy", 37, False),
    ]
    selected: dict[str, list[int]] = {"normal": [], "defective": []}
    used_ids: set[str] = set()
    used_hashes: set[str] = set()
    board_counts: Counter = Counter()
    class_board_counts: Counter = Counter()
    for difficulty, side, require_flip in slots:
        for label in ("normal", "defective"):
            candidates = [
                i for i, s in enumerate(dataset.samples)
                if s.label == label and s.difficulty == difficulty
                and s.shape_dhw == (40, side, side)
                and s.sample_id not in used_ids and s.raw_sha256 not in used_hashes
                and (not require_flip or s.direction_flip_required)
            ]
            if not candidates:
                raise ValueError(f"No sample for {label}/{difficulty}/{side}/flip={require_flip}")
            def rank(i: int) -> tuple:
                s = dataset.samples[i]
                digest = hashlib.sha256(("42|" + s.sample_id).encode()).hexdigest()
                return board_counts[s.board], class_board_counts[label, s.board], digest
            index = min(candidates, key=rank)
            sample = dataset.samples[index]
            selected[label].append(index)
            used_ids.add(sample.sample_id)
            used_hashes.add(sample.raw_sha256)
            board_counts[sample.board] += 1
            class_board_counts[label, sample.board] += 1
    return selected["normal"] + selected["defective"]


def gradient(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    smooth = ndimage.gaussian_filter(image, sigma=(1., 1., 1.), mode="reflect", truncate=4.)
    components = np.gradient(smooth, 1., 1., 1., edge_order=1)
    magnitude = np.sqrt(sum(component * component for component in components))
    return magnitude, tuple(components)


def training_threshold(dataset: MultiBoardRawDataset) -> tuple[float, int]:
    """Exact P90 of all positive, finite pre-NMS training gradients, not a sample."""
    capacity = sum(int(np.prod(s.shape_dhw)) for s in dataset.samples)
    tmp_root = PROJECT_ROOT / "tmp"
    tmp_root.mkdir(exist_ok=True)
    start = time.perf_counter()
    with tempfile.TemporaryFile(prefix="canny_gradient_", dir=tmp_root) as temporary:
        temporary.truncate(capacity * np.dtype(np.float32).itemsize)
        pool = np.memmap(temporary, dtype=np.float32, mode="r+", shape=(capacity,))
        try:
            used = 0
            for index in range(len(dataset)):
                image = dataset[index]["volume"][0].numpy()
                if not np.isfinite(image).all():
                    raise ValueError(f"Non-finite image: {dataset.samples[index].sample_id}")
                magnitude, _ = gradient(image)
                values = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
                pool[used:used + values.size] = values
                used += values.size
                if (index + 1) % 250 == 0 or index + 1 == len(dataset):
                    print(f"Threshold: {index + 1}/{len(dataset)}; {time.perf_counter()-start:.1f}s", flush=True)
            if not used:
                raise ValueError("Training gradients contain no positive finite value")
            high = float(np.percentile(pool[:used], 90., method="linear", overwrite_input=True))
            if not np.isfinite(high) or high <= 0:
                raise ValueError(f"Invalid training threshold: {high}")
        finally:
            pool._mmap.close()
    return high, used


def canny_masks(image: np.ndarray, high: float) -> tuple[np.ndarray, np.ndarray]:
    magnitude, components = gradient(image)
    nonzero = magnitude > 0
    direction = np.stack([
        np.divide(c, magnitude, out=np.zeros_like(c), where=nonzero)
        for c in components
    ])
    coordinates = np.indices(image.shape, dtype=np.float32)
    plus, minus = coordinates + direction, coordinates - direction
    valid = nonzero.copy()
    for axis, length in enumerate(image.shape):
        valid &= (plus[axis] >= 0) & (plus[axis] <= length - 1)
        valid &= (minus[axis] >= 0) & (minus[axis] <= length - 1)
    g_plus = ndimage.map_coordinates(magnitude, plus, order=1, mode="constant", cval=0., prefilter=False)
    g_minus = ndimage.map_coordinates(magnitude, minus, order=1, mode="constant", cval=0., prefilter=False)
    nms = np.where(valid & (magnitude >= g_plus) & (magnitude >= g_minus), magnitude, 0.)
    strong = nms >= high
    candidate = nms >= 0.5 * high
    components_map, count = ndimage.label(candidate, structure=np.ones((3, 3, 3), dtype=bool))
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(components_map[strong])] = True
    keep[0] = False
    edge = keep[components_map]
    dilated = ndimage.binary_dilation(edge, structure=np.ones((3, 3, 3), dtype=bool), iterations=1, border_value=0)
    return edge, dilated


def views(volume: np.ndarray) -> list[np.ndarray]:
    _, height, width = volume.shape
    return [volume[:, height // 2, :], volume[:, :, width // 2],
            volume.max(axis=1), volume[19], volume[23]]


def detailed_png(path: Path, record: dict, arrays: tuple[np.ndarray, ...]) -> None:
    cols = ["原图灰度", "二值边缘", "膨胀后 Mask", "处理后：原图 × Mask"]
    depth, height, width = record["shape_dhw"]
    rows = [f"D-W 中心切片\nH={height//2}", f"D-H 中心切片\nW={width//2}",
            "D-W 最大投影\n沿 H 投影，非单张切片", "H-W 切片\nD=19", "H-W 切片\nD=23"]
    panels = [views(a) for a in arrays]
    fig, axes = plt.subplots(5, 4, figsize=(12.8, 15.4))
    for row in range(5):
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(panels[col][row], cmap="gray", vmin=0., vmax=1., interpolation="nearest", origin="lower")
            if row == 0:
                ax.set_title(cols[col], fontsize=11)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=9)
            if row < 3:
                ax.set_yticks([0, 10, 20, 30, depth-1])
            ax.set_xlabel("W 体素索引" if row != 1 else "H 体素索引", fontsize=8)
    difficulty = DIFFICULTY_CN[record["difficulty"]]
    fig.suptitle(
        f"{record['index']:02d}  {difficulty}{record['label_cn']} | {record['board_key']} | 钻孔 {record['sequence']}\n"
        f"输入 {depth}×{height}×{width} | Mask 覆盖 {record['mask_ratio']:.1%}\n"
        "第 0 折训练样本；无随机增强；纵向 D=39 在上（头端）；所有灰度图固定显示 0～1",
        fontsize=13, y=.975,
    )
    fig.text(.5, .013, "标签来自现有清单；D=19/23 仅为显示切片，不代表人工标定表面。更多位置可在离线查看器切换。", ha="center", fontsize=9)
    fig.subplots_adjust(left=.13, right=.985, top=.90, bottom=.055, hspace=.35, wspace=.18)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def overview_png(path: Path, items: list[tuple[dict, tuple[np.ndarray, ...]]], label: str) -> None:
    fig, axes = plt.subplots(5, 8, figsize=(22., 15.))
    for j, (record, arrays) in enumerate(items):
        row, offset = j % 5, (j // 5) * 4
        for col, volume in enumerate(arrays):
            ax = axes[row, offset + col]
            ax.imshow(volume[:, volume.shape[1]//2, :], cmap="gray", vmin=0., vmax=1., interpolation="nearest", origin="lower")
            ax.axis("off")
            title = ["原图", "二值边缘", "膨胀 Mask", "处理后"][col]
            if col == 0:
                title = (f"{record['index']:02d}  {record['board_key']}\n钻孔 {record['sequence']} · "
                         f"{DIFFICULTY_CN[record['difficulty']]}\n{title}")
            else:
                title = "\n\n" + title
            ax.set_title(title, fontsize=10)
    fig.suptitle(f"{LABEL_CN[label]} 10 个训练样本：原图 / 二值边缘 / 膨胀掩膜 / 处理后\n"
                 "全部为 D-W 中心切片，头端在上；灰度窗固定 0～1。中心切片未必经过缺陷，请结合单样本图与切片查看器。",
                 fontsize=17, y=.985)
    fig.subplots_adjust(left=.012, right=.99, top=.91, bottom=.015, hspace=.38, wspace=.10)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def render_saved_arrays(output: Path) -> None:
    configure_plotting()
    index = json.loads((output / "preview_index.json").read_text(encoding="utf-8"))
    previews = []
    for record in index["samples"]:
        with np.load(output / record["array_file"], allow_pickle=False) as archive:
            arrays = tuple(archive[k] for k in ("image", "edge_mask", "dilated_mask", "processed"))
        detailed_png(output / record["png_file"], record, arrays)
        previews.append((record, arrays))
        print(f"Rendered {record['index']:02d}/20", flush=True)
    for label, filename in (("normal", "正常10样本_总览.png"), ("defective", "缺陷10样本_总览.png")):
        overview_png(output / filename, [p for p in previews if p[0]["label"] == label], label)
    print(f"PNG DONE: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("H:/多板数据集/multiboard_head40_toml"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "查找结果" / "Head40_3D_Canny_20样本预览_20261005")
    parser.add_argument("--render-only", action="store_true", help="Only render the previously generated NPZ files, without importing torch")
    parser.add_argument("--resume-output", action="store_true", help="Continue this task's partially generated preview directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.render_only:
        render_saved_arrays(output)
        return
    import torch
    from data_operate.data_load_resnet18_head40_toml_direction_unified_multiboard import (
        MultiBoardRawDataset, get_direction_standardization_params, get_normalization_params,
    )
    if output.exists() and not args.resume_output:
        raise FileExistsError(f"Refusing to overwrite an existing preview: {output}")
    manifest = PROJECT_ROOT / "datasets/resnet18_head40_toml_direction_unified_multiboard_5fold_clear_fuzzy_20260924/fold_0/train.csv"
    torch.set_num_threads(1)
    dataset = MultiBoardRawDataset(manifest, augment=False, dataset_root=args.dataset_root, verify_files=False)
    if any(s.validation_fold == 0 for s in dataset.samples):
        raise ValueError("Fold 0 validation sample found in its training manifest")
    selected = select_samples(dataset)
    print(f"Training records: {len(dataset)}; selected 10 normal + 10 defective.", flush=True)
    high, gradient_count = training_threshold(dataset)
    print(f"Exact training P90 high={high:.9f}, low={0.5*high:.9f}; gradients={gradient_count}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "单样本对照图").mkdir(exist_ok=True)
    (output / "处理后体数据").mkdir(exist_ok=True)
    config = {
        "representation": "edge_head40_hard_mask", "implementation_version": "preview_20261005_v1",
        "fold": 0, "seed_for_selection": 42, "training_sample_count": len(dataset),
        "training_manifest": str(manifest), "training_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "dataset_root": str(dataset.dataset_root), "normalization": get_normalization_params(),
        "direction": get_direction_standardization_params(), "augment": False,
        "gaussian_sigma": [1., 1., 1.], "gaussian_mode": "reflect", "gaussian_truncate": 4.,
        "gradient": "numpy.gradient(edge_order=1, spacing=[1,1,1])",
        "threshold_stage": "smoothed_gradient_before_nms", "threshold_population": "all_finite_positive_fold0_training_voxels",
        "gradient_value_count": gradient_count, "percentile": 90., "percentile_method": "linear",
        "high_threshold": high, "low_threshold": .5*high,
        "nms": "3D_unit_gradient_direction_trilinear_order1; reject_outside_ROI",
        "hysteresis_connectivity": 26, "dilation": "full_3x3x3_once_border0",
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "torch_version": torch.__version__,
        "selection": "per_class_6_clear_4_fuzzy; 2_at_29_6_at_37_2_at_49; one_clear_49_requires_D_flip; prioritize_board_diversity_then_sha256(42|sample_id)",
        "preview_only": True, "source_data_modified": False,
    }
    with (output / "edge_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    records = []
    for number, index in enumerate(selected, start=1):
        sample = dataset.samples[index]
        image = dataset[index]["volume"][0].numpy().copy()
        edge, mask = canny_masks(image, high)
        processed = image * mask
        if processed.dtype != np.float32 or np.any(processed[~mask] != 0):
            raise ValueError("Unexpected processed dtype/background")
        prefix = f"{number:02d}_{LABEL_CN[sample.label]}_{board_key(sample.board)}_钻孔{sample.sequence}"
        record = {
            "index": number, "sample_id": sample.sample_id, "label": sample.label,
            "label_cn": LABEL_CN[sample.label], "difficulty": sample.difficulty,
            "board": sample.board, "board_key": board_key(sample.board), "sequence": sample.sequence,
            "shape_dhw": list(sample.shape_dhw), "raw_path": str(sample.raw_path), "raw_sha256": sample.raw_sha256,
            "direction_flipped": sample.direction_flip_required, "validation_fold": sample.validation_fold,
            "edge_ratio": float(edge.mean()), "mask_ratio": float(mask.mean()),
            "empty_mask": not bool(mask.any()), "png_file": f"单样本对照图/{prefix}.png",
            "array_file": f"处理后体数据/{prefix}.npz",
        }
        np.savez_compressed(output / record["array_file"], image=image, edge_mask=edge, dilated_mask=mask, processed=processed)
        records.append(record)
        print(f"Preview {number:02d}/20: {sample.label}/{sample.difficulty} {record['board_key']} drill{sample.sequence}; mask={mask.mean():.1%}", flush=True)
    with (output / "preview_index.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "samples": records}, handle, ensure_ascii=False, indent=2)
    with (output / "20样本清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"ARRAYS DONE: {output}; selected boards={len({r['board'] for r in records})}; empty masks={sum(r['empty_mask'] for r in records)}", flush=True)


if __name__ == "__main__":
    main()
