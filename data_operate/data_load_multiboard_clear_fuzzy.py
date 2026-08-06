from __future__ import annotations

import csv
import ctypes
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


RAW_DTYPE = np.uint8
CLASS_TO_INDEX = {"normal": 0, "defective": 1}
INDEX_TO_CLASS = {0: "normal", 1: "defective"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DATASET_RELATIVE_PATH = Path("datasets")
PROJECT_DATASETS_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_PROJECT_DATASET_ROOT = PROJECT_ROOT / PROJECT_DATASET_RELATIVE_PATH
LEGACY_DATASET_RELATIVE_PATH = (
    Path("325_275_and_sphere_data")
    / "钻孔数据325_275"
    / "clear_binary_dataset"
)
LEGACY_DATASET_PARENT_RELATIVE_PATH = (
    Path("325_275_and_sphere_data")
    / "\u94bb\u5b54\u6570\u636e325_275"
)
DATASET_ROOT_ENV = "DRILL_COMBINED_DATASET_ROOT"
CLEAR_DATASET_NAME = "clear_binary_dataset"
FUZZY_DATASET_NAME = "fuzzy_binary_dataset"
NORMAL_DIR_NAME = "normal_samples_by_board"
CLEAR_DEFECT_DIR_NAME = "obvious_defects_by_board"
FUZZY_DEFECT_DIR_NAME = "fuzzy_defects_by_board"
MANIFEST_NAME = "all_boards_manifest.csv"

NORMALIZATION_METHOD = "sample_percentile"
NORMALIZATION_SCOPE = "per_sample_full_volume"
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
NORMALIZATION_EPSILON = 1e-6


def is_dataset_parent(path: Path) -> bool:
    required = (
        path / CLEAR_DATASET_NAME / NORMAL_DIR_NAME / MANIFEST_NAME,
        path / CLEAR_DATASET_NAME / CLEAR_DEFECT_DIR_NAME / MANIFEST_NAME,
        path / FUZZY_DATASET_NAME / NORMAL_DIR_NAME / MANIFEST_NAME,
        path / FUZZY_DATASET_NAME / FUZZY_DEFECT_DIR_NAME / MANIFEST_NAME,
    )
    return all(manifest.is_file() for manifest in required)


def normalize_parent_candidate(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if is_dataset_parent(candidate):
        return candidate
    if candidate.name in {CLEAR_DATASET_NAME, FUZZY_DATASET_NAME}:
        parent = candidate.parent
        if is_dataset_parent(parent):
            return parent
    raise FileNotFoundError(
        "Directory must contain clear_binary_dataset and fuzzy_binary_dataset: "
        f"{candidate}"
    )


def project_dataset_roots() -> list[Path]:
    if is_dataset_parent(DEFAULT_PROJECT_DATASET_ROOT):
        return [DEFAULT_PROJECT_DATASET_ROOT.resolve()]

    candidates = [PROJECT_DATASETS_ROOT]
    if PROJECT_DATASETS_ROOT.is_dir():
        candidates.extend(
            path for path in PROJECT_DATASETS_ROOT.iterdir() if path.is_dir()
        )
    return sorted(
        {path.resolve() for path in candidates if is_dataset_parent(path)},
        key=lambda path: str(path).lower(),
    )


def mounted_drive_roots() -> list[Path]:
    if os.name != "nt":
        return [Path("/")]
    bitmask = int(ctypes.windll.kernel32.GetLogicalDrives())
    return [
        Path(f"{chr(ord('A') + index)}:\\")
        for index in range(26)
        if bitmask & (1 << index)
    ]


def resolve_dataset_parent(
    explicit_root: str | Path | None = None,
    stored_root: str | Path | None = None,
) -> Path:
    if explicit_root is not None:
        return normalize_parent_candidate(Path(explicit_root))

    environment_root = os.environ.get(DATASET_ROOT_ENV, "").strip()
    if environment_root:
        try:
            return normalize_parent_candidate(Path(environment_root))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{DATASET_ROOT_ENV} is not a valid combined dataset parent"
            ) from exc

    project_candidates = project_dataset_roots()
    if len(project_candidates) == 1:
        return project_candidates[0]
    if len(project_candidates) > 1:
        joined = "\n  ".join(str(path) for path in project_candidates)
        raise RuntimeError(
            "Multiple combined datasets were found under the project datasets "
            "directory. Select one explicitly with --dataset-parent:\n  " + joined
        )

    if stored_root:
        try:
            return normalize_parent_candidate(Path(stored_root))
        except FileNotFoundError:
            pass

    candidates = []
    for drive_root in mounted_drive_roots():
        candidate = drive_root / LEGACY_DATASET_PARENT_RELATIVE_PATH
        if is_dataset_parent(candidate):
            candidates.append(candidate.resolve())
    candidates = sorted(set(candidates), key=lambda path: str(path).lower())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Could not find a parent containing clear_binary_dataset and "
            "fuzzy_binary_dataset. Put both under "
            f"{DEFAULT_PROJECT_DATASET_ROOT}, pass --dataset-parent, or set "
            f"{DATASET_ROOT_ENV}."
        )
    joined = "\n  ".join(str(path) for path in candidates)
    raise RuntimeError(
        "Multiple matching datasets were found. Select one explicitly:\n  " + joined
    )


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class RawSample:
    sample_id: str
    board: str
    sequence: str
    label: str
    label_index: int
    difficulty: str
    source_dataset: str
    source_class_dir: str
    shape_whd: tuple[int, int, int]
    shape_dhw: tuple[int, int, int]
    raw_path: Path
    raw_file_name: str
    raw_sha256: str
    validation_fold: int


def get_normalization_params() -> dict[str, object]:
    return {
        "method": NORMALIZATION_METHOD,
        "scope": NORMALIZATION_SCOPE,
        "lower_percentile": LOWER_PERCENTILE,
        "upper_percentile": UPPER_PERCENTILE,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "statistics_source": "each_input_volume_itself",
    }


def get_augmentation_params() -> dict[str, object]:
    return {
        "train_only": True,
        "length_flip": False,
        "spatial_flip_probability": 0.5,
        "spatial_rotation_probability": 0.5,
        "spatial_shift_probability": 0.5,
        "spatial_shift_voxels": [-3, 3],
        "noise_probability": 0.5,
        "noise_std_range": [0.01, 0.03],
        "brightness_probability": 0.5,
        "brightness_range": [0.9, 1.1],
        "contrast_probability": 0.5,
        "contrast_range": [0.9, 1.1],
    }


def validate_normalization_params(params: dict[str, object]) -> None:
    expected = get_normalization_params()
    if not isinstance(params, dict):
        raise ValueError("Checkpoint has no normalization metadata")
    for key in ("method", "scope"):
        if params.get(key) != expected[key]:
            raise ValueError(
                f"Normalization mismatch for {key}: "
                f"expected {expected[key]}, got {params.get(key)}"
            )
    for key in ("lower_percentile", "upper_percentile"):
        if not np.isclose(float(params.get(key, float("nan"))), float(expected[key])):
            raise ValueError(
                f"Normalization mismatch for {key}: "
                f"expected {expected[key]}, got {params.get(key)}"
            )


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"Invalid shape: {value}") from exc
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"Invalid shape: {value}")
    return shape


def load_manifest(
    manifest_path: Path,
    dataset_parent: str | Path | None = None,
    verify_files: bool = True,
) -> list[RawSample]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    dataset_parent = resolve_dataset_parent(dataset_parent)

    samples: list[RawSample] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "board",
            "sequence",
            "label",
            "label_index",
            "difficulty",
            "source_dataset",
            "source_class_dir",
            "raw_shape_whd",
            "input_shape_dhw",
            "raw_file_name",
            "raw_sha256",
            "validation_fold",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            label = row["label"].strip()
            label_index = int(row["label_index"])
            if CLASS_TO_INDEX.get(label) != label_index:
                raise ValueError(
                    f"Label/index mismatch in {manifest_path}:{row_number}: "
                    f"{label}/{label_index}"
                )
            difficulty = row["difficulty"].strip()
            if difficulty not in {"clear", "fuzzy"}:
                raise ValueError(
                    f"Invalid difficulty in {manifest_path}:{row_number}: "
                    f"{difficulty}"
                )
            source_dataset = row["source_dataset"].strip()
            source_class_dir = row["source_class_dir"].strip()

            shape_whd = parse_shape(row["raw_shape_whd"])
            shape_dhw = parse_shape(row["input_shape_dhw"])
            if shape_dhw != tuple(reversed(shape_whd)):
                raise ValueError(
                    f"WHD/DHW mismatch in {manifest_path}:{row_number}: "
                    f"{shape_whd}/{shape_dhw}"
                )

            relative_value = row.get("raw_relative_path", "").strip()
            if relative_value:
                raw_path = (dataset_parent / relative_value).resolve()
            else:
                raw_path = (
                    dataset_parent
                    / source_dataset
                    / source_class_dir
                    / row["board"].strip()
                    / "drill"
                    / row["raw_file_name"].strip()
                ).resolve()
            if not is_within(raw_path, dataset_parent):
                raise ValueError(f"RAW path escapes dataset parent: {raw_path}")
            if verify_files:
                if not raw_path.is_file():
                    raise FileNotFoundError(f"RAW file not found: {raw_path}")
                expected_bytes = math.prod(shape_dhw) * np.dtype(RAW_DTYPE).itemsize
                if raw_path.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"RAW size mismatch for {raw_path}: "
                        f"expected {expected_bytes}, got {raw_path.stat().st_size}"
                    )

            samples.append(
                RawSample(
                    sample_id=row["sample_id"].strip(),
                    board=row["board"].strip(),
                    sequence=row["sequence"].strip(),
                    label=label,
                    label_index=label_index,
                    difficulty=difficulty,
                    source_dataset=source_dataset,
                    source_class_dir=source_class_dir,
                    shape_whd=shape_whd,
                    shape_dhw=shape_dhw,
                    raw_path=raw_path,
                    raw_file_name=row["raw_file_name"].strip(),
                    raw_sha256=row["raw_sha256"].strip().lower(),
                    validation_fold=int(row["validation_fold"]),
                )
            )

    ids = [sample.sample_id for sample in samples]
    duplicate_ids = [value for value, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate sample IDs in {manifest_path}: {duplicate_ids[:5]}")
    if not samples:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return samples


def normalize_percentile(
    volume: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    volume = (volume - lower) / (upper - lower + NORMALIZATION_EPSILON)
    return torch.clamp(volume, 0.0, 1.0)


class MultiBoardRawDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        augment: bool,
        dataset_parent: str | Path | None = None,
        verify_files: bool = True,
        cache_percentiles: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.dataset_parent = resolve_dataset_parent(dataset_parent)
        self.samples = load_manifest(
            self.manifest_path,
            dataset_parent=self.dataset_parent,
            verify_files=verify_files,
        )
        self.augment = augment
        self.cache_percentiles = cache_percentiles
        self._percentile_cache: dict[str, tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def shapes(self) -> list[tuple[int, int, int]]:
        return [sample.shape_dhw for sample in self.samples]

    def _read_volume(self, sample: RawSample) -> tuple[np.ndarray, float, float]:
        raw = np.fromfile(sample.raw_path, dtype=RAW_DTYPE)
        expected_elements = math.prod(sample.shape_dhw)
        if raw.size != expected_elements:
            raise ValueError(
                f"RAW element count changed for {sample.raw_path}: "
                f"expected {expected_elements}, got {raw.size}"
            )
        raw = raw.reshape(sample.shape_dhw)

        bounds = self._percentile_cache.get(sample.sample_id)
        if bounds is None:
            lower, upper = np.percentile(
                raw, [LOWER_PERCENTILE, UPPER_PERCENTILE]
            )
            bounds = (float(lower), float(upper))
            if self.cache_percentiles:
                self._percentile_cache[sample.sample_id] = bounds
        return raw, bounds[0], bounds[1]

    @staticmethod
    def _geometric_augmentation(volume: torch.Tensor) -> torch.Tensor:
        # Tensor layout is C, D, H, W. Length D is deliberately not flipped.
        if torch.rand(1).item() > 0.5:
            flip_dim = int(torch.randint(2, 4, (1,)).item())
            volume = torch.flip(volume, dims=[flip_dim])

        if torch.rand(1).item() > 0.5:
            rotation_k = int(torch.randint(1, 4, (1,)).item())
            volume = torch.rot90(volume, k=rotation_k, dims=[2, 3])

        if torch.rand(1).item() > 0.5:
            shift_y = int(torch.randint(-3, 4, (1,)).item())
            shift_x = int(torch.randint(-3, 4, (1,)).item())
            height, width = volume.shape[-2:]
            pad_y1 = max(0, shift_y)
            pad_y2 = max(0, -shift_y)
            pad_x1 = max(0, shift_x)
            pad_x2 = max(0, -shift_x)
            padded = F.pad(
                volume,
                (pad_x1, pad_x2, pad_y1, pad_y2),
                mode="constant",
                value=0,
            )
            volume = padded[
                :,
                :,
                pad_y2 : pad_y2 + height,
                pad_x2 : pad_x2 + width,
            ]
        return volume

    @staticmethod
    def _intensity_augmentation(volume: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > 0.5:
            noise_std = 0.01 + torch.rand(1).item() * 0.02
            volume = torch.clamp(
                volume + torch.randn_like(volume) * noise_std, 0.0, 1.0
            )

        if torch.rand(1).item() > 0.5:
            brightness_factor = 0.9 + torch.rand(1).item() * 0.2
            volume = torch.clamp(volume * brightness_factor, 0.0, 1.0)

        if torch.rand(1).item() > 0.5:
            contrast_factor = 0.9 + torch.rand(1).item() * 0.2
            volume = torch.clamp(
                (volume - 0.5) * contrast_factor + 0.5, 0.0, 1.0
            )
        return volume

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        raw, lower, upper = self._read_volume(sample)
        volume = torch.from_numpy(raw.copy()).float().unsqueeze(0)

        if self.augment:
            volume = self._geometric_augmentation(volume)
        volume = normalize_percentile(volume, lower, upper)
        if self.augment:
            volume = self._intensity_augmentation(volume)

        return {
            "volume": volume,
            "target": torch.tensor(sample.label_index, dtype=torch.long),
            "sample_id": sample.sample_id,
            "board": sample.board,
            "sequence": sample.sequence,
            "difficulty": sample.difficulty,
            "source_dataset": sample.source_dataset,
            "source_class_dir": sample.source_class_dir,
            "raw_shape_whd": "x".join(str(value) for value in sample.shape_whd),
            "input_shape_dhw": "x".join(str(value) for value in sample.shape_dhw),
            "raw_path": str(sample.raw_path),
        }


class ShapeBatchSampler(Sampler[list[int]]):
    """Build batches containing only samples with the same D/H/W shape."""

    def __init__(
        self,
        shapes: list[tuple[int, int, int]],
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, shape in enumerate(shapes):
            self.groups[shape].append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for shape in sorted(self.groups):
            indices = list(self.groups[shape])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


def worker_init_fn(worker_id: int) -> None:
    del worker_id
    worker_info = torch.utils.data.get_worker_info()
    seed = torch.initial_seed() if worker_info is None else worker_info.seed
    seed %= 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    manifest_path: str | Path,
    dataset_parent: str | Path | None,
    batch_size: int,
    num_workers: int,
    augment: bool,
    seed: int,
    verify_files: bool = True,
) -> DataLoader:
    dataset = MultiBoardRawDataset(
        manifest_path,
        augment=augment,
        dataset_parent=dataset_parent,
        verify_files=verify_files,
    )
    batch_sampler = ShapeBatchSampler(
        dataset.shapes,
        batch_size=batch_size,
        shuffle=augment,
        seed=seed,
        drop_last=False,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(**loader_kwargs)


def get_fold_loaders(
    split_root: str | Path,
    fold_index: int,
    dataset_parent: str | Path | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
    verify_files: bool = True,
) -> tuple[DataLoader, DataLoader]:
    fold_dir = Path(split_root).resolve() / f"fold_{fold_index}"
    train_loader = make_loader(
        fold_dir / "train.csv",
        dataset_parent=dataset_parent,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=True,
        seed=seed,
        verify_files=verify_files,
    )
    val_loader = make_loader(
        fold_dir / "val.csv",
        dataset_parent=dataset_parent,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        seed=seed,
        verify_files=verify_files,
    )
    return train_loader, val_loader


def summarize_dataset(dataset: MultiBoardRawDataset) -> dict[str, object]:
    label_counts = Counter(sample.label for sample in dataset.samples)
    difficulty_counts = Counter(sample.difficulty for sample in dataset.samples)
    difficulty_label_counts = Counter(
        (sample.difficulty, sample.label) for sample in dataset.samples
    )
    shape_counts = Counter(
        "x".join(str(value) for value in sample.shape_whd)
        for sample in dataset.samples
    )
    return {
        "total": len(dataset),
        "normal": label_counts["normal"],
        "defective": label_counts["defective"],
        "clear": difficulty_counts["clear"],
        "fuzzy": difficulty_counts["fuzzy"],
        "clear_normal": difficulty_label_counts[("clear", "normal")],
        "clear_defective": difficulty_label_counts[("clear", "defective")],
        "fuzzy_normal": difficulty_label_counts[("fuzzy", "normal")],
        "fuzzy_defective": difficulty_label_counts[("fuzzy", "defective")],
        "boards": len({sample.board for sample in dataset.samples}),
        "raw_shape_whd_counts": dict(sorted(shape_counts.items())),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the multi-board loader")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "datasets"
        / "multiboard_clear_fuzzy_5fold",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-parent", type=Path, default=None)
    args = parser.parse_args()

    train_loader, val_loader = get_fold_loaders(
        args.split_root,
        args.fold,
        dataset_parent=args.dataset_parent,
        batch_size=args.batch_size,
    )
    for name, loader in (("train", train_loader), ("val", val_loader)):
        print(name, summarize_dataset(loader.dataset))
        batch = next(iter(loader))
        print(
            name,
            tuple(batch["volume"].shape),
            tuple(batch["target"].shape),
            float(batch["volume"].min()),
            float(batch["volume"].max()),
        )
