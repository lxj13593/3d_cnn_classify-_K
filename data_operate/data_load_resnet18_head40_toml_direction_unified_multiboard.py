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
PROJECT_DATASETS_ROOT = PROJECT_ROOT / "datasets"
DATASET_CONTAINER_RELATIVE_PATH = (
    Path("325_275_and_sphere_data") / "\u94bb\u5b54\u6570\u636e325_275"
)
TRAIN_DATASET_NAME = "multiboard_head40_toml"
TEST_DATASET_NAME = "test_3boards_head40_toml"
TRAIN_DATASET_ROOT_ENV = "DRILL_HEAD40_TOML_TRAIN_ROOT"
TEST_DATASET_ROOT_ENV = "DRILL_HEAD40_TOML_TEST_ROOT"
NORMAL_DIR_NAME = "normal_samples_by_board"
DEFECT_DIR_NAME = "defective_samples_by_board"
SOURCE_MANIFEST_NAME = "source_audit.csv"

NORMALIZATION_METHOD = "sample_percentile"
NORMALIZATION_SCOPE = "per_sample_head40_roi"
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
NORMALIZATION_EPSILON = 1e-6
DIRECTION_AXIS = "D"
STANDARDIZED_HEAD_SIDE = "high_depth_index"


def is_dataset_root(path: Path, expected_name: str | None = None) -> bool:
    if expected_name is not None and path.name != expected_name:
        return False
    return all(
        candidate.exists()
        for candidate in (
            path / SOURCE_MANIFEST_NAME,
            path / NORMAL_DIR_NAME,
            path / DEFECT_DIR_NAME,
        )
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


def resolve_dataset_root(
    explicit_root: str | Path | None = None,
    stored_root: str | Path | None = None,
    expected_name: str = TRAIN_DATASET_NAME,
) -> Path:
    def validate(candidate: Path) -> Path:
        candidate = candidate.expanduser().resolve()
        if not is_dataset_root(candidate, expected_name):
            raise FileNotFoundError(
                f"Not the expected dataset '{expected_name}': {candidate}"
            )
        return candidate

    if explicit_root is not None:
        return validate(Path(explicit_root))

    env_name = (
        TRAIN_DATASET_ROOT_ENV
        if expected_name == TRAIN_DATASET_NAME
        else TEST_DATASET_ROOT_ENV
    )
    environment_root = os.environ.get(env_name, "").strip()
    if environment_root:
        return validate(Path(environment_root))

    project_candidate = PROJECT_DATASETS_ROOT / expected_name
    if is_dataset_root(project_candidate, expected_name):
        return project_candidate.resolve()

    if stored_root:
        try:
            return validate(Path(stored_root))
        except FileNotFoundError:
            pass

    candidates = []
    for drive_root in mounted_drive_roots():
        candidate = drive_root / DATASET_CONTAINER_RELATIVE_PATH / expected_name
        if is_dataset_root(candidate, expected_name):
            candidates.append(candidate.resolve())
    candidates = sorted(set(candidates), key=lambda path: str(path).lower())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {expected_name}. Pass its path explicitly or set {env_name}."
        )
    joined = "\n  ".join(str(path) for path in candidates)
    raise RuntimeError(f"Multiple matching datasets were found:\n  {joined}")


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
    shape_whd: tuple[int, int, int]
    shape_dhw: tuple[int, int, int]
    raw_path: Path
    raw_file_name: str
    raw_sha256: str
    validation_fold: int
    b_head_up: bool
    direction_flip_required: bool
    standardized_head_side: str


def get_normalization_params() -> dict[str, object]:
    return {
        "method": NORMALIZATION_METHOD,
        "scope": NORMALIZATION_SCOPE,
        "lower_percentile": LOWER_PERCENTILE,
        "upper_percentile": UPPER_PERCENTILE,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "statistics_source": "each_pre_cropped_head40_roi_itself",
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


def get_direction_standardization_params() -> dict[str, object]:
    return {
        "enabled": True,
        "axis": DIRECTION_AXIS,
        "target_head_side": STANDARDIZED_HEAD_SIDE,
        "bHeadUp_false": "keep",
        "bHeadUp_true": "flip_D",
        "application_stage": "before_augmentation_and_normalization",
        "physical_raw_files_modified": False,
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
    manifest_path: str | Path,
    dataset_root: str | Path | None = None,
    verify_files: bool = True,
    expected_dataset_name: str = TRAIN_DATASET_NAME,
) -> list[RawSample]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    dataset_root = resolve_dataset_root(
        explicit_root=dataset_root,
        expected_name=expected_dataset_name,
    )

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
            "raw_shape_whd",
            "input_shape_dhw",
            "raw_file_name",
            "raw_relative_path",
            "raw_sha256",
            "validation_fold",
            "b_head_up",
            "original_head_side",
            "direction_flip_required",
            "direction_standardized",
            "standardized_head_side",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            label = row["label"].strip()
            label_index = int(row["label_index"])
            if CLASS_TO_INDEX.get(label) != label_index:
                raise ValueError(
                    f"Label/index mismatch at {manifest_path}:{row_number}"
                )
            shape_whd = parse_shape(row["raw_shape_whd"])
            shape_dhw = parse_shape(row["input_shape_dhw"])
            if shape_dhw != tuple(reversed(shape_whd)):
                raise ValueError(f"WHD/DHW mismatch at {manifest_path}:{row_number}")
            if shape_whd[2] != 40 or shape_whd[:2] not in {
                (29, 29),
                (37, 37),
                (49, 49),
            }:
                raise ValueError(
                    f"Unexpected TOML Head40 shape at "
                    f"{manifest_path}:{row_number}: {shape_whd}"
                )

            bool_values = {
                key: row[key].strip().lower()
                for key in (
                    "b_head_up",
                    "direction_flip_required",
                    "direction_standardized",
                )
            }
            if any(value not in {"true", "false"} for value in bool_values.values()):
                raise ValueError(
                    f"Invalid direction boolean at {manifest_path}:{row_number}"
                )
            b_head_up = bool_values["b_head_up"] == "true"
            direction_flip_required = (
                bool_values["direction_flip_required"] == "true"
            )
            if bool_values["direction_standardized"] != "true":
                raise ValueError(
                    f"Direction is not standardized at {manifest_path}:{row_number}"
                )
            if direction_flip_required != b_head_up:
                raise ValueError(
                    f"Direction flip/BHeadUp mismatch at {manifest_path}:{row_number}"
                )
            expected_original_side = (
                "low_depth_index" if b_head_up else "high_depth_index"
            )
            if row["original_head_side"].strip() != expected_original_side:
                raise ValueError(
                    f"Original head side mismatch at {manifest_path}:{row_number}"
                )
            if row["standardized_head_side"].strip() != STANDARDIZED_HEAD_SIDE:
                raise ValueError(
                    f"Target head side mismatch at {manifest_path}:{row_number}"
                )

            raw_path = (dataset_root / row["raw_relative_path"].strip()).resolve()
            if not is_within(raw_path, dataset_root):
                raise ValueError(f"RAW path escapes dataset root: {raw_path}")
            if verify_files:
                if not raw_path.is_file():
                    raise FileNotFoundError(f"RAW file not found: {raw_path}")
                expected_bytes = math.prod(shape_whd) * np.dtype(RAW_DTYPE).itemsize
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
                    difficulty=row["difficulty"].strip(),
                    shape_whd=shape_whd,
                    shape_dhw=shape_dhw,
                    raw_path=raw_path,
                    raw_file_name=row["raw_file_name"].strip(),
                    raw_sha256=row["raw_sha256"].strip().lower(),
                    validation_fold=int(row["validation_fold"]),
                    b_head_up=b_head_up,
                    direction_flip_required=direction_flip_required,
                    standardized_head_side=row["standardized_head_side"].strip(),
                )
            )

    ids = [sample.sample_id for sample in samples]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sample IDs: {duplicates[:5]}")
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
        dataset_root: str | Path | None = None,
        verify_files: bool = True,
        expected_dataset_name: str = TRAIN_DATASET_NAME,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.dataset_root = resolve_dataset_root(
            explicit_root=dataset_root,
            expected_name=expected_dataset_name,
        )
        self.samples = load_manifest(
            self.manifest_path,
            dataset_root=self.dataset_root,
            verify_files=verify_files,
            expected_dataset_name=expected_dataset_name,
        )
        self.augment = augment
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
            lower, upper = np.percentile(raw, [LOWER_PERCENTILE, UPPER_PERCENTILE])
            bounds = (float(lower), float(upper))
            self._percentile_cache[sample.sample_id] = bounds
        return raw, bounds[0], bounds[1]

    @staticmethod
    def _geometric_augmentation(volume: torch.Tensor) -> torch.Tensor:
        # C, D, H, W: preserve the physical head/tail direction D.
        if torch.rand(1).item() < 0.5:
            flip_dim = int(torch.randint(2, 4, (1,)).item())
            volume = torch.flip(volume, dims=[flip_dim])
        if torch.rand(1).item() < 0.5:
            rotation_k = int(torch.randint(1, 4, (1,)).item())
            volume = torch.rot90(volume, k=rotation_k, dims=[2, 3])
        if torch.rand(1).item() < 0.5:
            shift_y = int(torch.randint(-3, 4, (1,)).item())
            shift_x = int(torch.randint(-3, 4, (1,)).item())
            height, width = volume.shape[-2:]
            pad_y1, pad_y2 = max(0, shift_y), max(0, -shift_y)
            pad_x1, pad_x2 = max(0, shift_x), max(0, -shift_x)
            padded = F.pad(
                volume,
                (pad_x1, pad_x2, pad_y1, pad_y2),
                mode="constant",
                value=0,
            )
            volume = padded[:, :, pad_y2 : pad_y2 + height, pad_x2 : pad_x2 + width]
        return volume

    @staticmethod
    def _intensity_augmentation(volume: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            noise_std = 0.01 + torch.rand(1).item() * 0.02
            volume = torch.clamp(volume + torch.randn_like(volume) * noise_std, 0, 1)
        if torch.rand(1).item() < 0.5:
            volume = torch.clamp(volume * (0.9 + torch.rand(1).item() * 0.2), 0, 1)
        if torch.rand(1).item() < 0.5:
            factor = 0.9 + torch.rand(1).item() * 0.2
            volume = torch.clamp((volume - 0.5) * factor + 0.5, 0, 1)
        return volume

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        raw, lower, upper = self._read_volume(sample)
        volume = torch.from_numpy(raw.copy()).float().unsqueeze(0)
        if sample.direction_flip_required:
            # C, D, H, W: reverse only the drill-length axis.
            volume = torch.flip(volume, dims=[1])
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
            "raw_shape_whd": "x".join(str(value) for value in sample.shape_whd),
            "input_shape_dhw": "x".join(str(value) for value in sample.shape_dhw),
            "raw_path": str(sample.raw_path),
            "b_head_up": sample.b_head_up,
            "direction_flipped": sample.direction_flip_required,
            "direction_standardized": True,
            "standardized_head_side": sample.standardized_head_side,
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
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.groups.values())
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


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
    dataset_root: str | Path | None,
    batch_size: int,
    num_workers: int,
    augment: bool,
    seed: int,
    verify_files: bool = True,
    expected_dataset_name: str = TRAIN_DATASET_NAME,
) -> DataLoader:
    dataset = MultiBoardRawDataset(
        manifest_path,
        augment=augment,
        dataset_root=dataset_root,
        verify_files=verify_files,
        expected_dataset_name=expected_dataset_name,
    )
    batch_sampler = ShapeBatchSampler(
        dataset.shapes,
        batch_size=batch_size,
        shuffle=augment,
        seed=seed,
        drop_last=False,
    )
    kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def get_fold_loaders(
    split_root: str | Path,
    fold_index: int,
    dataset_root: str | Path | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
    verify_files: bool = True,
) -> tuple[DataLoader, DataLoader]:
    fold_dir = Path(split_root).resolve() / f"fold_{fold_index}"
    common = {
        "dataset_root": dataset_root,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "verify_files": verify_files,
        "expected_dataset_name": TRAIN_DATASET_NAME,
    }
    return (
        make_loader(fold_dir / "train.csv", augment=True, **common),
        make_loader(fold_dir / "val.csv", augment=False, **common),
    )


def summarize_dataset(dataset: MultiBoardRawDataset) -> dict[str, object]:
    label_counts = Counter(sample.label for sample in dataset.samples)
    shape_counts = Counter(
        "x".join(str(value) for value in sample.shape_whd)
        for sample in dataset.samples
    )
    return {
        "total": len(dataset),
        "normal": label_counts["normal"],
        "defective": label_counts["defective"],
        "boards": len({sample.board for sample in dataset.samples}),
        "raw_shape_whd_counts": dict(sorted(shape_counts.items())),
        "b_head_up_false": sum(not sample.b_head_up for sample in dataset.samples),
        "b_head_up_true": sum(sample.b_head_up for sample in dataset.samples),
        "direction_flipped": sum(
            sample.direction_flip_required for sample in dataset.samples
        ),
        "standardized_head_side": STANDARDIZED_HEAD_SIDE,
    }
