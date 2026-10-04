"""Standalone Head40 data pipeline for the IDO-GS experiment.

The implementation has no dependency on another experiment's loader.  Existing
CSV splits remain the source of sample identity and labels.  RAW bytes encode
uint8 voxels with W varying fastest, so a WHD manifest becomes a DHW array.
"""

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


CLASS_TO_INDEX = {"normal": 0, "defective": 1}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}
TRAIN_DATASET_NAME = "multiboard_head40_toml"
TEST_DATASET_NAME = "test_3boards_head40_toml"
TRAIN_DATASET_ROOT_ENV = "DRILL_HEAD40_TOML_TRAIN_ROOT"
TEST_DATASET_ROOT_ENV = "DRILL_HEAD40_TOML_TEST_ROOT"
SUPPORTED_DIFFICULTIES = frozenset({"clear", "fuzzy", "reviewed_defect", "supplemental"})
STANDARDIZED_HEAD_SIDE = "high_depth_index"
RAW_DTYPE = np.uint8
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
NORMALIZATION_EPSILON = 1e-6
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_CONTAINER = Path("325_275_and_sphere_data") / "钻孔数据325_275"
_ROOT_MARKERS = ("source_audit.csv", "normal_samples_by_board", "defective_samples_by_board")
_MANIFEST_COLUMNS = frozenset({
    "sample_id", "board", "sequence", "label", "label_index", "difficulty",
    "raw_shape_whd", "input_shape_dhw", "raw_file_name", "raw_relative_path",
    "raw_sha256", "validation_fold", "b_head_up", "original_head_side",
    "direction_flip_required", "direction_standardized", "standardized_head_side",
})


def is_dataset_root(path: Path, expected_name: str | None = None) -> bool:
    path = Path(path)
    return (expected_name is None or path.name == expected_name) and all(
        (path / marker).exists() for marker in _ROOT_MARKERS
    )


def mounted_drive_roots() -> list[Path]:
    if os.name != "nt":
        return [Path("/")]
    available = int(ctypes.windll.kernel32.GetLogicalDrives())
    return [Path(f"{chr(65 + index)}:\\") for index in range(26) if available & (1 << index)]


def resolve_dataset_root(
    explicit_root: str | Path | None = None,
    stored_root: str | Path | None = None,
    expected_name: str = TRAIN_DATASET_NAME,
) -> Path:
    """Locate a dataset without depending on this computer's drive letters.

    Precedence matches the original protocol: explicit path, environment,
    project datasets directory, stored checkpoint path, mounted-drive search.
    A supplied explicit/environment path is authoritative and must be valid.
    """
    env_name = TRAIN_DATASET_ROOT_ENV if expected_name == TRAIN_DATASET_NAME else TEST_DATASET_ROOT_ENV

    def checked(value: str | Path) -> Path:
        resolved = Path(value).expanduser().resolve()
        if not is_dataset_root(resolved, expected_name):
            raise FileNotFoundError(f"Not the expected dataset '{expected_name}': {resolved}")
        return resolved

    if explicit_root is not None:
        return checked(explicit_root)
    from_environment = os.environ.get(env_name, "").strip()
    if from_environment:
        return checked(from_environment)

    local_path = _PROJECT_ROOT / "datasets" / expected_name
    if is_dataset_root(local_path, expected_name):
        return local_path.resolve()
    if stored_root:
        remembered = Path(stored_root).expanduser().resolve()
        if is_dataset_root(remembered, expected_name):
            return remembered

    matches = {
        candidate.resolve()
        for drive in mounted_drive_roots()
        for candidate in [drive / _DATA_CONTAINER / expected_name]
        if is_dataset_root(candidate, expected_name)
    }
    if len(matches) == 1:
        return next(iter(matches))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {expected_name}. Pass its path explicitly or set {env_name}."
        )
    candidates = "\n  ".join(sorted(map(str, matches), key=str.lower))
    raise RuntimeError(f"Multiple matching datasets were found:\n  {candidates}")


@dataclass(frozen=True)
class RawSample:
    sample_id: str
    board: str
    sequence: str
    label: str
    label_index: int
    difficulty: str
    raw_path: Path
    raw_file_name: str
    raw_sha256: str
    validation_fold: int
    shape_whd: tuple[int, int, int]
    shape_dhw: tuple[int, int, int]
    b_head_up: bool
    direction_flip_required: bool
    standardized_head_side: str


def get_normalization_params() -> dict[str, object]:
    return {
        "method": "sample_percentile",
        "scope": "per_sample_head40_roi",
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
        "axis": "D",
        "target_head_side": STANDARDIZED_HEAD_SIDE,
        "bHeadUp_false": "keep",
        "bHeadUp_true": "flip_D",
        "application_stage": "before_augmentation_and_normalization",
        "physical_raw_files_modified": False,
    }


def validate_normalization_params(params: dict[str, object]) -> None:
    if not isinstance(params, dict):
        raise ValueError("Checkpoint has no normalization metadata")
    expected = get_normalization_params()
    for key in ("method", "scope", "lower_percentile", "upper_percentile"):
        actual = params.get(key)
        if key.endswith("percentile"):
            try:
                matches = bool(np.isclose(float(actual), float(expected[key])))
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected[key]
        if not matches:
            raise ValueError(f"Normalization mismatch for {key}: expected {expected[key]}, got {actual}")


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(map(int, value.lower().split("x")))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid shape: {value}") from exc
    if len(parts) != 3 or min(parts) <= 0:
        raise ValueError(f"Invalid shape: {value}")
    return parts


def is_ambiguous_difficulty(difficulty: str) -> bool:
    value = str(difficulty).strip().lower()
    if value not in SUPPORTED_DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty {difficulty!r}; expected {sorted(SUPPORTED_DIFFICULTIES)}")
    return value == "fuzzy"


def _read_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Invalid direction boolean: {value!r}")
    return normalized == "true"


def _sample_from_row(row: dict[str, str], root: Path, verify_files: bool) -> RawSample:
    label, label_index = row["label"].strip(), int(row["label_index"])
    if CLASS_TO_INDEX.get(label) != label_index:
        raise ValueError("Label/index mismatch")
    whd, dhw = parse_shape(row["raw_shape_whd"]), parse_shape(row["input_shape_dhw"])
    if dhw != whd[::-1]:
        raise ValueError(f"WHD/DHW mismatch: {whd} and {dhw}")
    if whd[2] != 40 or whd[:2] not in {(29, 29), (37, 37), (49, 49)}:
        raise ValueError(f"Unexpected TOML Head40 shape: {whd}")

    head_up = _read_boolean(row["b_head_up"])
    flip = _read_boolean(row["direction_flip_required"])
    if not _read_boolean(row["direction_standardized"]) or flip != head_up:
        raise ValueError("Direction standardization/flip metadata is inconsistent")
    original_side = "low_depth_index" if head_up else "high_depth_index"
    if row["original_head_side"].strip() != original_side:
        raise ValueError("Original head side mismatch")
    if row["standardized_head_side"].strip() != STANDARDIZED_HEAD_SIDE:
        raise ValueError("Target head side mismatch")

    identity = row["sample_id"].strip()
    if not identity:
        raise ValueError("SampleID must not be empty")
    difficulty = row["difficulty"].strip()
    is_ambiguous_difficulty(difficulty)
    path = (root / row["raw_relative_path"].strip()).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"RAW path escapes dataset root: {path}") from exc
    if verify_files:
        if not path.is_file():
            raise FileNotFoundError(f"RAW file not found: {path}")
        expected_bytes = math.prod(whd) * np.dtype(RAW_DTYPE).itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f"RAW size mismatch for {path}: expected {expected_bytes}, got {actual_bytes}")

    return RawSample(
        sample_id=identity, board=row["board"].strip(), sequence=row["sequence"].strip(),
        label=label, label_index=label_index, difficulty=difficulty,
        raw_path=path, raw_file_name=row["raw_file_name"].strip(),
        raw_sha256=row["raw_sha256"].strip().lower(), validation_fold=int(row["validation_fold"]),
        shape_whd=whd, shape_dhw=dhw, b_head_up=head_up,
        direction_flip_required=flip, standardized_head_side=STANDARDIZED_HEAD_SIDE,
    )


def load_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path | None = None,
    verify_files: bool = True,
    expected_dataset_name: str = TRAIN_DATASET_NAME,
) -> list[RawSample]:
    manifest = Path(manifest_path).resolve()
    root = resolve_dataset_root(dataset_root, expected_name=expected_dataset_name)
    samples: list[RawSample] = []
    seen: set[str] = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        missing = _MANIFEST_COLUMNS.difference(rows.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for line, row in enumerate(rows, 2):
            try:
                sample = _sample_from_row(row, root, verify_files)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"{manifest}:{line}: {exc}") from exc
            if sample.sample_id in seen:
                raise ValueError(f"Duplicate sample ID at {manifest}:{line}: {sample.sample_id}")
            seen.add(sample.sample_id)
            samples.append(sample)
    if not samples:
        raise ValueError(f"Manifest is empty: {manifest}")
    return samples


def normalize_percentile(volume: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return ((volume - lower) / (upper - lower + NORMALIZATION_EPSILON)).clamp(0.0, 1.0)


def _augment_geometry(volume: torch.Tensor) -> torch.Tensor:
    # Tensor axes are C,D,H,W. Random geometry never reverses D.
    if torch.rand(1).item() < 0.5:
        volume = volume.flip([int(torch.randint(2, 4, (1,)).item())])
    if torch.rand(1).item() < 0.5:
        volume = volume.rot90(int(torch.randint(1, 4, (1,)).item()), [2, 3])
    if torch.rand(1).item() < 0.5:
        dy = int(torch.randint(-3, 4, (1,)).item())
        dx = int(torch.randint(-3, 4, (1,)).item())
        height, width = volume.shape[-2:]
        top, bottom = max(0, dy), max(0, -dy)
        left, right = max(0, dx), max(0, -dx)
        volume = F.pad(volume, (left, right, top, bottom), mode="constant", value=0)
        volume = volume[:, :, bottom:bottom + height, right:right + width]
    return volume


def _augment_intensity(volume: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        sigma = 0.01 + 0.02 * torch.rand(1).item()
        volume = (volume + torch.randn_like(volume) * sigma).clamp(0, 1)
    if torch.rand(1).item() < 0.5:
        gain = 0.9 + 0.2 * torch.rand(1).item()
        volume = (volume * gain).clamp(0, 1)
    if torch.rand(1).item() < 0.5:
        contrast = 0.9 + 0.2 * torch.rand(1).item()
        volume = ((volume - 0.5) * contrast + 0.5).clamp(0, 1)
    return volume


class Head40Dataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path | None = None,
        augment: bool = False,
        verify_files: bool = True,
        expected_dataset_name: str = TRAIN_DATASET_NAME,
        num_views: int = 1,
    ) -> None:
        if num_views not in (1, 2):
            raise ValueError("num_views must be 1 or 2")
        self.manifest_path = Path(manifest_path).resolve()
        self.dataset_root = resolve_dataset_root(dataset_root, expected_name=expected_dataset_name)
        self.samples = load_manifest(self.manifest_path, self.dataset_root, verify_files, expected_dataset_name)
        self.augment = bool(augment)
        self.num_views = int(num_views)
        self._percentile_cache: dict[str, tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def shapes(self) -> list[tuple[int, int, int]]:
        return [sample.shape_dhw for sample in self.samples]

    def _read_volume(self, sample: RawSample) -> tuple[torch.Tensor, float, float]:
        raw = np.fromfile(sample.raw_path, dtype=RAW_DTYPE)
        expected = math.prod(sample.shape_dhw)
        if raw.size != expected:
            raise ValueError(f"RAW element count changed for {sample.raw_path}: expected {expected}, got {raw.size}")
        raw = raw.reshape(sample.shape_dhw)
        if sample.sample_id not in self._percentile_cache:
            low, high = np.percentile(raw, [LOWER_PERCENTILE, UPPER_PERCENTILE])
            self._percentile_cache[sample.sample_id] = float(low), float(high)
        volume = torch.from_numpy(raw.copy()).float().unsqueeze(0)
        if sample.direction_flip_required:
            volume = volume.flip([1])
        lower, upper = self._percentile_cache[sample.sample_id]
        return volume, lower, upper

    def _make_view(self, volume: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        if self.augment:
            volume = _augment_geometry(volume)
        volume = normalize_percentile(volume, lower, upper)
        return _augment_intensity(volume) if self.augment else volume

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        raw_volume, lower, upper = self._read_volume(sample)
        item = {
            "volume": self._make_view(raw_volume, lower, upper),
            "target": torch.tensor(sample.label_index, dtype=torch.long),
            "sample_id": sample.sample_id,
            "board": sample.board,
            "sequence": sample.sequence,
            "difficulty": sample.difficulty,
            "ambiguous": torch.tensor(is_ambiguous_difficulty(sample.difficulty), dtype=torch.bool),
            "raw_shape_whd": "x".join(map(str, sample.shape_whd)),
            "input_shape_dhw": "x".join(map(str, sample.shape_dhw)),
            "raw_path": str(sample.raw_path),
            "b_head_up": sample.b_head_up,
            "direction_flipped": sample.direction_flip_required,
            "direction_standardized": True,
            "standardized_head_side": sample.standardized_head_side,
        }
        if self.num_views == 2:
            item["volume_view2"] = self._make_view(raw_volume, lower, upper)
        return item


class ShapeBatchSampler(Sampler[list[int]]):
    """Keep each batch shape-homogeneous and preserve every sample by default."""

    def __init__(
        self, shapes: list[tuple[int, int, int]], batch_size: int,
        shuffle: bool = False, seed: int = 42, drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.drop_last, self.epoch = drop_last, 0
        self.groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, shape in enumerate(shapes):
            self.groups[tuple(shape)].append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for shape in sorted(self.groups):
            indices = self.groups[shape].copy()
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(
                indices[start:start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
                if not self.drop_last or start + self.batch_size <= len(indices)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum(
            len(group) // self.batch_size if self.drop_last else math.ceil(len(group) / self.batch_size)
            for group in self.groups.values()
        )


def worker_init_fn(worker_id: int) -> None:
    del worker_id
    worker = torch.utils.data.get_worker_info()
    seed = (torch.initial_seed() if worker is None else worker.seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    manifest_path: str | Path,
    dataset_root: str | Path | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    augment: bool = False,
    seed: int = 42,
    verify_files: bool = True,
    expected_dataset_name: str = TRAIN_DATASET_NAME,
    num_views: int = 1,
) -> DataLoader:
    dataset = Head40Dataset(
        manifest_path, dataset_root, augment, verify_files, expected_dataset_name, num_views,
    )
    return DataLoader(
        dataset,
        batch_sampler=ShapeBatchSampler(dataset.shapes, batch_size, shuffle=augment, seed=seed),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=num_workers > 0,
    )


def get_fold_loaders(
    split_root: str | Path,
    fold_index: int,
    dataset_root: str | Path | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
    verify_files: bool = True,
) -> tuple[DataLoader, DataLoader]:
    fold_directory = Path(split_root).resolve() / f"fold_{fold_index}"
    common = dict(dataset_root=dataset_root, batch_size=batch_size,
                  num_workers=num_workers, seed=seed, verify_files=verify_files)
    training = make_loader(fold_directory / "train.csv", augment=True, **common)
    validation = make_loader(fold_directory / "val.csv", augment=False, **common)
    return training, validation


def summarize_dataset(dataset: Head40Dataset) -> dict[str, object]:
    classes = Counter(sample.label for sample in dataset.samples)
    shapes = Counter("x".join(map(str, sample.shape_whd)) for sample in dataset.samples)
    difficulties = Counter(sample.difficulty.strip().lower() for sample in dataset.samples)
    return {
        "total": len(dataset), "normal": classes["normal"], "defective": classes["defective"],
        "boards": len({sample.board for sample in dataset.samples}),
        "raw_shape_whd_counts": dict(sorted(shapes.items())),
        "b_head_up_false": sum(not sample.b_head_up for sample in dataset.samples),
        "b_head_up_true": sum(sample.b_head_up for sample in dataset.samples),
        "direction_flipped": sum(sample.direction_flip_required for sample in dataset.samples),
        "standardized_head_side": STANDARDIZED_HEAD_SIDE,
        "difficulty_counts": dict(sorted(difficulties.items())),
        "ambiguous_definition": "difficulty == 'fuzzy'",
        "ambiguous_sample_count": difficulties["fuzzy"],
        "clear_supervision_sample_count": len(dataset) - difficulties["fuzzy"],
    }
