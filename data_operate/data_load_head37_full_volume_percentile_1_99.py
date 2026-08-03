import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from data_operate.config import (
        BATCH_SIZE,
        CLASS_NAMES,
        CLASS_TO_IDX,
        NUM_WORKERS,
        RANDOM_SEED,
        RAW_DTYPE,
        VOLUME_SHAPE,
    )
except ImportError:
    from config import (
        BATCH_SIZE,
        CLASS_NAMES,
        CLASS_TO_IDX,
        NUM_WORKERS,
        RANDOM_SEED,
        RAW_DTYPE,
        VOLUME_SHAPE,
    )


HEAD_REGION_VOLUME_SHAPE = (37, 37, 37)
NORMALIZATION_METHOD = "sample_percentile"
NORMALIZATION_SCOPE = "per_sample"
STATISTICS_SOURCE = "corresponding_full_volume"
NORMALIZATION_TARGET = "head_roi"
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
NORMALIZATION_EPSILON = 1e-6

# Five-fold folders contain copies of the same samples. This process-local cache
# avoids recalculating a sample's full-volume percentiles when it changes split.
_FULL_VOLUME_BOUNDS_CACHE = {}


def get_normalization_params():
    return {
        "method": NORMALIZATION_METHOD,
        "scope": NORMALIZATION_SCOPE,
        "statistics_source": STATISTICS_SOURCE,
        "normalization_target": NORMALIZATION_TARGET,
        "lower_percentile": LOWER_PERCENTILE,
        "upper_percentile": UPPER_PERCENTILE,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "processing_order": "full_volume_percentiles_then_normalize_head_roi",
    }


def validate_normalization_params(params):
    if not isinstance(params, dict):
        raise ValueError("Checkpoint has no full-volume percentile metadata")

    expected_strings = {
        "method": NORMALIZATION_METHOD,
        "scope": NORMALIZATION_SCOPE,
        "statistics_source": STATISTICS_SOURCE,
        "normalization_target": NORMALIZATION_TARGET,
        "processing_order": "full_volume_percentiles_then_normalize_head_roi",
    }
    for key, expected in expected_strings.items():
        actual = params.get(key)
        if actual != expected:
            raise ValueError(
                f"Checkpoint normalization {key} mismatch: "
                f"expected {expected}, got {actual}"
            )

    lower = float(params.get("lower_percentile", float("nan")))
    upper = float(params.get("upper_percentile", float("nan")))
    if not np.isclose(lower, LOWER_PERCENTILE) or not np.isclose(
        upper, UPPER_PERCENTILE
    ):
        raise ValueError(
            "Checkpoint percentile range does not match this experiment: "
            f"expected P{LOWER_PERCENTILE:g}/P{UPPER_PERCENTILE:g}, "
            f"got P{lower:g}/P{upper:g}"
        )
    return get_normalization_params()


def sample_percentile_bounds(data):
    lower, upper = np.percentile(
        data,
        [LOWER_PERCENTILE, UPPER_PERCENTILE],
    )
    return float(lower), float(upper)


def normalize_head_with_full_volume_bounds(data, lower, upper):
    data = (data - lower) / (upper - lower + NORMALIZATION_EPSILON)
    return torch.clamp(data, 0.0, 1.0)


def _cache_namespace(full_root_dir):
    full_root_dir = Path(full_root_dir).resolve()
    fold_dir = full_root_dir.parent
    if fold_dir.name.startswith("fold_"):
        return str(fold_dir.parent)
    return str(full_root_dir)


def _expected_bytes(shape, dtype):
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


class HeadROIFullVolumePercentileDataset(Dataset):
    """Normalize an existing Head37 crop with its full drill's P1/P99 bounds."""

    def __init__(
        self,
        root_dir,
        full_root_dir,
        shape=HEAD_REGION_VOLUME_SHAPE,
        full_shape=VOLUME_SHAPE,
        dtype=RAW_DTYPE,
        augment=False,
    ):
        self.root_dir = Path(root_dir)
        self.full_root_dir = Path(full_root_dir)
        self.shape = tuple(shape)
        self.full_shape = tuple(full_shape)
        self.dtype = dtype
        self.augment = augment
        self.normalization_params = get_normalization_params()
        self.classes = list(CLASS_NAMES)
        self.class_to_idx = dict(CLASS_TO_IDX)
        self.samples = []

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Missing Head37 data directory: {self.root_dir}")
        if not self.full_root_dir.is_dir():
            raise FileNotFoundError(
                f"Missing corresponding full-volume directory: {self.full_root_dir}"
            )

        head_bytes = _expected_bytes(self.shape, self.dtype)
        full_bytes = _expected_bytes(self.full_shape, self.dtype)
        namespace = _cache_namespace(self.full_root_dir)
        cache_hits = 0
        cache_misses = 0

        pending_samples = []
        for target_class in self.classes:
            head_class_dir = self.root_dir / target_class
            full_class_dir = self.full_root_dir / target_class
            if not head_class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing Head37 class directory: {head_class_dir}"
                )
            if not full_class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing full-volume class directory: {full_class_dir}"
                )

            for head_path in sorted(head_class_dir.rglob("*.raw")):
                relative_path = head_path.relative_to(head_class_dir)
                full_path = full_class_dir / relative_path
                if not full_path.is_file():
                    raise FileNotFoundError(
                        "Head37 sample has no corresponding full volume: "
                        f"{head_path} -> {full_path}"
                    )
                if head_path.stat().st_size != head_bytes:
                    raise ValueError(
                        f"Head37 RAW byte count mismatch for {head_path}: "
                        f"got {head_path.stat().st_size}, expected {head_bytes}"
                    )
                if full_path.stat().st_size != full_bytes:
                    raise ValueError(
                        f"Full RAW byte count mismatch for {full_path}: "
                        f"got {full_path.stat().st_size}, expected {full_bytes}"
                    )

                cache_key = (
                    namespace,
                    target_class,
                    relative_path.as_posix().lower(),
                )
                pending_samples.append(
                    (
                        head_path,
                        full_path,
                        self.class_to_idx[target_class],
                        cache_key,
                    )
                )

        print(
            f"Preparing full-volume P{LOWER_PERCENTILE:g}/P{UPPER_PERCENTILE:g} "
            f"bounds for {len(pending_samples)} Head37 samples from {self.root_dir}"
        )
        for head_path, full_path, target, cache_key in pending_samples:
            if cache_key in _FULL_VOLUME_BOUNDS_CACHE:
                lower, upper = _FULL_VOLUME_BOUNDS_CACHE[cache_key]
                cache_hits += 1
            else:
                full_volume = np.fromfile(full_path, dtype=self.dtype).reshape(
                    self.full_shape
                )
                lower, upper = sample_percentile_bounds(full_volume)
                _FULL_VOLUME_BOUNDS_CACHE[cache_key] = (lower, upper)
                cache_misses += 1
            self.samples.append((head_path, target, lower, upper))

        print(
            f"Loaded {len(self.samples)} Head37 samples; full-volume percentile "
            f"cache hits={cache_hits}, calculated={cache_misses}"
        )

    def __len__(self):
        return len(self.samples)

    def _geometric_augmentation(self, data):
        if torch.rand(1).item() > 0.5:
            flip_dim = torch.randint(2, 4, (1,)).item()
            data = torch.flip(data, dims=[flip_dim])

        if torch.rand(1).item() > 0.5:
            rotation_k = torch.randint(1, 4, (1,)).item()
            data = torch.rot90(data, k=rotation_k, dims=[2, 3])

        if torch.rand(1).item() > 0.5:
            shift_y = torch.randint(-3, 4, (1,)).item()
            shift_x = torch.randint(-3, 4, (1,)).item()
            pad_y1 = max(0, shift_y)
            pad_y2 = max(0, -shift_y)
            pad_x1 = max(0, shift_x)
            pad_x2 = max(0, -shift_x)
            padded = torch.nn.functional.pad(
                data,
                (pad_x1, pad_x2, pad_y1, pad_y2),
                mode="constant",
                value=0,
            )
            start_y = pad_y2
            start_x = pad_x2
            data = padded[
                :,
                :,
                start_y : start_y + self.shape[1],
                start_x : start_x + self.shape[2],
            ]
        return data

    @staticmethod
    def _intensity_augmentation(data):
        if torch.rand(1).item() > 0.5:
            noise_std = 0.01 + torch.rand(1).item() * 0.02
            data = torch.clamp(data + torch.randn_like(data) * noise_std, 0.0, 1.0)

        if torch.rand(1).item() > 0.5:
            brightness_factor = 0.9 + torch.rand(1).item() * 0.2
            data = torch.clamp(data * brightness_factor, 0.0, 1.0)

        if torch.rand(1).item() > 0.5:
            contrast_factor = 0.9 + torch.rand(1).item() * 0.2
            data = torch.clamp(
                (data - 0.5) * contrast_factor + 0.5,
                0.0,
                1.0,
            )
        return data

    def __getitem__(self, index):
        head_path, target, lower, upper = self.samples[index]
        head_volume = np.fromfile(head_path, dtype=self.dtype).reshape(self.shape)
        data = torch.from_numpy(head_volume.copy()).float().unsqueeze(0)

        if self.augment:
            data = self._geometric_augmentation(data)

        data = normalize_head_with_full_volume_bounds(data, lower, upper)

        if self.augment:
            data = self._intensity_augmentation(data)

        return data, target


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed % (2**32)
    else:
        seed = RANDOM_SEED
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_data_loaders(
    data_root,
    full_data_root,
    shape=HEAD_REGION_VOLUME_SHAPE,
    full_shape=VOLUME_SHAPE,
    dtype=RAW_DTYPE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=RANDOM_SEED,
):
    print(
        "Normalization: each Head37 sample uses P1/P99 calculated from its "
        "corresponding full volume, clipped to [0, 1]"
    )
    train_dataset = HeadROIFullVolumePercentileDataset(
        Path(data_root) / "train",
        Path(full_data_root) / "train",
        shape=shape,
        full_shape=full_shape,
        dtype=dtype,
        augment=True,
    )
    val_dataset = HeadROIFullVolumePercentileDataset(
        Path(data_root) / "val",
        Path(full_data_root) / "val",
        shape=shape,
        full_shape=full_shape,
        dtype=dtype,
        augment=False,
    )

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(seed),
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    )


def get_test_loader(
    data_root,
    full_data_root,
    shape=HEAD_REGION_VOLUME_SHAPE,
    full_shape=VOLUME_SHAPE,
    dtype=RAW_DTYPE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    test_dataset = HeadROIFullVolumePercentileDataset(
        Path(data_root) / "test",
        Path(full_data_root) / "test",
        shape=shape,
        full_shape=full_shape,
        dtype=dtype,
        augment=False,
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
