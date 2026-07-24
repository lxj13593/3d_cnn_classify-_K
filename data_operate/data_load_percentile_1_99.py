import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from data_operate.config import (
        BATCH_SIZE,
        CLASS_NAMES,
        CLASS_TO_IDX,
        DATA_DIR,
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
        DATA_DIR,
        NUM_WORKERS,
        RANDOM_SEED,
        RAW_DTYPE,
        VOLUME_SHAPE,
    )


NORMALIZATION_METHOD = "sample_percentile"
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
NORMALIZATION_EPSILON = 1e-6


def get_normalization_params():
    return {
        "method": NORMALIZATION_METHOD,
        "scope": "per_sample",
        "lower_percentile": LOWER_PERCENTILE,
        "upper_percentile": UPPER_PERCENTILE,
        "clip_min": 0.0,
        "clip_max": 1.0,
    }


def validate_normalization_params(params):
    if not isinstance(params, dict):
        raise ValueError("Checkpoint has no percentile normalization metadata")
    if params.get("method") != NORMALIZATION_METHOD:
        raise ValueError(
            "Checkpoint normalization method does not match this experiment: "
            f"expected {NORMALIZATION_METHOD}, got {params.get('method')}"
        )
    if params.get("scope") != "per_sample":
        raise ValueError(
            "Checkpoint normalization scope must be per_sample, got "
            f"{params.get('scope')}"
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


def normalize_sample_percentile(data, lower, upper):
    data = (data - lower) / (upper - lower + NORMALIZATION_EPSILON)
    return torch.clamp(data, 0.0, 1.0)


class RawCTDataset(Dataset):
    def __init__(
        self,
        root_dir,
        shape=VOLUME_SHAPE,
        dtype=RAW_DTYPE,
        augment=False,
    ):
        self.root_dir = root_dir
        self.shape = tuple(shape)
        self.dtype = dtype
        self.augment = augment
        self.normalization_params = get_normalization_params()
        self.classes = list(CLASS_NAMES)
        self.class_to_idx = dict(CLASS_TO_IDX)

        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            if not os.path.isdir(class_dir):
                print(f"Warning: class directory does not exist: {class_dir}")
                continue
            for current_root, _, file_names in sorted(os.walk(class_dir)):
                for file_name in sorted(file_names):
                    if file_name.lower().endswith(".raw"):
                        self.samples.append(
                            (
                                os.path.join(current_root, file_name),
                                self.class_to_idx[target_class],
                            )
                        )

        print(f"Loaded {len(self.samples)} samples from {root_dir}")

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
        path, target = self.samples[index]
        raw = np.fromfile(path, dtype=self.dtype).reshape(self.shape)
        lower, upper = sample_percentile_bounds(raw)
        data = torch.from_numpy(raw.copy()).float().unsqueeze(0)

        if self.augment:
            data = self._geometric_augmentation(data)

        data = normalize_sample_percentile(data, lower, upper)

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
    shape=VOLUME_SHAPE,
    dtype=RAW_DTYPE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=RANDOM_SEED,
):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    print(
        "Normalization: per-sample percentile "
        f"P{LOWER_PERCENTILE:g}/P{UPPER_PERCENTILE:g}, clipped to [0, 1]"
    )

    train_dataset = RawCTDataset(
        train_dir,
        shape=shape,
        dtype=dtype,
        augment=True,
    )
    val_dataset = RawCTDataset(
        val_dir,
        shape=shape,
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
    shape=VOLUME_SHAPE,
    dtype=RAW_DTYPE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    test_dataset = RawCTDataset(
        os.path.join(data_root, "test"),
        shape=shape,
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


if __name__ == "__main__":
    print(f"Data root: {DATA_DIR}")
    train_loader, val_loader = get_data_loaders(DATA_DIR)
    for loader_name, loader in (("train", train_loader), ("val", val_loader)):
        batch_data, batch_labels = next(iter(loader))
        print(
            loader_name,
            tuple(batch_data.shape),
            tuple(batch_labels.shape),
            float(batch_data.min()),
            float(batch_data.max()),
        )
