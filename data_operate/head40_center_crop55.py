"""Small preprocessing adapter on the existing loader; no new data/splits."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_operate import data_load_resnet18_head40_toml_direction_unified_multiboard as original


def cropped_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    d, h, w = shape
    if d != 40 or (h, w) not in {(29, 29), (37, 37), (49, 49)}:
        raise ValueError(f"Expected original Head40 shape, got {shape}")
    return d, (55 * h + 99) // 100, (55 * w + 99) // 100


def crop_box(shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    _, h, w = shape
    _, nh, nw = cropped_shape(shape)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    return y0, y0 + nh, x0, x0 + nw


def preprocessing_metadata() -> dict[str, object]:
    return {
        "method": "head40_geometric_center_crop_hw",
        "version": "center55_v1",
        "fraction": 0.55,
        "axes": ["H", "W"],
        "depth": 40,
        "rounding": "ceil(55*n/100)",
        "start": "floor((n-kept)/2)",
        "source_shapes_dhw": [[40, 29, 29], [40, 37, 37], [40, 49, 49]],
        "output_shapes_dhw": [[40, 16, 16], [40, 21, 21], [40, 27, 27]],
        "application_stage": "after_full_head40_percentiles_before_direction_flip_and_augmentation",
        "normalization_statistics_source": "original_uncropped_head40_P1_P99",
        "training_validation_test_same_crop": True,
        "resize": False,
        "denoising": False,
        "physical_raw_files_modified": False,
        "odd_even_center_offset": "29->16 removes 6 and 7 voxels; center offset -0.5 voxel",
    }


class CenterCrop55Dataset(original.MultiBoardRawDataset):
    """Reuse all manifest, RAW, direction, normalization and augmentation logic."""

    @property
    def shapes(self) -> list[tuple[int, int, int]]:
        return [cropped_shape(s.shape_dhw) for s in self.samples]

    def _read_volume(self, sample):
        # Compute/cache bounds on the FULL original Head40, just as before.
        raw, lower, upper = super()._read_volume(sample)
        y0, y1, x0, x1 = crop_box(sample.shape_dhw)
        return raw[:, y0:y1, x0:x1].copy(), lower, upper

    def __getitem__(self, index: int) -> dict[str, object]:
        row = super().__getitem__(index)
        shape = cropped_shape(self.samples[index].shape_dhw)
        if tuple(row["volume"].shape) != (1, *shape):
            raise ValueError("Actual crop shape mismatch")
        row["input_shape_dhw"] = "x".join(str(v) for v in shape)
        return row


def make_loader(
    manifest_path: str | Path,
    dataset_root: str | Path | None,
    batch_size: int,
    num_workers: int,
    augment: bool,
    seed: int,
    verify_files: bool = True,
    expected_dataset_name: str = original.TRAIN_DATASET_NAME,
) -> DataLoader:
    dataset = CenterCrop55Dataset(
        manifest_path, augment=augment, dataset_root=dataset_root,
        verify_files=verify_files, expected_dataset_name=expected_dataset_name,
    )
    sampler = original.ShapeBatchSampler(
        dataset.shapes, batch_size=batch_size, shuffle=augment, seed=seed, drop_last=False,
    )
    kwargs = dict(
        dataset=dataset, batch_sampler=sampler, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=original.worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )
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
    common = dict(dataset_root=dataset_root, batch_size=batch_size,
                  num_workers=num_workers, seed=seed, verify_files=verify_files)
    return (
        make_loader(fold_dir / "train.csv", augment=True, **common),
        make_loader(fold_dir / "val.csv", augment=False, **common),
    )


def summarize_dataset(dataset: CenterCrop55Dataset) -> dict[str, object]:
    from collections import Counter
    summary = original.summarize_dataset(dataset)
    summary["input_shape_dhw_counts"] = dict(sorted(Counter(
        "x".join(str(v) for v in shape) for shape in dataset.shapes
    ).items()))
    summary["input_preprocessing"] = preprocessing_metadata()
    return summary
