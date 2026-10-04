"""Head40 loader for the clear-defect-1.25/fuzzy-0.5 experiment.

The normal/defective target remains binary.  The separate ``ambiguous`` flag is
derived from ``difficulty == 'fuzzy'`` and is used only to select the training
loss weight.  It never becomes a third prediction class.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_operate import data_load_resnet18_head40_toml_direction_unified_multiboard as shared


AMBIGUOUS_DIFFICULTY = "fuzzy"
SUPPORTED_DIFFICULTIES = frozenset({"clear", AMBIGUOUS_DIFFICULTY})

# Re-export the unchanged shared protocol and helpers for the experiment entry
# points.  Keeping them sourced from one module prevents data-preprocessing
# drift relative to the Head40-533 baseline.
CLASS_TO_INDEX = shared.CLASS_TO_INDEX
TRAIN_DATASET_NAME = shared.TRAIN_DATASET_NAME
TEST_DATASET_NAME = shared.TEST_DATASET_NAME
ShapeBatchSampler = shared.ShapeBatchSampler
get_augmentation_params = shared.get_augmentation_params
get_direction_standardization_params = shared.get_direction_standardization_params
get_normalization_params = shared.get_normalization_params
resolve_dataset_root = shared.resolve_dataset_root
validate_normalization_params = shared.validate_normalization_params
worker_init_fn = shared.worker_init_fn


def is_ambiguous_difficulty(difficulty: str) -> bool:
    """Return whether a sample belongs to either fuzzy binary-label group."""
    normalized = str(difficulty).strip().lower()
    if normalized not in SUPPORTED_DIFFICULTIES:
        raise ValueError(
            f"Unsupported difficulty {difficulty!r}; expected one of "
            f"{sorted(SUPPORTED_DIFFICULTIES)}"
        )
    return normalized == AMBIGUOUS_DIFFICULTY


class ReliabilityWeightedMultiBoardRawDataset(shared.MultiBoardRawDataset):
    """Direction-unified Head40 dataset with a fuzzy flag per sample."""

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        item["ambiguous"] = torch.tensor(
            is_ambiguous_difficulty(str(item["difficulty"])), dtype=torch.bool
        )
        return item


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
    dataset = ReliabilityWeightedMultiBoardRawDataset(
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


def summarize_dataset(dataset: ReliabilityWeightedMultiBoardRawDataset) -> dict[str, object]:
    summary = dict(shared.summarize_dataset(dataset))
    difficulty_counts = Counter(
        str(sample.difficulty).strip().lower() for sample in dataset.samples
    )
    unknown = set(difficulty_counts) - SUPPORTED_DIFFICULTIES
    if unknown:
        raise ValueError(f"Unknown difficulties in manifest: {sorted(unknown)}")
    summary["difficulty_counts"] = dict(sorted(difficulty_counts.items()))
    summary["ambiguous_definition"] = f"difficulty == {AMBIGUOUS_DIFFICULTY!r}"
    summary["ambiguous_sample_count"] = int(difficulty_counts[AMBIGUOUS_DIFFICULTY])
    summary["clear_supervision_sample_count"] = int(
        len(dataset.samples) - difficulty_counts[AMBIGUOUS_DIFFICULTY]
    )
    difficulty_label_counts = Counter(
        (str(sample.difficulty).strip().lower(), str(sample.label).strip().lower())
        for sample in dataset.samples
    )
    summary["difficulty_label_counts"] = {
        f"{difficulty}_{label}": int(count)
        for (difficulty, label), count in sorted(difficulty_label_counts.items())
    }
    return summary
