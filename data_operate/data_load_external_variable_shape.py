import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


RAW_DTYPE = np.uint8
SHAPE_PATTERN = re.compile(r"_(\d+)_(\d+)_(\d+)-")
FULL_CLASS_DIRECTORIES = {
    "drill_normal_3d": 0,
    "drill_defect_3d": 1,
}
HEAD_CLASS_DIRECTORIES = {
    "normal": 0,
    "defective": 1,
}


@dataclass(frozen=True)
class ExternalSample:
    board: str
    sample_order: int
    raw_path: Path
    label: int
    input_shape: tuple
    source_raw_file: str
    source_picture_file: str
    source_shape: str
    crop_metadata: dict

    @property
    def sample_key(self):
        return f"{self.board}/{self.source_raw_file}"


def parse_raw_shape(file_name):
    """Parse filename dimensions W_H_D and return numpy layout D_H_W."""
    match = SHAPE_PATTERN.search(file_name)
    if match is None:
        raise ValueError(f"Cannot parse W_H_D dimensions from filename: {file_name}")
    width, height, depth = (int(value) for value in match.groups())
    return depth, height, width


def read_csv_by_key(path, key):
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row[key]
            if value in rows:
                raise ValueError(f"Duplicate {key} in {path}: {value}")
            rows[value] = row
    return rows


def build_full_samples(board_dir):
    manifest_path = board_dir / "manifest.csv"
    manifest = read_csv_by_key(manifest_path, "DrillFile")
    samples = []
    labeled_names = set()

    for class_directory, label in FULL_CLASS_DIRECTORIES.items():
        class_dir = board_dir / class_directory
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing labeled class directory: {class_dir}")
        for raw_path in sorted(class_dir.glob("*.raw")):
            if raw_path.name not in manifest:
                raise KeyError(f"{raw_path.name} is missing from {manifest_path}")
            if raw_path.name in labeled_names:
                raise ValueError(f"Sample appears in both label directories: {raw_path.name}")
            labeled_names.add(raw_path.name)

            row = manifest[raw_path.name]
            shape = parse_raw_shape(raw_path.name)
            samples.append(
                ExternalSample(
                    board=board_dir.name,
                    sample_order=int(row["SampleOrder"]),
                    raw_path=raw_path,
                    label=label,
                    input_shape=shape,
                    source_raw_file=raw_path.name,
                    source_picture_file=row["FeatureFile"],
                    source_shape="x".join(str(value) for value in shape),
                    crop_metadata={
                        "foreground_left": -1,
                        "head_end": -1,
                        "crop_start": 0,
                        "crop_end_exclusive": shape[0],
                        "left_pad": 0,
                    },
                )
            )

    if labeled_names != set(manifest):
        missing = sorted(set(manifest) - labeled_names)
        raise ValueError(f"Board {board_dir.name} has unlabeled manifest rows: {missing[:5]}")
    return samples


def build_prepared_head_samples(board_dir):
    manifest_path = board_dir / "manifest_head.csv"
    manifest = read_csv_by_key(manifest_path, "PreparedRawFile")
    samples = []
    labeled_names = set()

    for class_directory, label in HEAD_CLASS_DIRECTORIES.items():
        class_dir = board_dir / class_directory
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing prepared class directory: {class_dir}")
        for raw_path in sorted(class_dir.glob("*.raw")):
            if raw_path.name not in manifest:
                raise KeyError(f"{raw_path.name} is missing from {manifest_path}")
            if raw_path.name in labeled_names:
                raise ValueError(f"Prepared sample has duplicate labels: {raw_path.name}")
            labeled_names.add(raw_path.name)

            row = manifest[raw_path.name]
            if int(row["Label"]) != label or row["ClassName"] != class_directory:
                raise ValueError(f"Prepared label mismatch for {raw_path}")
            shape = tuple(int(value) for value in row["OutputShape"].split("x"))
            if parse_raw_shape(raw_path.name) != shape:
                raise ValueError(f"Filename/manifest shape mismatch for {raw_path}")

            samples.append(
                ExternalSample(
                    board=board_dir.name,
                    sample_order=int(row["SampleOrder"]),
                    raw_path=raw_path,
                    label=label,
                    input_shape=shape,
                    source_raw_file=row["SourceRawFile"],
                    source_picture_file=row["SourcePictureFile"],
                    source_shape=row["SourceShape"],
                    crop_metadata={
                        "foreground_left": int(row["ForegroundLeft"]),
                        "head_end": int(row["HeadEnd"]),
                        "crop_start": int(row["CropStart"]),
                        "crop_end_exclusive": int(row["CropEndExclusive"]),
                        "left_pad": int(row["LeftPad"]),
                    },
                )
            )

    if labeled_names != set(manifest):
        missing = sorted(set(manifest) - labeled_names)
        raise ValueError(
            f"Prepared board {board_dir.name} has manifest rows without RAW files: {missing[:5]}"
        )
    return samples


def build_external_samples(board_dir, dataset_kind):
    board_dir = Path(board_dir)
    if dataset_kind == "full":
        samples = build_full_samples(board_dir)
    elif dataset_kind == "head":
        samples = build_prepared_head_samples(board_dir)
    else:
        raise ValueError(f"Unsupported dataset kind: {dataset_kind}")

    samples.sort(key=lambda sample: sample.sample_order)
    if len(samples) != 490:
        raise ValueError(
            f"Board {board_dir.name} has {len(samples)} samples; expected 490"
        )

    sample_keys = [sample.sample_key for sample in samples]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError(f"Board {board_dir.name} contains duplicate source samples")

    sample_orders = [sample.sample_order for sample in samples]
    if len(set(sample_orders)) != len(sample_orders):
        raise ValueError(f"Board {board_dir.name} contains duplicate SampleOrder values")

    for sample in samples:
        expected_bytes = int(np.prod(sample.input_shape)) * np.dtype(RAW_DTYPE).itemsize
        if sample.raw_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"RAW byte count mismatch for {sample.raw_path}: "
                f"got {sample.raw_path.stat().st_size}, expected {expected_bytes}"
            )
    return samples


class ExternalBoardDataset(Dataset):
    """Read full RAW data or already-prepared head RAW data without cropping."""

    def __init__(self, board_dir, dataset_kind, global_min, global_max):
        if global_min is None or global_max is None:
            raise ValueError("Training-set normalization parameters are required")

        global_min = float(global_min)
        global_max = float(global_max)
        if not np.isfinite(global_min) or not np.isfinite(global_max):
            raise ValueError("Training-set normalization parameters must be finite")
        if global_max <= global_min:
            raise ValueError(
                "Training-set global_max must be greater than global_min"
            )

        self.board_dir = Path(board_dir)
        self.dataset_kind = dataset_kind
        self.global_min = global_min
        self.global_max = global_max
        self.samples = build_external_samples(self.board_dir, dataset_kind)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        volume = np.fromfile(sample.raw_path, dtype=RAW_DTYPE).reshape(sample.input_shape)
        tensor = torch.from_numpy(volume.copy()).float().unsqueeze(0)
        denominator = self.global_max - self.global_min + 1e-6
        tensor = torch.clamp((tensor - self.global_min) / denominator, 0.0, 1.0)

        metadata = {
            "sample_key": sample.sample_key,
            "board": sample.board,
            "sample_order": sample.sample_order,
            "raw_file": sample.source_raw_file,
            "picture_file": sample.source_picture_file,
            "raw_shape": sample.source_shape,
            "input_shape": "x".join(str(value) for value in sample.input_shape),
            **sample.crop_metadata,
        }
        return tensor, sample.label, metadata


def single_item_collate(batch):
    if len(batch) != 1:
        raise ValueError("Variable-shape external testing requires batch_size=1")
    return batch[0]


def get_external_loader(
    board_dir,
    dataset_kind,
    global_min,
    global_max,
    num_workers=0,
):
    dataset = ExternalBoardDataset(
        board_dir=board_dir,
        dataset_kind=dataset_kind,
        global_min=global_min,
        global_max=global_max,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=single_item_collate,
    )
