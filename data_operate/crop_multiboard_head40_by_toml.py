from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


RAW_DTYPE = np.uint8
HEAD_LENGTH = 40
NORMAL_DIR = "normal_samples_by_board"
DEFECT_DIR = "defective_samples_by_board"
CLASS_DIRS = {"normal": NORMAL_DIR, "defective": DEFECT_DIR}
SEQUENCE_PATTERN = re.compile(r"^drill-(\d+)-", re.IGNORECASE)
SHAPE_PATTERN = re.compile(r"_(\d+)_(\d+)_(\d+)-")
SOURCE_MANIFEST = "source_audit.csv"
TOML_NAME = "InsightInspection.toml"


@dataclass(frozen=True)
class CropTask:
    source_row: dict[str, str]
    class_dir: str
    board: str
    file_sequence: int
    toml_sequence: int
    source_raw: Path
    source_feature: Path
    source_toml: Path
    source_shape_whd: tuple[int, int, int]
    center2: float
    center_index: int
    b_head_up: bool
    crop_start: int
    crop_end: int
    output_raw_name: str
    output_feature_name: str
    source_raw_sha256: str
    source_feature_sha256: str
    source_raw_mtime_ns: int
    source_feature_mtime_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop mixed-size drill RAW volumes to Head40 using center2 and "
            "bHeadUp from each board's TOML."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("SOURCE_ROOT", "OUTPUT_ROOT"),
        required=True,
        help="May be supplied more than once to process multiple datasets.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def copy_file_exact(source: Path, destination: Path) -> None:
    """Copy bytes without Windows CopyFile2, which rejects some long paths."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("wb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle, length=1024 * 1024)


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"Invalid RawShape: {value}") from exc
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"Invalid RawShape: {value}")
    return shape


def sequence_from_name(name: str) -> int:
    match = SEQUENCE_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Cannot parse drill sequence from: {name}")
    return int(match.group(1))


def replace_depth_in_name(name: str, shape_whd: tuple[int, int, int]) -> str:
    width, height, depth = shape_whd
    expected = f"_{width}_{height}_{depth}-"
    replacement = f"_{width}_{height}_{HEAD_LENGTH}-"
    if name.count(expected) != 1:
        raise ValueError(f"Expected one shape token {expected!r} in {name}")
    output_name = name.replace(expected, replacement, 1)
    match = SHAPE_PATTERN.search(output_name)
    if match is None or tuple(map(int, match.groups())) != (width, height, HEAD_LENGTH):
        raise ValueError(f"Failed to update RAW shape in file name: {name}")
    return output_name


def round_half_up_nonnegative(value: float) -> int:
    if value < 0:
        raise ValueError(f"Cannot round negative center2: {value}")
    return int(math.floor(value + 0.5))


def class_dir_for_label(label: str) -> str:
    try:
        return CLASS_DIRS[label.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported label: {label}") from exc


def verify_board_tomls(source_root: Path, board: str) -> Path:
    candidates = [
        source_root / class_dir / board / TOML_NAME
        for class_dir in (NORMAL_DIR, DEFECT_DIR)
        if (source_root / class_dir / board / TOML_NAME).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No {TOML_NAME} for board {board}")
    hashes = {sha256_file(path) for path in candidates}
    if len(hashes) != 1:
        raise ValueError(f"Normal/defective TOML copies differ for board {board}")
    return candidates[0]


def load_toml_records(path: Path) -> dict[str, dict[str, object]]:
    with path.open("rb") as handle:
        records = tomllib.load(handle).get("drillOutputSeq")
    if not isinstance(records, dict):
        raise ValueError(f"Missing [drillOutputSeq] in {path}")
    return records


def preflight_dataset(source_root: Path) -> tuple[list[str], list[CropTask]]:
    source_root = source_root.resolve()
    source_fields, rows = read_csv(source_root / SOURCE_MANIFEST)
    required = {
        "SampleID",
        "Board",
        "Sequence",
        "Label",
        "LabelIndex",
        "Difficulty",
        "RawShape",
        "RawFileName",
        "FeatureFileName",
        "RawSHA256",
        "FeatureSHA256",
    }
    missing = required - set(source_fields)
    if missing:
        raise ValueError(f"{SOURCE_MANIFEST} is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError(f"No samples in {source_root / SOURCE_MANIFEST}")

    toml_paths: dict[str, Path] = {}
    toml_records: dict[str, dict[str, dict[str, object]]] = {}
    for board in sorted({row["Board"].strip() for row in rows}):
        toml_path = verify_board_tomls(source_root, board)
        toml_paths[board] = toml_path
        toml_records[board] = load_toml_records(toml_path)

    tasks: list[CropTask] = []
    output_keys: set[tuple[str, str, str]] = set()
    sample_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        label = row["Label"].strip().lower()
        class_dir = class_dir_for_label(label)
        if int(row["LabelIndex"]) != (0 if label == "normal" else 1):
            raise ValueError(f"Label/index mismatch at row {row_number}")
        board = row["Board"].strip()
        sample_id = row["SampleID"].strip()
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate SampleID at row {row_number}: {sample_id}")
        sample_ids.add(sample_id)

        source_raw = source_root / class_dir / board / "drill" / row["RawFileName"]
        source_feature = (
            source_root / class_dir / board / "drill_feature" / row["FeatureFileName"]
        )
        if not source_raw.is_file():
            raise FileNotFoundError(f"Missing source RAW: {source_raw}")
        if not source_feature.is_file():
            raise FileNotFoundError(f"Missing source feature image: {source_feature}")

        file_sequence = int(row["Sequence"])
        if sequence_from_name(source_raw.name) != file_sequence:
            raise ValueError(f"RAW/manifest sequence mismatch: {source_raw}")
        if sequence_from_name(source_feature.name) != file_sequence:
            raise ValueError(f"Feature/manifest sequence mismatch: {source_feature}")
        if file_sequence < 1:
            raise ValueError(f"File sequence must be >= 1: {source_raw}")

        toml_sequence = file_sequence - 1
        record = toml_records[board].get(str(toml_sequence))
        if record is None:
            raise KeyError(
                f"File sequence {file_sequence} has no TOML record {toml_sequence}: "
                f"{source_raw}"
            )
        if int(record.get("drillNo", -1)) != toml_sequence:
            raise ValueError(
                f"TOML drillNo mismatch for file {file_sequence}: "
                f"expected {toml_sequence}, got {record.get('drillNo')}"
            )

        center2 = float(record.get("center2", -1.0))
        center_index = round_half_up_nonnegative(center2)
        b_head_up = record.get("bHeadUp")
        if not isinstance(b_head_up, bool):
            raise ValueError(f"bHeadUp is not boolean for {source_raw}")
        crop_start = center_index - (25 if b_head_up else 15)
        crop_end = crop_start + HEAD_LENGTH

        shape_whd = parse_shape(row["RawShape"])
        width, height, depth = shape_whd
        if source_raw.stat().st_size != width * height * depth:
            raise ValueError(
                f"RAW size mismatch for {source_raw}: expected {width * height * depth}, "
                f"got {source_raw.stat().st_size}"
            )
        if crop_start < 0 or crop_end > depth:
            raise ValueError(
                f"Head40 crop is out of range for {source_raw}: "
                f"center2={center2}, range=[{crop_start}, {crop_end}), depth={depth}"
            )
        with Image.open(source_feature) as image:
            if image.width != depth or image.height != 2 * height:
                raise ValueError(
                    f"Feature/RAW dimensions disagree for {source_feature}: "
                    f"image={image.size}, expected=({depth}, {2 * height})"
                )

        source_raw_hash = sha256_file(source_raw)
        source_feature_hash = sha256_file(source_feature)
        if source_raw_hash != row["RawSHA256"].strip().upper():
            raise ValueError(f"Source RAW hash mismatch: {source_raw}")
        if source_feature_hash != row["FeatureSHA256"].strip().upper():
            raise ValueError(f"Source feature hash mismatch: {source_feature}")

        output_raw_name = replace_depth_in_name(source_raw.name, shape_whd)
        output_key = (class_dir, board, output_raw_name.lower())
        if output_key in output_keys:
            raise ValueError(f"Duplicate output RAW path: {output_key}")
        output_keys.add(output_key)
        tasks.append(
            CropTask(
                source_row=row,
                class_dir=class_dir,
                board=board,
                file_sequence=file_sequence,
                toml_sequence=toml_sequence,
                source_raw=source_raw,
                source_feature=source_feature,
                source_toml=toml_paths[board],
                source_shape_whd=shape_whd,
                center2=center2,
                center_index=center_index,
                b_head_up=b_head_up,
                crop_start=crop_start,
                crop_end=crop_end,
                output_raw_name=output_raw_name,
                output_feature_name=source_feature.name,
                source_raw_sha256=source_raw_hash,
                source_feature_sha256=source_feature_hash,
                source_raw_mtime_ns=source_raw.stat().st_mtime_ns,
                source_feature_mtime_ns=source_feature.stat().st_mtime_ns,
            )
        )
    return source_fields, tasks


def draw_check_image(
    source_feature: Path,
    output_path: Path,
    center2: float,
    crop_start: int,
    crop_end: int,
) -> None:
    with Image.open(source_feature) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle(
        (crop_start, 0, crop_end - 1, image.height - 1),
        fill=(50, 205, 220, 58),
    )
    draw.line(
        (crop_start, 0, crop_start, image.height - 1),
        fill=(37, 190, 90, 255),
        width=2,
    )
    draw.line(
        (crop_end, 0, crop_end, image.height - 1),
        fill=(35, 120, 255, 255),
        width=2,
    )
    center_x = int(math.floor(center2 + 0.5))
    draw.line(
        (center_x, 0, center_x, image.height - 1),
        fill=(255, 205, 35, 255),
        width=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def crop_feature_image(
    source_feature: Path,
    output_path: Path,
    crop_start: int,
    crop_end: int,
) -> None:
    with Image.open(source_feature) as source:
        crop = source.crop((crop_start, 0, crop_end, source.height)).convert("RGB")
    if crop.width != HEAD_LENGTH:
        raise ValueError(f"Bad 2D Head40 crop width for {source_feature}: {crop.width}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)


def output_metadata_row(
    task: CropTask,
    final_output_root: Path,
    output_raw_sha256: str,
) -> dict[str, object]:
    width, height, depth = task.source_shape_whd
    final_board = final_output_root / task.class_dir / task.board
    output_raw = final_board / "drill" / task.output_raw_name
    output_feature = final_board / "drill_feature" / task.output_feature_name
    crop_picture_name = f"{Path(task.output_raw_name).stem}.png"
    row: dict[str, object] = dict(task.source_row)
    row.update(
        {
            "RawShape": f"{width}x{height}x{HEAD_LENGTH}",
            "RawFileName": task.output_raw_name,
            "SourceRawPath": str(task.source_raw),
            "SourceFeaturePath": str(task.source_feature),
            "DestinationRawPath": str(output_raw),
            "DestinationFeaturePath": str(output_feature),
            "RawSHA256": output_raw_sha256,
            "FeatureSHA256": task.source_feature_sha256,
            "FullVolumeRawShape": f"{width}x{height}x{depth}",
            "FullVolumeRawFileName": task.source_raw.name,
            "FullVolumeRawSHA256": task.source_raw_sha256,
            "FileSequence": f"{task.file_sequence:04d}",
            "TomlSequence": task.toml_sequence,
            "SequenceOffsetRule": "toml_sequence=file_sequence-1",
            "Center2": f"{task.center2:.10g}",
            "CenterIndex": task.center_index,
            "BHeadUp": str(task.b_head_up).lower(),
            "HeadSide": "left" if task.b_head_up else "right",
            "LeftDistance": 25 if task.b_head_up else 15,
            "RightDistance": 15 if task.b_head_up else 25,
            "CropStart": task.crop_start,
            "CropEndExclusive": task.crop_end,
            "OutputDepth": HEAD_LENGTH,
            "DirectionFlipped": "false",
            "Head40FeatureFileName": crop_picture_name,
            "CheckImageFileName": crop_picture_name,
        }
    )
    return row


def make_summaries(rows: list[dict[str, object]]) -> tuple[list[dict], list[dict]]:
    by_board: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_board[str(row["Board"])].append(row)
    board_rows = []
    for board, board_samples in sorted(by_board.items()):
        labels = Counter(str(row["Label"]) for row in board_samples)
        directions = Counter(str(row["BHeadUp"]) for row in board_samples)
        shapes = Counter(str(row["RawShape"]) for row in board_samples)
        board_rows.append(
            {
                "Board": board,
                "NormalCount": labels["normal"],
                "DefectiveCount": labels["defective"],
                "TotalCount": len(board_samples),
                "BHeadUpFalse": directions["false"],
                "BHeadUpTrue": directions["true"],
                "OutputShapeCounts": "; ".join(
                    f"{shape}: {count}" for shape, count in sorted(shapes.items())
                ),
            }
        )
    labels = Counter(str(row["Label"]) for row in rows)
    label_rows = [
        {"Label": label, "Count": labels[label]}
        for label in ("normal", "defective")
    ]
    return board_rows, label_rows


def process_dataset(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    staging_root = output_root.with_name(f"{output_root.name}.building")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    if staging_root.exists():
        raise FileExistsError(f"Staging output already exists: {staging_root}")

    print(f"Preflight: {source_root}", flush=True)
    source_fields, tasks = preflight_dataset(source_root)
    print(
        f"Preflight passed: samples={len(tasks)}, "
        f"boards={len({task.board for task in tasks})}, "
        f"bHeadUp=true={sum(task.b_head_up for task in tasks)}, "
        f"bHeadUp=false={sum(not task.b_head_up for task in tasks)}",
        flush=True,
    )

    staging_root.mkdir(parents=True)
    output_rows: list[dict[str, object]] = []
    copied_tomls: set[tuple[str, str]] = set()
    for index, task in enumerate(tasks, start=1):
        width, height, depth = task.source_shape_whd
        source_volume = np.fromfile(task.source_raw, dtype=RAW_DTYPE).reshape(
            (depth, height, width)
        )
        crop = np.ascontiguousarray(source_volume[task.crop_start : task.crop_end])
        if crop.shape != (HEAD_LENGTH, height, width):
            raise ValueError(f"Bad Head40 crop shape for {task.source_raw}: {crop.shape}")

        staging_board = staging_root / task.class_dir / task.board
        staging_raw = staging_board / "drill" / task.output_raw_name
        staging_feature = staging_board / "drill_feature" / task.output_feature_name
        crop_picture_name = f"{Path(task.output_raw_name).stem}.png"
        staging_crop_picture = (
            staging_board / "drill_feature_head40" / crop_picture_name
        )
        staging_check = staging_board / "check_images" / crop_picture_name
        staging_raw.parent.mkdir(parents=True, exist_ok=True)
        staging_feature.parent.mkdir(parents=True, exist_ok=True)
        crop.tofile(staging_raw)
        expected_hash = hashlib.sha256(crop.tobytes(order="C")).hexdigest().upper()
        output_hash = sha256_file(staging_raw)
        if output_hash != expected_hash:
            raise ValueError(f"Written RAW hash mismatch: {staging_raw}")
        copy_file_exact(task.source_feature, staging_feature)
        if sha256_file(staging_feature) != task.source_feature_sha256:
            raise ValueError(f"Copied feature hash mismatch: {staging_feature}")
        crop_feature_image(
            task.source_feature,
            staging_crop_picture,
            task.crop_start,
            task.crop_end,
        )
        draw_check_image(
            task.source_feature,
            staging_check,
            task.center2,
            task.crop_start,
            task.crop_end,
        )

        toml_key = (task.class_dir, task.board)
        if toml_key not in copied_tomls:
            destination_toml = staging_board / TOML_NAME
            copy_file_exact(task.source_toml, destination_toml)
            if sha256_file(destination_toml) != sha256_file(task.source_toml):
                raise ValueError(f"Copied TOML hash mismatch: {destination_toml}")
            copied_tomls.add(toml_key)

        output_rows.append(output_metadata_row(task, output_root, output_hash))
        if index % 250 == 0 or index == len(tasks):
            print(f"  Cropped {index}/{len(tasks)}", flush=True)

    extra_fields = [
        "FullVolumeRawShape",
        "FullVolumeRawFileName",
        "FullVolumeRawSHA256",
        "FileSequence",
        "TomlSequence",
        "SequenceOffsetRule",
        "Center2",
        "CenterIndex",
        "BHeadUp",
        "HeadSide",
        "LeftDistance",
        "RightDistance",
        "CropStart",
        "CropEndExclusive",
        "OutputDepth",
        "DirectionFlipped",
        "Head40FeatureFileName",
        "CheckImageFileName",
    ]
    output_fields = source_fields + [field for field in extra_fields if field not in source_fields]
    write_csv(staging_root / SOURCE_MANIFEST, output_rows, output_fields)
    write_csv(staging_root / "all_samples_manifest.csv", output_rows, output_fields)
    write_csv(staging_root / "crop_audit.csv", output_rows, output_fields)

    per_board_class: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in output_rows:
        class_dir = class_dir_for_label(str(row["Label"]))
        per_board_class[(class_dir, str(row["Board"]))].append(row)
    for (class_dir, board), board_rows in per_board_class.items():
        write_csv(
            staging_root / class_dir / board / "manifest.csv",
            board_rows,
            output_fields,
        )

    board_rows, label_rows = make_summaries(output_rows)
    write_csv(staging_root / "board_summary.csv", board_rows)
    write_csv(staging_root / "label_summary.csv", label_rows)
    shape_counts = Counter(str(row["RawShape"]) for row in output_rows)
    integrity_rows = [
        {"Check": "SourceDataset", "Value": str(source_root)},
        {"Check": "SampleCount", "Value": len(tasks)},
        {"Check": "BoardCount", "Value": len({task.board for task in tasks})},
        {"Check": "TOMLMapping", "Value": "file_sequence-1"},
        {"Check": "OutputDepth", "Value": HEAD_LENGTH},
        {"Check": "BHeadUpFalseRule", "Value": "left15_right25"},
        {"Check": "BHeadUpTrueRule", "Value": "left25_right15"},
        {"Check": "DirectionFlipped", "Value": "false"},
        {
            "Check": "OutputShapeCounts",
            "Value": "; ".join(
                f"{shape}: {count}" for shape, count in sorted(shape_counts.items())
            ),
        },
    ]
    write_csv(staging_root / "integrity_summary.csv", integrity_rows)
    (staging_root / "README.txt").write_text(
        "TOML Head40 dataset\n"
        "===================\n"
        "Mapping: file sequence N -> TOML drillOutputSeq.(N-1).\n"
        "center2 is rounded to the nearest integer using half-up rounding.\n"
        "bHeadUp=false: [center-15, center+25), head remains on the right.\n"
        "bHeadUp=true:  [center-25, center+15), head remains on the left.\n"
        "Every output RAW has depth 40. No direction flip, interpolation, "
        "padding, or clipping is applied.\n"
        "drill_feature_head40 contains the cropped 2D views.\n"
        "check_images contains full 2D views with green/blue boundaries and "
        "a yellow center line.\n",
        encoding="utf-8",
    )

    expected_count = len(tasks)
    raw_files = list(staging_root.glob("*_samples_by_board/*/drill/*.raw"))
    full_features = list(staging_root.glob("*_samples_by_board/*/drill_feature/*"))
    crop_features = list(
        staging_root.glob("*_samples_by_board/*/drill_feature_head40/*.png")
    )
    check_images = list(staging_root.glob("*_samples_by_board/*/check_images/*.png"))
    if not all(
        count == expected_count
        for count in (
            len(raw_files),
            len(full_features),
            len(crop_features),
            len(check_images),
        )
    ):
        raise ValueError(
            "Output count mismatch: "
            f"raw={len(raw_files)}, full2d={len(full_features)}, "
            f"crop2d={len(crop_features)}, checks={len(check_images)}, "
            f"expected={expected_count}"
        )
    for task in tasks:
        if (
            task.source_raw.stat().st_size
            != math.prod(task.source_shape_whd) * np.dtype(RAW_DTYPE).itemsize
            or task.source_raw.stat().st_mtime_ns != task.source_raw_mtime_ns
            or task.source_feature.stat().st_mtime_ns != task.source_feature_mtime_ns
        ):
            raise ValueError(f"Source file changed during processing: {task.source_raw}")

    staging_root.replace(output_root)
    print(f"Completed: {output_root}", flush=True)


def main() -> None:
    args = parse_args()
    for source_value, output_value in args.dataset:
        process_dataset(Path(source_value), Path(output_value))


if __name__ == "__main__":
    main()
