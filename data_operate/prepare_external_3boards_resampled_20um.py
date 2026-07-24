import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_operate.data_load_external_variable_shape import (  # noqa: E402
    RAW_DTYPE,
    build_external_samples,
)


TARGET_SPACING_UM = 20
DEFAULT_BOARDS = ("1022", "1077", "1305")
DEFAULT_FULL_SOURCE = PROJECT_ROOT / "datasets" / "external_test"
DEFAULT_HEAD_SOURCE = PROJECT_ROOT / "datasets" / "external_test_head37"
DEFAULT_FULL_OUTPUT = PROJECT_ROOT / "datasets" / "external_test_20um"
DEFAULT_HEAD_OUTPUT = PROJECT_ROOT / "datasets" / "external_test_head37_20um"

# numpy layout is D, H, W. These mappings come from the VGI spacing audit.
FULL_SHAPE_TO_SPACING_UM = {
    (237, 29, 29): 25,
    (271, 37, 37): 20,
    (291, 37, 37): 20,
    (296, 37, 37): 20,
    (361, 49, 49): 15,
}
HEAD_SHAPE_TO_SPACING_UM = {
    (29, 29, 29): 25,
    (37, 37, 37): 20,
    (49, 49, 49): 15,
}
SHAPE_SUFFIX_PATTERN = re.compile(r"_(\d+)_(\d+)_(\d+)(?=-|$)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create independent 20 um external-test datasets. Full volumes keep "
            "their physical axial coverage; Head37 is re-cropped only after the "
            "full volume has been resampled."
        )
    )
    parser.add_argument("--full-source", type=Path, default=DEFAULT_FULL_SOURCE)
    parser.add_argument("--head-source", type=Path, default=DEFAULT_HEAD_SOURCE)
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_OUTPUT)
    parser.add_argument("--head-output", type=Path, default=DEFAULT_HEAD_OUTPUT)
    parser.add_argument("--boards", nargs="+", default=list(DEFAULT_BOARDS))
    return parser.parse_args()


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shape_text(shape):
    return "x".join(str(value) for value in shape)


def replace_filename_shape(name, target_shape):
    depth, height, width = target_shape
    replacement = f"_{width}_{height}_{depth}"
    updated, count = SHAPE_SUFFIX_PATTERN.subn(replacement, name, count=1)
    if count != 1:
        raise ValueError(f"Cannot replace W_H_D dimensions in name: {name}")
    return updated


def target_full_shape(source_shape, spacing_um):
    depth, _, _ = source_shape
    target_depth = int(math.floor(depth * spacing_um / TARGET_SPACING_UM + 0.5))
    return target_depth, 37, 37


def round_half_up(value):
    return int(math.floor(float(value) + 0.5))


def crop_head_from_resampled_full(volume, crop_end_exclusive, output_depth=37):
    requested_start = crop_end_exclusive - output_depth
    requested_end = crop_end_exclusive
    source_start = max(0, requested_start)
    source_end = min(volume.shape[0], requested_end)
    left_pad = max(0, -requested_start)
    right_pad = max(0, requested_end - volume.shape[0])
    cropped = volume[source_start:source_end]
    if left_pad or right_pad:
        cropped = np.pad(
            cropped,
            ((left_pad, right_pad), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    expected_shape = (output_depth, volume.shape[1], volume.shape[2])
    if cropped.shape != expected_shape:
        raise ValueError(
            f"Head crop has shape {cropped.shape}, expected {expected_shape}; "
            f"end={crop_end_exclusive}, full_shape={volume.shape}"
        )
    return cropped, source_start, source_end, left_pad, right_pad


def resample_or_copy(source_path, destination_path, source_shape, target_shape):
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if tuple(source_shape) == tuple(target_shape):
        shutil.copy2(source_path, destination_path)
        return False

    volume = np.fromfile(source_path, dtype=RAW_DTYPE).reshape(source_shape)
    tensor = torch.from_numpy(volume.copy()).float()[None, None]
    with torch.no_grad():
        resized = F.interpolate(
            tensor,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]
    output = (
        resized.round()
        .clamp(np.iinfo(RAW_DTYPE).min, np.iinfo(RAW_DTYPE).max)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    output.tofile(destination_path)
    return True


def prepare_full_board(source_board, output_board):
    samples = build_external_samples(source_board, "full")
    sample_by_name = {sample.source_raw_file: sample for sample in samples}
    rows, original_fields = read_csv_rows(source_board / "manifest.csv")
    if len(rows) != len(samples):
        raise ValueError(f"Manifest/sample count mismatch for {source_board.name}")

    extra_fields = [
        "SourceDrillFile",
        "SourceInputShape",
        "SourceSpacingUm",
        "TargetInputShape",
        "TargetSpacingUm",
        "WasResampled",
        "Interpolation",
    ]
    output_rows = []
    transformations = Counter()
    output_names = set()

    for row in rows:
        source_name = row["DrillFile"]
        sample = sample_by_name[source_name]
        source_shape = tuple(sample.input_shape)
        if source_shape not in FULL_SHAPE_TO_SPACING_UM:
            raise ValueError(
                f"Unknown full-volume shape/spacing for {sample.raw_path}: {source_shape}"
            )
        source_spacing = FULL_SHAPE_TO_SPACING_UM[source_shape]
        target_shape = target_full_shape(source_shape, source_spacing)
        target_name = replace_filename_shape(source_name, target_shape)
        if target_name in output_names:
            raise ValueError(f"Resampling creates duplicate full RAW name: {target_name}")
        output_names.add(target_name)

        class_directory = "drill_defect_3d" if sample.label == 1 else "drill_normal_3d"
        destination = output_board / class_directory / target_name
        was_resampled = resample_or_copy(
            sample.raw_path,
            destination,
            source_shape,
            target_shape,
        )
        expected_bytes = int(np.prod(target_shape)) * np.dtype(RAW_DTYPE).itemsize
        if destination.stat().st_size != expected_bytes:
            raise ValueError(f"Unexpected output byte count: {destination}")

        output_row = dict(row)
        output_row["DrillFile"] = target_name
        output_row["PairKey"] = replace_filename_shape(row["PairKey"], target_shape)
        output_row["DrillBytes"] = str(expected_bytes)
        output_row.update(
            {
                "SourceDrillFile": source_name,
                "SourceInputShape": shape_text(source_shape),
                "SourceSpacingUm": str(source_spacing),
                "TargetInputShape": shape_text(target_shape),
                "TargetSpacingUm": str(TARGET_SPACING_UM),
                "WasResampled": str(was_resampled),
                "Interpolation": "trilinear_align_corners_false" if was_resampled else "none",
            }
        )
        output_rows.append(output_row)
        transformations[(source_shape, source_spacing, target_shape)] += 1

    write_csv_rows(
        output_board / "manifest.csv",
        output_rows,
        [*original_fields, *extra_fields],
    )
    build_external_samples(output_board, "full")
    return transformations


def prepare_head_board(source_board, full_20um_board, output_board):
    samples = build_external_samples(source_board, "head")
    sample_by_prepared_name = {sample.raw_path.name: sample for sample in samples}
    full_samples = build_external_samples(full_20um_board, "full")
    full_sample_by_name = {
        sample.source_raw_file: sample for sample in full_samples
    }
    full_rows, _ = read_csv_rows(full_20um_board / "manifest.csv")
    full_row_by_source_name = {
        row["SourceDrillFile"]: row for row in full_rows
    }
    rows, original_fields = read_csv_rows(source_board / "manifest_head.csv")
    if len(rows) != len(samples):
        raise ValueError(f"Head manifest/sample count mismatch for {source_board.name}")

    extra_fields = [
        "SourcePreparedRawFile",
        "SourceOutputShape",
        "SourceSpacingUm",
        "TargetSpacingUm",
        "WasResampled",
        "Interpolation",
        "SourcePictureWidth",
        "SourceForegroundLeft",
        "SourceHeadEnd",
        "SourceCropStart",
        "SourceCropEndExclusive",
        "SourceLeftPad",
        "SourceCheckImage",
        "TargetFullRawFile",
        "TargetFullShape",
        "RightPad",
        "CropMethod",
    ]
    output_rows = []
    transformations = Counter()
    output_names = set()
    target_shape = (37, 37, 37)

    for row in rows:
        source_name = row["PreparedRawFile"]
        sample = sample_by_prepared_name[source_name]
        source_shape = tuple(sample.input_shape)
        if source_shape not in HEAD_SHAPE_TO_SPACING_UM:
            raise ValueError(
                f"Unknown head-ROI shape/spacing for {sample.raw_path}: {source_shape}"
            )
        source_spacing = HEAD_SHAPE_TO_SPACING_UM[source_shape]
        target_name = replace_filename_shape(source_name, target_shape)
        if target_name in output_names:
            raise ValueError(f"Resampling creates duplicate head RAW name: {target_name}")
        output_names.add(target_name)

        source_full_name = row["SourceRawFile"]
        if source_full_name not in full_row_by_source_name:
            raise KeyError(
                f"Head source RAW is missing from resampled full manifest: "
                f"{source_full_name}"
            )
        full_row = full_row_by_source_name[source_full_name]
        target_full_name = full_row["DrillFile"]
        full_sample = full_sample_by_name[target_full_name]
        if full_sample.label != sample.label:
            raise ValueError(
                f"Full/head label mismatch for {source_full_name}: "
                f"full={full_sample.label}, head={sample.label}"
            )

        source_full_shape = tuple(
            int(value) for value in row["SourceShape"].split("x")
        )
        target_full_shape_value = tuple(full_sample.input_shape)
        if source_full_shape[1:] != source_shape[1:]:
            raise ValueError(
                f"Head/full source-shape mismatch for {source_full_name}: "
                f"head={source_shape}, full={source_full_shape}"
            )
        if target_full_shape_value[1:] != (37, 37):
            raise ValueError(
                f"Resampled full data must have HxW=37x37: "
                f"{full_sample.raw_path}, shape={target_full_shape_value}"
            )

        full_volume = np.fromfile(full_sample.raw_path, dtype=RAW_DTYPE).reshape(
            target_full_shape_value
        )
        scale_depth = target_full_shape_value[0] / source_full_shape[0]
        source_crop_end = int(row["CropEndExclusive"])
        target_crop_end = round_half_up(source_crop_end * scale_depth)
        cropped, crop_start, crop_end, left_pad, right_pad = (
            crop_head_from_resampled_full(full_volume, target_crop_end)
        )

        destination = output_board / row["ClassName"] / target_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        cropped.tofile(destination)
        was_resampled = source_spacing != TARGET_SPACING_UM
        expected_bytes = int(np.prod(target_shape)) * np.dtype(RAW_DTYPE).itemsize
        if destination.stat().st_size != expected_bytes:
            raise ValueError(f"Unexpected output byte count: {destination}")

        output_row = dict(row)
        output_row["PreparedRawFile"] = target_name
        output_row["OutputShape"] = shape_text(target_shape)
        output_row["PictureWidth"] = str(target_full_shape_value[0])
        output_row["ForegroundLeft"] = str(
            round_half_up(int(row["ForegroundLeft"]) * scale_depth)
        )
        output_row["HeadEnd"] = str(target_crop_end - 1)
        output_row["CropStart"] = str(crop_start)
        output_row["CropEndExclusive"] = str(crop_end)
        output_row["LeftPad"] = str(left_pad)
        output_row["CheckImage"] = ""
        output_row.update(
            {
                "SourcePreparedRawFile": source_name,
                "SourceOutputShape": shape_text(source_shape),
                "SourceSpacingUm": str(source_spacing),
                "TargetSpacingUm": str(TARGET_SPACING_UM),
                "WasResampled": str(was_resampled),
                "Interpolation": full_row["Interpolation"],
                "SourcePictureWidth": row["PictureWidth"],
                "SourceForegroundLeft": row["ForegroundLeft"],
                "SourceHeadEnd": row["HeadEnd"],
                "SourceCropStart": row["CropStart"],
                "SourceCropEndExclusive": row["CropEndExclusive"],
                "SourceLeftPad": row["LeftPad"],
                "SourceCheckImage": row["CheckImage"],
                "TargetFullRawFile": target_full_name,
                "TargetFullShape": shape_text(target_full_shape_value),
                "RightPad": str(right_pad),
                "CropMethod": "resample_full_then_crop_head_end_aligned",
            }
        )
        output_rows.append(output_row)
        transformations[(source_shape, source_spacing, target_shape)] += 1

    write_csv_rows(
        output_board / "manifest_head.csv",
        output_rows,
        [*original_fields, *extra_fields],
    )
    build_external_samples(output_board, "head")
    return transformations


def serialize_transformations(counter):
    return [
        {
            "source_shape": shape_text(source_shape),
            "source_spacing_um": spacing_um,
            "target_shape": shape_text(target_shape),
            "target_spacing_um": TARGET_SPACING_UM,
            "samples": count,
        }
        for (source_shape, spacing_um, target_shape), count in sorted(counter.items())
    ]


def ensure_new_output(path):
    if path.exists():
        raise FileExistsError(
            f"Output already exists; existing data will not be overwritten: {path}"
        )


def main():
    args = parse_args()
    args.full_source = args.full_source.resolve()
    args.head_source = args.head_source.resolve()
    args.full_output = args.full_output.resolve()
    args.head_output = args.head_output.resolve()
    if len(args.boards) != len(set(args.boards)):
        raise ValueError(f"Duplicate board names: {args.boards}")

    ensure_new_output(args.full_output)
    ensure_new_output(args.head_output)
    full_build = args.full_output.with_name(args.full_output.name + ".building")
    head_build = args.head_output.with_name(args.head_output.name + ".building")
    ensure_new_output(full_build)
    ensure_new_output(head_build)
    full_build.mkdir(parents=True)
    head_build.mkdir(parents=True)

    full_summary = {}
    head_summary = {}
    for board in args.boards:
        print(f"Preparing full board {board}...")
        full_summary[board] = serialize_transformations(
            prepare_full_board(args.full_source / board, full_build / board)
        )
        print(f"Preparing Head37 board {board}...")
        head_summary[board] = serialize_transformations(
            prepare_head_board(
                args.head_source / board,
                full_build / board,
                head_build / board,
            )
        )

    common = {
        "boards": args.boards,
        "target_spacing_um": TARGET_SPACING_UM,
        "raw_dtype": np.dtype(RAW_DTYPE).name,
        "interpolation": "trilinear, align_corners=False, rounded to uint8",
        "original_data_unchanged": True,
    }
    (full_build / "resampling_summary.json").write_text(
        json.dumps(
            {
                **common,
                "dataset_kind": "full",
                "source_root": str(args.full_source),
                "transformations": full_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (head_build / "resampling_summary.json").write_text(
        json.dumps(
            {
                **common,
                "dataset_kind": "head37",
                "source_root": str(args.head_source),
                "full_20um_root": str(args.full_output),
                "processing_order": "resample full volume, then crop Head37",
                "transformations": head_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    full_build.rename(args.full_output)
    head_build.rename(args.head_output)
    print(f"Full 20 um data: {args.full_output}")
    print(f"Head37 20 um data: {args.head_output}")


if __name__ == "__main__":
    main()
