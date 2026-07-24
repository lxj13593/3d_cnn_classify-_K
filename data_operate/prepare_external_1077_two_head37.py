import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "datasets" / "external_test"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "external_test_head37"
BOARD_NAME = "1077_two"
RAW_DTYPE = np.uint8
FOREGROUND_THRESHOLD = 20
SOURCE_CLASS_DIRECTORIES = {
    "drill_normal_3d": ("normal", 0),
    "drill_defect_3d": ("defective", 1),
}
SHAPE_PATTERN = re.compile(r"_(\d+)_(\d+)_(\d+)-")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-crop the head37 ROI for external board 1077_two."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def parse_raw_shape(file_name):
    match = SHAPE_PATTERN.search(file_name)
    if match is None:
        raise ValueError(f"Cannot parse W_H_D dimensions from filename: {file_name}")
    width, height, depth = (int(value) for value in match.groups())
    return depth, height, width


def replace_shape_in_name(file_name, side_length):
    replacement = f"_{side_length}_{side_length}_{side_length}-"
    output_name, count = SHAPE_PATTERN.subn(replacement, file_name, count=1)
    if count != 1:
        raise ValueError(f"Cannot replace dimensions in filename: {file_name}")
    return output_name


def read_manifest(board_dir):
    manifest_path = board_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rows = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_name = row["DrillFile"]
            if raw_name in rows:
                raise ValueError(f"Duplicate DrillFile in {manifest_path}: {raw_name}")
            rows[raw_name] = row
    return rows


def locate_right_head(picture_path, expected_depth):
    with Image.open(picture_path) as image:
        gray = np.asarray(image.convert("L"))

    if gray.shape[1] != expected_depth:
        raise ValueError(
            f"Picture/raw depth mismatch for {picture_path}: "
            f"picture width={gray.shape[1]}, raw depth={expected_depth}"
        )

    columns = np.flatnonzero((gray > FOREGROUND_THRESHOLD).any(axis=0))
    if columns.size == 0:
        raise ValueError(f"No foreground found in picture: {picture_path}")
    return {
        "picture_width": int(gray.shape[1]),
        "foreground_left": int(columns[0]),
        "head_end": int(columns[-1]),
    }


def crop_head(volume, picture_path):
    depth, height, width = volume.shape
    if height != width or height not in (29, 37):
        raise ValueError(
            f"Unsupported source shape {volume.shape}; expected D x 29 x 29 "
            "or D x 37 x 37"
        )

    head_length = height
    location = locate_right_head(picture_path, expected_depth=depth)
    crop_end = location["head_end"] + 1
    crop_start = max(0, crop_end - head_length)
    crop = volume[crop_start:crop_end]
    left_pad = head_length - crop.shape[0]
    if left_pad > 0:
        padding = np.zeros((left_pad, height, width), dtype=volume.dtype)
        crop = np.concatenate((padding, crop), axis=0)

    expected_shape = (head_length, height, width)
    if crop.shape != expected_shape:
        raise ValueError(
            f"Bad crop shape for {picture_path}: {crop.shape}, expected {expected_shape}"
        )

    return crop, {
        **location,
        "crop_start": crop_start,
        "crop_end_exclusive": crop_end,
        "left_pad": left_pad,
    }


def save_check_picture(source_path, output_path, location):
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    start = location["crop_start"]
    end = location["head_end"]
    draw.rectangle((start, 0, end, image.height - 1), fill=(80, 180, 80, 45))
    draw.line((start, 0, start, image.height - 1), fill=(0, 255, 0, 255), width=2)
    draw.line((end, 0, end, image.height - 1), fill=(0, 120, 255, 255), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def prepare_board(source_root, output_root, board):
    source_board = source_root / board
    output_board = output_root / board
    manifest = read_manifest(source_board)
    metadata_rows = []
    labeled_source_names = set()
    expected_outputs = {"normal": set(), "defective": set()}

    for source_directory, (output_class, label) in SOURCE_CLASS_DIRECTORIES.items():
        source_class_dir = source_board / source_directory
        if not source_class_dir.is_dir():
            raise FileNotFoundError(f"Missing source class directory: {source_class_dir}")

        output_class_dir = output_board / output_class
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for raw_path in sorted(source_class_dir.glob("*.raw")):
            if raw_path.name not in manifest:
                raise KeyError(f"{raw_path.name} is missing from {source_board / 'manifest.csv'}")
            if raw_path.name in labeled_source_names:
                raise ValueError(f"Duplicate labeled source sample: {raw_path.name}")
            labeled_source_names.add(raw_path.name)

            manifest_row = manifest[raw_path.name]
            picture_path = source_board / "drill_feature" / manifest_row["FeatureFile"]
            if not picture_path.is_file():
                raise FileNotFoundError(f"Missing paired picture: {picture_path}")

            source_shape = parse_raw_shape(raw_path.name)
            expected_bytes = int(np.prod(source_shape)) * np.dtype(RAW_DTYPE).itemsize
            if raw_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"RAW byte count mismatch for {raw_path}: got {raw_path.stat().st_size}, "
                    f"expected {expected_bytes} for {source_shape}"
                )

            volume = np.fromfile(raw_path, dtype=RAW_DTYPE).reshape(source_shape)
            crop, location = crop_head(volume, picture_path)
            output_name = replace_shape_in_name(raw_path.name, crop.shape[0])
            output_path = output_class_dir / output_name
            crop.astype(RAW_DTYPE, copy=False).tofile(output_path)
            expected_outputs[output_class].add(output_name)

            check_path = output_board / "check_images" / f"{Path(output_name).stem}.png"
            save_check_picture(picture_path, check_path, location)

            metadata_rows.append(
                {
                    "Board": board,
                    "SampleOrder": int(manifest_row["SampleOrder"]),
                    "Label": label,
                    "ClassName": output_class,
                    "SourceRawFile": raw_path.name,
                    "PreparedRawFile": output_name,
                    "SourcePictureFile": picture_path.name,
                    "SourceShape": "x".join(str(value) for value in source_shape),
                    "OutputShape": "x".join(str(value) for value in crop.shape),
                    "PictureWidth": location["picture_width"],
                    "ForegroundLeft": location["foreground_left"],
                    "HeadEnd": location["head_end"],
                    "CropStart": location["crop_start"],
                    "CropEndExclusive": location["crop_end_exclusive"],
                    "LeftPad": location["left_pad"],
                    "CheckImage": str(check_path.relative_to(output_board)),
                }
            )

    if labeled_source_names != set(manifest):
        missing = sorted(set(manifest) - labeled_source_names)
        raise ValueError(f"Board {board} has unlabeled manifest rows: {missing[:5]}")
    if len(metadata_rows) != 490:
        raise ValueError(f"Board {board} produced {len(metadata_rows)} crops; expected 490")

    for class_name, expected_names in expected_outputs.items():
        actual_names = {path.name for path in (output_board / class_name).glob("*.raw")}
        if actual_names != expected_names:
            stale = sorted(actual_names - expected_names)
            missing = sorted(expected_names - actual_names)
            raise ValueError(
                f"Output mismatch for board {board}/{class_name}: "
                f"stale={stale[:5]}, missing={missing[:5]}"
            )

    metadata_rows.sort(key=lambda row: row["SampleOrder"])
    metadata_path = output_board / "manifest_head.csv"
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)

    shape_counts = {}
    label_counts = {"normal": 0, "defective": 0}
    for row in metadata_rows:
        shape_counts[row["OutputShape"]] = shape_counts.get(row["OutputShape"], 0) + 1
        label_counts[row["ClassName"]] += 1
    return shape_counts, label_counts, metadata_path


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print("Head rule: rightmost 2D foreground; crop side equals source H/W (29 or 37).")
    shape_counts, label_counts, metadata_path = prepare_board(
        source_root, output_root, BOARD_NAME
    )
    print(
        f"Board {BOARD_NAME}: shapes={shape_counts}, labels={label_counts}, "
        f"metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
