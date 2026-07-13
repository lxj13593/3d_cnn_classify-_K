from pathlib import Path
import csv
import re

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
PICTURE_DIR = DATASETS_DIR / "original" / "data" / "all_picture"
RAW_DATA_DIR = DATASETS_DIR / "original" / "data"
RAW_5FOLD_DIR = DATASETS_DIR / "original" / "data_5fold"

OUT_DATA_DIR = DATASETS_DIR / "head_37" / "data"
OUT_5FOLD_DIR = DATASETS_DIR / "head_37" / "data_5fold"
OUT_LINE_DIR = DATASETS_DIR / "head_37" / "check_images"
OUT_META_CSV = OUT_LINE_DIR / "_crop_metadata.csv"

VOLUME_SHAPE = (296, 37, 37)
HEAD_LEN = 37
RAW_DTYPE = np.uint8
FOREGROUND_THRESHOLD = 20
CLASS_NAMES = ("normal", "defective")


def drill_id_from_name(name):
    match = re.search(r"drill-(\d+)-", name)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)", name)
    if match:
        return match.group(1).zfill(4)
    return None


def locate_head_from_picture(picture_path):
    image = Image.open(picture_path).convert("L")
    array = np.asarray(image)
    mask = array > FOREGROUND_THRESHOLD
    columns = np.where(mask.any(axis=0))[0]
    if len(columns) == 0:
        raise ValueError(f"No foreground found in {picture_path}")

    left = int(columns[0])
    head_end = int(columns[-1])
    crop_start = max(0, head_end - HEAD_LEN + 1)
    available_len = head_end - crop_start + 1
    left_pad = HEAD_LEN - available_len
    return {
        "image_width": int(array.shape[1]),
        "foreground_left": left,
        "head_end": head_end,
        "crop_start": crop_start,
        "crop_end_exclusive": head_end + 1,
        "available_len": available_len,
        "left_pad": left_pad,
    }


def build_picture_locations():
    locations = {}
    for picture_path in sorted(PICTURE_DIR.glob("*.jpg")):
        drill_id = drill_id_from_name(picture_path.name)
        if drill_id is None:
            continue
        locations[drill_id] = locate_head_from_picture(picture_path)
    return locations


def crop_raw(raw_path, output_path, location):
    volume = np.fromfile(raw_path, dtype=RAW_DTYPE)
    expected = int(np.prod(VOLUME_SHAPE))
    if volume.size != expected:
        raise ValueError(
            f"Unexpected raw size: {raw_path}, got {volume.size}, expected {expected}"
        )

    volume = volume.reshape(VOLUME_SHAPE)
    crop = volume[location["crop_start"]:location["crop_end_exclusive"], :, :]
    if location["left_pad"] > 0:
        pad = np.zeros(
            (location["left_pad"], VOLUME_SHAPE[1], VOLUME_SHAPE[2]),
            dtype=RAW_DTYPE,
        )
        crop = np.concatenate([pad, crop], axis=0)

    if crop.shape != (HEAD_LEN, VOLUME_SHAPE[1], VOLUME_SHAPE[2]):
        raise ValueError(f"Bad crop shape for {raw_path}: {crop.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.astype(RAW_DTYPE, copy=False).tofile(output_path)


def draw_check_image(picture_path, output_path, location):
    image = Image.open(picture_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    height = image.height
    start_x = location["crop_start"]
    end_x = location["head_end"]

    draw.rectangle((start_x, 0, end_x, height - 1), fill=(80, 180, 80, 45))
    draw.line((start_x, 0, start_x, height - 1), fill=(0, 255, 0, 255), width=2)
    draw.line((end_x, 0, end_x, height - 1), fill=(0, 120, 255, 255), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def iter_raw_files(root_dir):
    for raw_path in sorted(root_dir.rglob("*.raw")):
        yield raw_path


def make_metadata_row(drill_id, raw_path, out_path, location, source):
    return {
        "source": source,
        "drill_id": drill_id,
        "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
        "output_path": str(out_path.relative_to(PROJECT_ROOT)),
        "image_width": location["image_width"],
        "foreground_left": location["foreground_left"],
        "crop_start_green": location["crop_start"],
        "head_end_blue": location["head_end"],
        "crop_end_exclusive": location["crop_end_exclusive"],
        "available_len": location["available_len"],
        "left_pad": location["left_pad"],
        "output_shape": f"{HEAD_LEN},37,37",
    }


def crop_flat_data(locations, metadata_rows):
    count = 0
    for class_name in CLASS_NAMES:
        class_dir = RAW_DATA_DIR / class_name
        if not class_dir.exists():
            continue
        for raw_path in sorted(class_dir.glob("*.raw")):
            drill_id = drill_id_from_name(raw_path.name)
            if drill_id not in locations:
                raise KeyError(f"No picture location for {raw_path.name}")
            out_path = OUT_DATA_DIR / raw_path.relative_to(RAW_DATA_DIR)
            crop_raw(raw_path, out_path, locations[drill_id])
            metadata_rows.append(
                make_metadata_row(
                    drill_id, raw_path, out_path, locations[drill_id], "data"
                )
            )
            count += 1
    return count


def crop_5fold_data(locations, metadata_rows):
    count = 0
    if not RAW_5FOLD_DIR.exists():
        return count
    for raw_path in iter_raw_files(RAW_5FOLD_DIR):
        drill_id = drill_id_from_name(raw_path.name)
        if drill_id not in locations:
            raise KeyError(f"No picture location for {raw_path.name}")
        out_path = OUT_5FOLD_DIR / raw_path.relative_to(RAW_5FOLD_DIR)
        crop_raw(raw_path, out_path, locations[drill_id])
        metadata_rows.append(
            make_metadata_row(
                drill_id, raw_path, out_path, locations[drill_id], "data_5fold"
            )
        )
        count += 1
    return count


def draw_all_check_images(locations):
    count = 0
    for picture_path in sorted(PICTURE_DIR.glob("*.jpg")):
        drill_id = drill_id_from_name(picture_path.name)
        if drill_id not in locations:
            continue
        draw_check_image(
            picture_path, OUT_LINE_DIR / f"{picture_path.stem}.png", locations[drill_id]
        )
        count += 1
    return count


def write_metadata(rows):
    OUT_META_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "drill_id",
        "raw_path",
        "output_path",
        "image_width",
        "foreground_left",
        "crop_start_green",
        "head_end_blue",
        "crop_end_exclusive",
        "available_len",
        "left_pad",
        "output_shape",
    ]
    with OUT_META_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    for path in (OUT_DATA_DIR, OUT_5FOLD_DIR, OUT_LINE_DIR):
        path.mkdir(parents=True, exist_ok=True)

    locations = build_picture_locations()
    if not locations:
        raise RuntimeError(f"No pictures found in {PICTURE_DIR}")

    metadata_rows = []
    data_count = crop_flat_data(locations, metadata_rows)
    fold_count = crop_5fold_data(locations, metadata_rows)
    line_count = draw_all_check_images(locations)
    write_metadata(metadata_rows)

    left_pad_count = sum(1 for loc in locations.values() if loc["left_pad"] > 0)
    print(f"Located pictures: {len(locations)}")
    print(f"Cropped data raw files: {data_count} -> {OUT_DATA_DIR}")
    print(f"Cropped 5-fold raw files: {fold_count} -> {OUT_5FOLD_DIR}")
    print(f"Check images: {line_count} -> {OUT_LINE_DIR}")
    print(f"Metadata: {OUT_META_CSV}")
    print(f"Output raw shape: ({HEAD_LEN}, 37, 37)")
    print(f"Left-padded samples because head_end < {HEAD_LEN}: {left_pad_count}")


if __name__ == "__main__":
    main()
