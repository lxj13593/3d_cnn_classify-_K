from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DATASETS_ROOT = PROJECT_ROOT / "datasets"
DATASET_CONTAINER_RELATIVE_PATH = (
    Path("325_275_and_sphere_data") / "\u94bb\u5b54\u6570\u636e325_275"
)
SOURCE_SPLIT_NAME = "resnet18_head40_toml_multiboard_5fold"
TRAIN_DATASET_NAME = "multiboard_head40_toml"
TEST_DATASET_NAME = "test_3boards_head40_toml"
TRAIN_DATASET_ROOT_ENV = "DRILL_HEAD40_TOML_TRAIN_ROOT"
TEST_DATASET_ROOT_ENV = "DRILL_HEAD40_TOML_TEST_ROOT"
NORMAL_DIR_NAME = "normal_samples_by_board"
DEFECT_DIR_NAME = "defective_samples_by_board"
DEFAULT_SOURCE_SPLIT_ROOT = PROJECT_DATASETS_ROOT / SOURCE_SPLIT_NAME
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DATASETS_ROOT
    / "resnet18_head40_toml_direction_unified_multiboard_5fold"
)

N_SPLITS = 5
EXPECTED_TRAIN_COUNTS = {
    "sample_count": 4364,
    "normal_count": 3181,
    "defective_count": 1183,
    "board_count": 22,
}
EXPECTED_TEST_COUNTS = {
    "sample_count": 778,
    "normal_count": 556,
    "defective_count": 222,
    "board_count": 3,
}
DIRECTION_FIELDS = (
    "b_head_up",
    "original_head_side",
    "direction_flip_required",
    "direction_standardized",
    "standardized_head_side",
)


def is_dataset_root(path: Path, expected_name: str) -> bool:
    return path.name == expected_name and all(
        candidate.exists()
        for candidate in (
            path / "source_audit.csv",
            path / NORMAL_DIR_NAME,
            path / DEFECT_DIR_NAME,
        )
    )


def mounted_drive_roots() -> list[Path]:
    if os.name != "nt":
        return [Path("/")]
    bitmask = int(ctypes.windll.kernel32.GetLogicalDrives())
    return [
        Path(f"{chr(ord('A') + index)}:\\")
        for index in range(26)
        if bitmask & (1 << index)
    ]


def resolve_dataset_root(
    explicit_root: Path | None,
    expected_name: str,
    environment_name: str,
) -> Path:
    def validate(candidate: Path) -> Path:
        candidate = candidate.expanduser().resolve()
        if not is_dataset_root(candidate, expected_name):
            raise FileNotFoundError(f"Not the expected dataset '{expected_name}': {candidate}")
        return candidate

    if explicit_root is not None:
        return validate(explicit_root)
    environment_root = os.environ.get(environment_name, "").strip()
    if environment_root:
        return validate(Path(environment_root))
    project_candidate = PROJECT_DATASETS_ROOT / expected_name
    if is_dataset_root(project_candidate, expected_name):
        return project_candidate.resolve()
    candidates = []
    for drive_root in mounted_drive_roots():
        candidate = drive_root / DATASET_CONTAINER_RELATIVE_PATH / expected_name
        if is_dataset_root(candidate, expected_name):
            candidates.append(candidate.resolve())
    candidates = sorted(set(candidates), key=lambda path: str(path).lower())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {expected_name}. Pass its path explicitly or set "
            f"{environment_name}."
        )
    raise RuntimeError(
        "Multiple matching datasets were found:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently create Head40 direction-unified five-fold manifests "
            "while preserving the locked Head40 assignments."
        )
    )
    parser.add_argument("--source-split-root", type=Path, default=DEFAULT_SOURCE_SPLIT_ROOT)
    parser.add_argument("--train-dataset-root", type=Path, default=None)
    parser.add_argument("--test-dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def parse_bool(value: object, *, field: str, sample_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid {field} for {sample_id}: {value!r}")


def load_orientation_rows(
    dataset_root: Path,
    expected_count: int,
) -> dict[str, dict[str, str]]:
    fields, rows = read_csv(dataset_root / "source_audit.csv")
    required = {
        "SampleID",
        "Board",
        "Sequence",
        "Label",
        "LabelIndex",
        "RawShape",
        "RawSHA256",
        "FullVolumeRawShape",
        "FullVolumeRawSHA256",
        "BHeadUp",
        "HeadSide",
        "DirectionFlipped",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(
            f"Orientation source is missing fields at {dataset_root}: {sorted(missing)}"
        )
    if len(rows) != expected_count:
        raise ValueError(
            f"Orientation source count mismatch at {dataset_root}: "
            f"expected {expected_count}, got {len(rows)}"
        )

    output: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["SampleID"].strip()
        if sample_id in output:
            raise ValueError(f"Duplicate orientation SampleID: {sample_id}")
        b_head_up = parse_bool(row["BHeadUp"], field="BHeadUp", sample_id=sample_id)
        already_flipped = parse_bool(
            row["DirectionFlipped"],
            field="DirectionFlipped",
            sample_id=sample_id,
        )
        if already_flipped:
            raise ValueError(
                f"Orientation source is already direction-flipped: {sample_id}"
            )
        expected_side = "left" if b_head_up else "right"
        if row["HeadSide"].strip().lower() != expected_side:
            raise ValueError(
                f"BHeadUp/HeadSide mismatch for {sample_id}: "
                f"{row['BHeadUp']}/{row['HeadSide']}"
            )
        output[sample_id] = row
    return output


def merge_orientation(
    rows: list[dict[str, str]],
    orientation: dict[str, dict[str, str]],
    volume_kind: str,
) -> list[dict[str, object]]:
    if volume_kind not in {"full_volume", "head40"}:
        raise ValueError(f"Unsupported volume kind: {volume_kind}")
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"].strip()
        if sample_id in seen:
            raise ValueError(f"Duplicate split SampleID: {sample_id}")
        seen.add(sample_id)
        try:
            source = orientation[sample_id]
        except KeyError as exc:
            raise ValueError(f"Missing orientation metadata for {sample_id}") from exc

        identity_pairs = (
            ("board", "Board"),
            ("sequence", "Sequence"),
            ("label", "Label"),
            ("label_index", "LabelIndex"),
        )
        for split_field, source_field in identity_pairs:
            left = row[split_field].strip().lower()
            right = source[source_field].strip().lower()
            if split_field == "sequence":
                left = left.zfill(4)
                right = right.zfill(4)
            if left != right:
                raise ValueError(
                    f"Identity mismatch for {sample_id}: "
                    f"{split_field}={row[split_field]!r}, "
                    f"{source_field}={source[source_field]!r}"
                )

        orientation_shape_field = (
            "FullVolumeRawShape" if volume_kind == "full_volume" else "RawShape"
        )
        if row["raw_shape_whd"].strip().lower() != source[
            orientation_shape_field
        ].strip().lower():
            raise ValueError(
                f"Shape mismatch for {sample_id}: split={row['raw_shape_whd']}, "
                f"orientation={source[orientation_shape_field]}"
            )
        orientation_hash_field = (
            "FullVolumeRawSHA256" if volume_kind == "full_volume" else "RawSHA256"
        )
        if row["raw_sha256"].strip().lower() != source[
            orientation_hash_field
        ].strip().lower():
            raise ValueError(
                f"RAW hash mismatch for {sample_id}: "
                f"split={row['raw_sha256']}, "
                f"orientation={source[orientation_hash_field]}"
            )

        b_head_up = parse_bool(
            source["BHeadUp"], field="BHeadUp", sample_id=sample_id
        )
        merged.append(
            {
                **row,
                "b_head_up": str(b_head_up).lower(),
                "original_head_side": (
                    "low_depth_index" if b_head_up else "high_depth_index"
                ),
                "direction_flip_required": str(b_head_up).lower(),
                "direction_standardized": "true",
                "standardized_head_side": "high_depth_index",
            }
        )
    return merged


def counts(rows: list[dict[str, object]]) -> dict[str, int]:
    labels = Counter(str(row["label"]) for row in rows)
    return {
        "sample_count": len(rows),
        "normal_count": labels["normal"],
        "defective_count": labels["defective"],
        "board_count": len({str(row["board"]) for row in rows}),
    }


def require_counts(
    name: str,
    rows: list[dict[str, object]],
    expected: dict[str, int],
) -> None:
    actual = counts(rows)
    if actual != expected:
        raise ValueError(f"{name} counts changed: expected {expected}, got {actual}")


def audit_folds(rows: list[dict[str, object]]) -> None:
    all_ids = {str(row["sample_id"]) for row in rows}
    boards = {str(row["board"]) for row in rows}
    fold_values = {int(row["validation_fold"]) for row in rows}
    if fold_values != set(range(N_SPLITS)):
        raise ValueError(f"Invalid validation folds: {sorted(fold_values)}")
    for fold in range(N_SPLITS):
        train = [row for row in rows if int(row["validation_fold"]) != fold]
        val = [row for row in rows if int(row["validation_fold"]) == fold]
        train_ids = {str(row["sample_id"]) for row in train}
        val_ids = {str(row["sample_id"]) for row in val}
        if train_ids & val_ids or train_ids | val_ids != all_ids:
            raise ValueError(f"Fold {fold}: leakage or coverage error")
        if {str(row["board"]) for row in train} != boards:
            raise ValueError(f"Fold {fold}: training does not contain all boards")
        if {str(row["board"]) for row in val} != boards:
            raise ValueError(f"Fold {fold}: validation does not contain all boards")


def summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for fold in range(N_SPLITS):
        for split in ("train", "val"):
            selected = [
                row
                for row in rows
                if (int(row["validation_fold"]) != fold) == (split == "train")
            ]
            for board in ["ALL", *sorted({str(row["board"]) for row in selected})]:
                group = (
                    selected
                    if board == "ALL"
                    else [row for row in selected if row["board"] == board]
                )
                label_counts = Counter(str(row["label"]) for row in group)
                direction_counts = Counter(str(row["b_head_up"]) for row in group)
                output.append(
                    {
                        "fold": fold,
                        "split": split,
                        "board": board,
                        "normal": label_counts["normal"],
                        "defective": label_counts["defective"],
                        "b_head_up_false": direction_counts["false"],
                        "b_head_up_true": direction_counts["true"],
                        "total": len(group),
                    }
                )
    return output


def direction_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    values = Counter(str(row["b_head_up"]) for row in rows)
    return {
        "kept_original_direction": values["false"],
        "flipped_along_depth": values["true"],
    }


def build_direction_unified_splits(
    *,
    source_split_root: Path,
    orientation_train_root: Path,
    orientation_test_root: Path,
    output_root: Path,
    volume_kind: str,
    experiment_label: str,
    assignment_name: str,
) -> None:
    source_split_root = source_split_root.resolve()
    orientation_train_root = orientation_train_root.resolve()
    orientation_test_root = orientation_test_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_split_root:
        raise ValueError("Output split root must differ from the source split root")

    source_config = read_json(source_split_root / "split_config.json")
    if int(source_config.get("n_splits", -1)) != N_SPLITS:
        raise ValueError("Source split is not five-fold")
    if source_config.get("internal_test_set") is not False:
        raise ValueError("Source split must not contain an internal test set")

    train_fields, source_train = read_csv(
        source_split_root / "all_samples_with_validation_fold.csv"
    )
    test_fields, source_test = read_csv(source_split_root / "locked_test_manifest.csv")
    if train_fields != test_fields:
        raise ValueError("Source train and locked-test manifest fields differ")
    output_fields = [*train_fields, *DIRECTION_FIELDS]

    train_orientation = load_orientation_rows(
        orientation_train_root, EXPECTED_TRAIN_COUNTS["sample_count"]
    )
    test_orientation = load_orientation_rows(
        orientation_test_root, EXPECTED_TEST_COUNTS["sample_count"]
    )
    train_rows = merge_orientation(source_train, train_orientation, volume_kind)
    test_rows = merge_orientation(source_test, test_orientation, volume_kind)
    require_counts("Training data", train_rows, EXPECTED_TRAIN_COUNTS)
    require_counts("Locked test data", test_rows, EXPECTED_TEST_COUNTS)
    audit_folds(train_rows)
    if {str(row["sample_id"]) for row in train_rows} & {
        str(row["sample_id"]) for row in test_rows
    }:
        raise ValueError("Train/locked-test SampleID leakage detected")

    train_rows.sort(key=lambda row: str(row["sample_id"]))
    test_rows.sort(key=lambda row: str(row["sample_id"]))
    write_csv(
        output_root / "all_samples_with_validation_fold.csv",
        train_rows,
        output_fields,
    )
    write_csv(output_root / "locked_test_manifest.csv", test_rows, output_fields)
    for fold in range(N_SPLITS):
        fold_dir = output_root / f"fold_{fold}"
        write_csv(
            fold_dir / "train.csv",
            [row for row in train_rows if int(row["validation_fold"]) != fold],
            output_fields,
        )
        write_csv(
            fold_dir / "val.csv",
            [row for row in train_rows if int(row["validation_fold"]) == fold],
            output_fields,
        )
    write_csv(
        output_root / "fold_board_summary.csv",
        summary_rows(train_rows),
        [
            "fold",
            "split",
            "board",
            "normal",
            "defective",
            "b_head_up_false",
            "b_head_up_true",
            "total",
        ],
    )

    config = {
        **source_config,
        "experiment_label": experiment_label,
        "assignment": assignment_name,
        "source_assignment": source_config.get("assignment"),
        "source_split_root": str(source_split_root),
        "orientation_train_dataset_root_at_creation": str(orientation_train_root),
        "orientation_test_dataset_root_at_creation": str(orientation_test_root),
        "direction_standardized": True,
        "direction_standardization": {
            "axis": "D",
            "target_head_side": "high_depth_index",
            "bHeadUp_false": "keep",
            "bHeadUp_true": "flip_D",
            "application_stage": "data_loading_before_augmentation_and_normalization",
            "physical_raw_files_modified": False,
        },
        "direction_counts_train": direction_counts(train_rows),
        "direction_counts_locked_test": direction_counts(test_rows),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "split_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print(f"{experiment_label} manifests created")
    print(f"Source split: {source_split_root}")
    print(f"Output split: {output_root}")
    print(f"Training direction counts: {direction_counts(train_rows)}")
    print(f"Locked-test direction counts: {direction_counts(test_rows)}")
    print("Target direction: drill head at high D index")


def main() -> None:
    args = parse_args()
    train_root = resolve_dataset_root(
        args.train_dataset_root,
        TRAIN_DATASET_NAME,
        TRAIN_DATASET_ROOT_ENV,
    )
    test_root = resolve_dataset_root(
        args.test_dataset_root,
        TEST_DATASET_NAME,
        TEST_DATASET_ROOT_ENV,
    )
    build_direction_unified_splits(
        source_split_root=args.source_split_root,
        orientation_train_root=train_root,
        orientation_test_root=test_root,
        output_root=args.output_root,
        volume_kind="head40",
        experiment_label="TOML Head40 direction-unified ResNet18",
        assignment_name=(
            "inherited_from_head40_baseline_with_runtime_direction_standardization"
        ),
    )


if __name__ == "__main__":
    main()
