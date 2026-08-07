from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DATASETS_ROOT = PROJECT_ROOT / "datasets"
DATASET_CONTAINER_RELATIVE_PATH = (
    Path("325_275_and_sphere_data") / "\u94bb\u5b54\u6570\u636e325_275"
)
TRAIN_DATASET_NAME = "multiboard_head40_toml"
TEST_DATASET_NAME = "test_3boards_head40_toml"
TRAIN_DATASET_ROOT_ENV = "DRILL_HEAD40_TOML_TRAIN_ROOT"
TEST_DATASET_ROOT_ENV = "DRILL_HEAD40_TOML_TEST_ROOT"
SOURCE_MANIFEST_NAME = "source_audit.csv"
NORMAL_DIR_NAME = "normal_samples_by_board"
DEFECT_DIR_NAME = "defective_samples_by_board"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DATASETS_ROOT
    / "resnet18_head40_toml_direction_unified_multiboard_5fold"
)

CLASS_TO_INDEX = {"normal": 0, "defective": 1}
N_SPLITS = 5
SPLIT_SEED = 42
SPLIT_ASSIGNMENT = (
    "within_board_label_full_volume_shape_balanced_direction_unified"
)
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
OUTPUT_FIELDS = (
    "sample_id",
    "board",
    "sequence",
    "label",
    "label_index",
    "difficulty",
    "raw_shape_whd",
    "input_shape_dhw",
    "raw_file_name",
    "raw_relative_path",
    "raw_sha256",
    "source_full_volume_shape_whd",
    "validation_fold",
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
            path / SOURCE_MANIFEST_NAME,
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
            raise FileNotFoundError(
                f"Not the expected dataset '{expected_name}': {candidate}"
            )
        return candidate

    if explicit_root is not None:
        return validate(explicit_root)
    environment_root = os.environ.get(environment_name, "").strip()
    if environment_root:
        return validate(Path(environment_root))
    project_candidate = PROJECT_DATASETS_ROOT / expected_name
    if is_dataset_root(project_candidate, expected_name):
        return project_candidate.resolve()

    candidates: list[Path] = []
    for drive_root in mounted_drive_roots():
        candidate = drive_root / DATASET_CONTAINER_RELATIVE_PATH / expected_name
        if is_dataset_root(candidate, expected_name):
            candidates.append(candidate.resolve())
    candidates = sorted(set(candidates), key=lambda path: str(path).lower())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {expected_name}. Pass its path with the matching "
            f"command-line option or set {environment_name}."
        )
    raise RuntimeError(
        "Multiple matching datasets were found:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the standalone TOML Head40 direction-unified five-fold "
            "manifests directly from the training and locked-test source_audit.csv "
            "files. No existing split manifest is required."
        )
    )
    parser.add_argument("--train-dataset-root", type=Path, default=None)
    parser.add_argument("--test-dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--skip-file-check", action="store_true")
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
    fields: tuple[str, ...] = OUTPUT_FIELDS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.strip().lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"Invalid RAW shape: {value!r}") from exc
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"Invalid RAW shape: {value!r}")
    return shape


def parse_bool(value: object, *, field: str, sample_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid {field} for {sample_id}: {value!r}")


def load_dataset_samples(
    dataset_root: Path,
    verify_files: bool,
) -> list[dict[str, object]]:
    fields, source_rows = read_csv(dataset_root / SOURCE_MANIFEST_NAME)
    required = {
        "SampleID",
        "Board",
        "Sequence",
        "Label",
        "LabelIndex",
        "Difficulty",
        "RawShape",
        "RawFileName",
        "RawSHA256",
        "FullVolumeRawShape",
        "BHeadUp",
        "HeadSide",
        "DirectionFlipped",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(
            f"{dataset_root / SOURCE_MANIFEST_NAME} is missing columns: "
            f"{sorted(missing)}"
        )

    samples: list[dict[str, object]] = []
    for row_number, row in enumerate(source_rows, start=2):
        sample_id = row["SampleID"].strip()
        label = row["Label"].strip().lower()
        label_index = int(row["LabelIndex"])
        if CLASS_TO_INDEX.get(label) != label_index:
            raise ValueError(
                f"Label/index mismatch at {dataset_root / SOURCE_MANIFEST_NAME}:"
                f"{row_number}"
            )

        board = row["Board"].strip()
        sequence = row["Sequence"].strip().zfill(4)
        raw_file_name = row["RawFileName"].strip()
        shape_whd = parse_shape(row["RawShape"])
        full_volume_shape_whd = parse_shape(row["FullVolumeRawShape"])
        if shape_whd[2] != 40 or shape_whd[:2] not in {
            (29, 29),
            (37, 37),
            (49, 49),
        }:
            raise ValueError(
                f"Unexpected TOML Head40 shape at row {row_number}: {shape_whd}"
            )
        if shape_whd[:2] != full_volume_shape_whd[:2]:
            raise ValueError(
                f"Head40/full-volume cross-section mismatch for {sample_id}: "
                f"{shape_whd} vs {full_volume_shape_whd}"
            )

        b_head_up = parse_bool(
            row["BHeadUp"], field="BHeadUp", sample_id=sample_id
        )
        already_flipped = parse_bool(
            row["DirectionFlipped"],
            field="DirectionFlipped",
            sample_id=sample_id,
        )
        if already_flipped:
            raise ValueError(
                f"Head40 source is already direction-flipped: {sample_id}"
            )
        expected_head_side = "left" if b_head_up else "right"
        if row["HeadSide"].strip().lower() != expected_head_side:
            raise ValueError(
                f"BHeadUp/HeadSide mismatch for {sample_id}: "
                f"{row['BHeadUp']}/{row['HeadSide']}"
            )

        class_dir = NORMAL_DIR_NAME if label == "normal" else DEFECT_DIR_NAME
        relative_path = Path(class_dir) / board / "drill" / raw_file_name
        raw_path = (dataset_root / relative_path).resolve()
        if verify_files:
            if not raw_path.is_file():
                raise FileNotFoundError(f"RAW file not found: {raw_path}")
            expected_bytes = math.prod(shape_whd)
            if raw_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"RAW size mismatch: {raw_path}; "
                    f"expected {expected_bytes}, got {raw_path.stat().st_size}"
                )

        samples.append(
            {
                "sample_id": sample_id,
                "board": board,
                "sequence": sequence,
                "label": label,
                "label_index": label_index,
                "difficulty": row["Difficulty"].strip(),
                "raw_shape_whd": "x".join(str(value) for value in shape_whd),
                "input_shape_dhw": "x".join(
                    str(value) for value in reversed(shape_whd)
                ),
                "raw_file_name": raw_file_name,
                "raw_relative_path": relative_path.as_posix(),
                "raw_sha256": row["RawSHA256"].strip().lower(),
                "source_full_volume_shape_whd": "x".join(
                    str(value) for value in full_volume_shape_whd
                ),
                "validation_fold": -1,
                "b_head_up": str(b_head_up).lower(),
                "original_head_side": (
                    "low_depth_index" if b_head_up else "high_depth_index"
                ),
                "direction_flip_required": str(b_head_up).lower(),
                "direction_standardized": "true",
                "standardized_head_side": "high_depth_index",
            }
        )
    audit_unique(samples)
    return samples


def audit_unique(samples: list[dict[str, object]]) -> None:
    for field in ("sample_id", "raw_sha256"):
        values = [str(row[field]) for row in samples]
        duplicates = [
            value for value, count in Counter(values).items() if count > 1
        ]
        if duplicates:
            raise ValueError(f"Duplicate {field}: {duplicates[:5]}")


def counts(samples: list[dict[str, object]]) -> dict[str, int]:
    labels = Counter(str(row["label"]) for row in samples)
    return {
        "sample_count": len(samples),
        "normal_count": labels["normal"],
        "defective_count": labels["defective"],
        "board_count": len({str(row["board"]) for row in samples}),
    }


def require_counts(
    name: str,
    samples: list[dict[str, object]],
    expected: dict[str, int],
) -> None:
    actual = counts(samples)
    if actual != expected:
        raise ValueError(f"{name} counts changed: expected {expected}, got {actual}")


def assign_validation_folds(
    samples: list[dict[str, object]],
    n_splits: int,
    seed: int,
) -> None:
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for sample in samples:
        key = (
            str(sample["board"]),
            int(sample["label_index"]),
            str(sample["source_full_volume_shape_whd"]),
        )
        groups[key].append(sample)

    board_label_counts = defaultdict(lambda: [0] * n_splits)
    board_total_counts = defaultdict(lambda: [0] * n_splits)
    shape_label_counts = defaultdict(lambda: [0] * n_splits)
    label_counts = defaultdict(lambda: [0] * n_splits)
    total_counts = [0] * n_splits

    for group_index, (key, group) in enumerate(sorted(groups.items())):
        board, label_index, shape = key
        rng = random.Random(seed + group_index * 1009)
        group.sort(key=lambda row: str(row["sample_id"]))
        rng.shuffle(group)
        base, remainder = divmod(len(group), n_splits)
        quotas = [base] * n_splits
        for fold in range(n_splits):
            board_label_counts[(board, label_index)][fold] += base
            board_total_counts[board][fold] += base
            shape_label_counts[(shape, label_index)][fold] += base
            label_counts[label_index][fold] += base
            total_counts[fold] += base

        chosen: set[int] = set()
        tie_order = list(range(n_splits))
        rng.shuffle(tie_order)
        tie_rank = {fold: rank for rank, fold in enumerate(tie_order)}
        for _ in range(remainder):
            candidates = [fold for fold in range(n_splits) if fold not in chosen]
            fold = min(
                candidates,
                key=lambda item: (
                    board_label_counts[(board, label_index)][item],
                    board_total_counts[board][item],
                    shape_label_counts[(shape, label_index)][item],
                    label_counts[label_index][item],
                    total_counts[item],
                    tie_rank[item],
                ),
            )
            chosen.add(fold)
            quotas[fold] += 1
            board_label_counts[(board, label_index)][fold] += 1
            board_total_counts[board][fold] += 1
            shape_label_counts[(shape, label_index)][fold] += 1
            label_counts[label_index][fold] += 1
            total_counts[fold] += 1

        cursor = 0
        for fold, quota in enumerate(quotas):
            for sample in group[cursor : cursor + quota]:
                sample["validation_fold"] = fold
            cursor += quota
        if cursor != len(group):
            raise RuntimeError("Internal fold assignment error")


def audit_folds(samples: list[dict[str, object]]) -> None:
    all_ids = {str(row["sample_id"]) for row in samples}
    boards = {str(row["board"]) for row in samples}
    folds = {int(row["validation_fold"]) for row in samples}
    if folds != set(range(N_SPLITS)):
        raise ValueError(f"Invalid validation fold set: {sorted(folds)}")
    for fold in range(N_SPLITS):
        train = [row for row in samples if int(row["validation_fold"]) != fold]
        val = [row for row in samples if int(row["validation_fold"]) == fold]
        train_ids = {str(row["sample_id"]) for row in train}
        val_ids = {str(row["sample_id"]) for row in val}
        if train_ids & val_ids or train_ids | val_ids != all_ids:
            raise ValueError(f"Fold {fold}: train/validation leakage or coverage error")
        if {str(row["board"]) for row in train} != boards:
            raise ValueError(f"Fold {fold}: training does not contain all boards")
        if {str(row["board"]) for row in val} != boards:
            raise ValueError(f"Fold {fold}: validation does not contain all boards")


def leakage_audit(
    train_samples: list[dict[str, object]],
    test_samples: list[dict[str, object]],
) -> dict[str, object]:
    train_boards = {str(row["board"]) for row in train_samples}
    test_boards = {str(row["board"]) for row in test_samples}
    train_ids = {str(row["sample_id"]) for row in train_samples}
    test_ids = {str(row["sample_id"]) for row in test_samples}
    train_hashes = {str(row["raw_sha256"]) for row in train_samples}
    test_hashes = {str(row["raw_sha256"]) for row in test_samples}
    audit = {
        "board_overlap_count": len(train_boards & test_boards),
        "sample_id_overlap_count": len(train_ids & test_ids),
        "raw_sha256_overlap_count": len(train_hashes & test_hashes),
        "train_boards": sorted(train_boards),
        "test_boards": sorted(test_boards),
    }
    if any(
        int(value)
        for key, value in audit.items()
        if key.endswith("_overlap_count")
    ):
        raise ValueError(f"Locked-test leakage detected: {audit}")
    return audit


def summary_rows(
    samples: list[dict[str, object]],
    fold: int,
    split: str,
):
    selected = [
        row
        for row in samples
        if (int(row["validation_fold"]) != fold) == (split == "train")
    ]
    board_names = ["ALL", *sorted({str(row["board"]) for row in selected})]
    for board in board_names:
        rows = (
            selected
            if board == "ALL"
            else [row for row in selected if str(row["board"]) == board]
        )
        label_counts = Counter(str(row["label"]) for row in rows)
        head_counts = Counter(str(row["b_head_up"]) for row in rows)
        yield {
            "fold": fold,
            "split": split,
            "board": board,
            "normal": label_counts["normal"],
            "defective": label_counts["defective"],
            "b_head_up_false": head_counts["false"],
            "b_head_up_true": head_counts["true"],
            "total": len(rows),
        }


def direction_counts(samples: list[dict[str, object]]) -> dict[str, int]:
    values = Counter(str(row["b_head_up"]) for row in samples)
    return {
        "kept_original_direction": values["false"],
        "flipped_along_depth": values["true"],
    }


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
    output_root = args.output_root.expanduser().resolve()
    if output_root in {train_root, test_root}:
        raise ValueError("Output root must differ from both dataset roots")

    train_samples = load_dataset_samples(train_root, not args.skip_file_check)
    test_samples = load_dataset_samples(test_root, not args.skip_file_check)
    require_counts("Training dataset", train_samples, EXPECTED_TRAIN_COUNTS)
    require_counts("Locked test dataset", test_samples, EXPECTED_TEST_COUNTS)
    leakage = leakage_audit(train_samples, test_samples)

    assign_validation_folds(train_samples, N_SPLITS, args.seed)
    audit_folds(train_samples)
    train_samples.sort(key=lambda row: str(row["sample_id"]))
    test_samples.sort(key=lambda row: str(row["sample_id"]))

    write_csv(output_root / "all_samples_with_validation_fold.csv", train_samples)
    write_csv(output_root / "locked_test_manifest.csv", test_samples)
    summary: list[dict[str, object]] = []
    for fold in range(N_SPLITS):
        fold_dir = output_root / f"fold_{fold}"
        train_rows = [
            row for row in train_samples if int(row["validation_fold"]) != fold
        ]
        val_rows = [
            row for row in train_samples if int(row["validation_fold"]) == fold
        ]
        write_csv(fold_dir / "train.csv", train_rows)
        write_csv(fold_dir / "val.csv", val_rows)
        summary.extend(summary_rows(train_samples, fold, "train"))
        summary.extend(summary_rows(train_samples, fold, "val"))
    write_csv(
        output_root / "fold_board_summary.csv",
        summary,
        (
            "fold",
            "split",
            "board",
            "normal",
            "defective",
            "b_head_up_false",
            "b_head_up_true",
            "total",
        ),
    )

    train_direction_counts = direction_counts(train_samples)
    test_direction_counts = direction_counts(test_samples)
    config = {
        "n_splits": N_SPLITS,
        "split_seed": args.seed,
        "assignment": SPLIT_ASSIGNMENT,
        "assignment_method": (
            "deterministic balanced round-robin generated directly from "
            "source_audit.csv; groups are Board + LabelIndex + "
            "FullVolumeRawShape"
        ),
        "assignment_group_fields": [
            "Board",
            "LabelIndex",
            "FullVolumeRawShape",
        ],
        "train_fraction": 0.8,
        "validation_fraction": 0.2,
        "internal_test_set": False,
        "locked_external_test_set": True,
        "train_dataset_name": TRAIN_DATASET_NAME,
        "test_dataset_name": TEST_DATASET_NAME,
        "train_dataset_root_at_creation": str(train_root),
        "test_dataset_root_at_creation": str(test_root),
        "source_manifest_name": SOURCE_MANIFEST_NAME,
        "manifests_built_directly_from_source_audit": True,
        **EXPECTED_TRAIN_COUNTS,
        "locked_test_counts": EXPECTED_TEST_COUNTS,
        "leakage_audit": leakage,
        "experiment_label": "TOML Head40 direction-unified ResNet18",
        "direction_standardized": True,
        "direction_standardization": {
            "axis": "D",
            "target_head_side": "high_depth_index",
            "bHeadUp_false": "keep",
            "bHeadUp_true": "flip_D",
            "application_stage": (
                "data_loading_before_augmentation_and_normalization"
            ),
            "physical_raw_files_modified": False,
        },
        "direction_counts_train": train_direction_counts,
        "direction_counts_locked_test": test_direction_counts,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "split_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("Standalone TOML Head40 direction-unified manifests created")
    print(f"Training dataset: {train_root}")
    print(f"Locked test set: {test_root}")
    print(f"Output: {output_root}")
    print(f"Counts: {counts(train_samples)}")
    print(f"Locked test: {counts(test_samples)}")
    print(f"Training direction counts: {train_direction_counts}")
    print(f"Locked-test direction counts: {test_direction_counts}")
    print("Leakage audit: board=0, sample_id=0, RAW SHA256=0")
    print("Target direction: drill head at high D index")


if __name__ == "__main__":
    main()
