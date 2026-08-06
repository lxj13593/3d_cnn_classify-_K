from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from data_load_resnet18_multiboard_final import (
    CLASS_TO_INDEX,
    DEFECT_DIR_NAME,
    NORMAL_DIR_NAME,
    SOURCE_MANIFEST_NAME,
    TEST_DATASET_NAME,
    TRAIN_DATASET_NAME,
    parse_shape,
    resolve_dataset_root,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "resnet18_multiboard_final_5fold"
N_SPLITS = 5
SPLIT_SEED = 42
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
    "validation_fold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create within-board, label-and-shape-balanced five-fold manifests "
            "for the final 22-board dataset and audit the locked three-board test set."
        )
    )
    parser.add_argument("--train-dataset-root", type=Path, default=None)
    parser.add_argument("--test-dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--skip-file-check", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields=OUTPUT_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def load_dataset_samples(
    dataset_root: Path,
    verify_files: bool,
) -> list[dict[str, object]]:
    source_rows = read_csv(dataset_root / SOURCE_MANIFEST_NAME)
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
    }
    if source_rows:
        missing = required - set(source_rows[0])
        if missing:
            raise ValueError(f"source_audit.csv is missing columns: {sorted(missing)}")

    samples: list[dict[str, object]] = []
    for row_number, row in enumerate(source_rows, start=2):
        label = row["Label"].strip().lower()
        label_index = int(row["LabelIndex"])
        if CLASS_TO_INDEX.get(label) != label_index:
            raise ValueError(f"Label/index mismatch in row {row_number}")
        board = row["Board"].strip()
        sequence = row["Sequence"].strip().zfill(4)
        raw_file_name = row["RawFileName"].strip()
        shape_whd = parse_shape(row["RawShape"].strip())
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
                "sample_id": row["SampleID"].strip(),
                "board": board,
                "sequence": sequence,
                "label": label,
                "label_index": label_index,
                "difficulty": row["Difficulty"].strip(),
                "raw_shape_whd": "x".join(str(value) for value in shape_whd),
                "input_shape_dhw": "x".join(str(value) for value in reversed(shape_whd)),
                "raw_file_name": raw_file_name,
                "raw_relative_path": relative_path.as_posix(),
                "raw_sha256": row["RawSHA256"].strip().lower(),
                "validation_fold": -1,
            }
        )
    audit_unique(samples)
    return samples


def audit_unique(samples: list[dict[str, object]]) -> None:
    for field in ("sample_id", "raw_sha256"):
        values = [str(row[field]) for row in samples]
        duplicates = [value for value, count in Counter(values).items() if count > 1]
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
            str(sample["raw_shape_whd"]),
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
    if any(int(audit[key]) for key in audit if key.endswith("overlap_count")):
        raise ValueError(f"Locked-test leakage detected: {audit}")
    return audit


def summary_rows(samples: list[dict[str, object]], fold: int, split: str):
    selected = [
        row
        for row in samples
        if (int(row["validation_fold"]) != fold) == (split == "train")
    ]
    label_counts = Counter(str(row["label"]) for row in selected)
    yield {
        "fold": fold,
        "split": split,
        "board": "ALL",
        "normal": label_counts["normal"],
        "defective": label_counts["defective"],
        "total": len(selected),
    }
    for board in sorted({str(row["board"]) for row in selected}):
        rows = [row for row in selected if row["board"] == board]
        board_counts = Counter(str(row["label"]) for row in rows)
        yield {
            "fold": fold,
            "split": split,
            "board": board,
            "normal": board_counts["normal"],
            "defective": board_counts["defective"],
            "total": len(rows),
        }


def main() -> None:
    args = parse_args()
    train_root = resolve_dataset_root(
        args.train_dataset_root,
        expected_name=TRAIN_DATASET_NAME,
    )
    test_root = resolve_dataset_root(
        args.test_dataset_root,
        expected_name=TEST_DATASET_NAME,
    )
    output_root = args.output_root.resolve()

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
    summary = []
    for fold in range(N_SPLITS):
        fold_dir = output_root / f"fold_{fold}"
        train_rows = [row for row in train_samples if int(row["validation_fold"]) != fold]
        val_rows = [row for row in train_samples if int(row["validation_fold"]) == fold]
        write_csv(fold_dir / "train.csv", train_rows)
        write_csv(fold_dir / "val.csv", val_rows)
        summary.extend(summary_rows(train_samples, fold, "train"))
        summary.extend(summary_rows(train_samples, fold, "val"))
    write_csv(
        output_root / "fold_board_summary.csv",
        summary,
        ("fold", "split", "board", "normal", "defective", "total"),
    )

    config = {
        "n_splits": N_SPLITS,
        "split_seed": args.seed,
        "assignment": "within_board_label_shape_balanced_round_robin",
        "train_fraction": 0.8,
        "validation_fraction": 0.2,
        "internal_test_set": False,
        "locked_external_test_set": True,
        "train_dataset_name": TRAIN_DATASET_NAME,
        "test_dataset_name": TEST_DATASET_NAME,
        "train_dataset_root_at_creation": str(train_root),
        "test_dataset_root_at_creation": str(test_root),
        **EXPECTED_TRAIN_COUNTS,
        "locked_test_counts": EXPECTED_TEST_COUNTS,
        "leakage_audit": leakage,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "split_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("Final multi-board five-fold manifests created")
    print(f"Training dataset: {train_root}")
    print(f"Locked test set: {test_root}")
    print(f"Output: {output_root}")
    print(f"Counts: {counts(train_samples)}")
    print(f"Locked test: {counts(test_samples)}")
    print("Leakage audit: board=0, sample_id=0, RAW SHA256=0")


if __name__ == "__main__":
    main()
