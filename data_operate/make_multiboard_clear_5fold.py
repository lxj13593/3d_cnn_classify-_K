from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_RELATIVE_PATH = (
    Path("325_275_and_sphere_data")
    / "钻孔数据325_275"
    / "clear_binary_dataset"
)
DATASET_ROOT_ENV = "DRILL_DATASET_ROOT"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "multiboard_clear_5fold"

NORMAL_DIR_NAME = "normal_samples_by_board"
DEFECT_DIR_NAME = "obvious_defects_by_board"
MANIFEST_NAME = "all_boards_manifest.csv"
N_SPLITS = 5
SPLIT_SEED = 42
EXPECTED_NORMAL = 3262
EXPECTED_DEFECTIVE = 1141
RAW_DTYPE_BYTES = 1  # uint8

SHAPE_PATTERN = re.compile(r"_(\d+)_(\d+)_(\d+)-")
OUTPUT_FIELDS = (
    "sample_id",
    "board",
    "sequence",
    "label",
    "label_index",
    "raw_shape_whd",
    "input_shape_dhw",
    "raw_relative_path",
    "raw_file_name",
    "raw_sha256",
    "validation_fold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build read-only, within-board stratified 5-fold manifests for the "
            "clear multi-board binary dataset. No RAW file is copied or modified."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Path to clear_binary_dataset. If omitted, use DRILL_DATASET_ROOT "
            "or search mounted drives for the known relative data path."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--expected-normal", type=int, default=EXPECTED_NORMAL)
    parser.add_argument(
        "--expected-defective", type=int, default=EXPECTED_DEFECTIVE
    )
    return parser.parse_args()


def is_dataset_root(path: Path) -> bool:
    return all(
        (
            path / class_dir / MANIFEST_NAME
        ).is_file()
        for class_dir in (NORMAL_DIR_NAME, DEFECT_DIR_NAME)
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


def resolve_dataset_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        candidate = explicit_root.expanduser().resolve()
        if not is_dataset_root(candidate):
            raise FileNotFoundError(
                f"Not a valid clear_binary_dataset directory: {candidate}"
            )
        return candidate

    environment_root = os.environ.get(DATASET_ROOT_ENV, "").strip()
    if environment_root:
        candidate = Path(environment_root).expanduser().resolve()
        if not is_dataset_root(candidate):
            raise FileNotFoundError(
                f"{DATASET_ROOT_ENV} is not a valid dataset root: {candidate}"
            )
        return candidate

    candidates = []
    for drive_root in mounted_drive_roots():
        candidate = drive_root / DATASET_RELATIVE_PATH
        if is_dataset_root(candidate):
            candidates.append(candidate.resolve())
    candidates = sorted(set(candidates), key=lambda path: str(path).lower())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Could not find clear_binary_dataset on any mounted drive. "
            "Pass --dataset-root or set DRILL_DATASET_ROOT."
        )
    joined = "\n  ".join(str(path) for path in candidates)
    raise RuntimeError(
        "Multiple matching datasets were found. Select one with --dataset-root:\n  "
        + joined
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str] | tuple[str, ...], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_shape_from_name(file_name: str) -> tuple[int, int, int]:
    match = SHAPE_PATTERN.search(file_name)
    if match is None:
        raise ValueError(f"Cannot parse W_H_D shape from RAW file name: {file_name}")
    return tuple(int(value) for value in match.groups())


def parse_manifest_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"Invalid RawShape value: {value}") from exc
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"Invalid RawShape value: {value}")
    return shape


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def make_sample_id(label: str, board: str, sequence: str, raw_hash: str) -> str:
    return f"{label}|{board}|{sequence}|{raw_hash[:16].lower()}"


def load_class_samples(
    class_root: Path,
    label: str,
    label_index: int,
    dataset_root: Path,
) -> list[dict[str, object]]:
    manifest_path = class_root / MANIFEST_NAME
    rows = read_csv(manifest_path)
    samples: list[dict[str, object]] = []

    for row_number, row in enumerate(rows, start=2):
        board = row.get("Board", "").strip()
        sequence = row.get("Sequence", "").strip()
        file_name = row.get("RawFileName", "").strip()
        raw_hash = row.get("RawSHA256", "").strip().lower()
        if not board or not sequence or not file_name:
            raise ValueError(
                f"Missing Board/Sequence/RawFileName in {manifest_path}:{row_number}"
            )
        if len(raw_hash) != 64 or any(ch not in "0123456789abcdef" for ch in raw_hash):
            raise ValueError(f"Invalid RAW SHA256 in {manifest_path}:{row_number}")

        name_shape = parse_shape_from_name(file_name)
        manifest_shape_text = row.get("RawShape", "").strip()
        if manifest_shape_text:
            manifest_shape = parse_manifest_shape(manifest_shape_text)
            if manifest_shape != name_shape:
                raise ValueError(
                    f"Manifest/file-name shape mismatch in {manifest_path}:{row_number}: "
                    f"{manifest_shape} != {name_shape}"
                )

        raw_path = (class_root / board / "drill" / file_name).resolve()
        if not is_within(raw_path, dataset_root):
            raise ValueError(f"RAW path escapes dataset root: {raw_path}")
        if not raw_path.is_file():
            raise FileNotFoundError(f"RAW file not found: {raw_path}")

        expected_bytes = name_shape[0] * name_shape[1] * name_shape[2] * RAW_DTYPE_BYTES
        actual_bytes = raw_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"RAW byte-size mismatch: {raw_path} has {actual_bytes}, "
                f"expected {expected_bytes} for W/H/D={name_shape}"
            )

        shape_whd = "x".join(str(value) for value in name_shape)
        shape_dhw = "x".join(str(value) for value in reversed(name_shape))
        sample_id = make_sample_id(label, board, sequence, raw_hash)
        samples.append(
            {
                "sample_id": sample_id,
                "board": board,
                "sequence": sequence,
                "label": label,
                "label_index": label_index,
                "raw_shape_whd": shape_whd,
                "input_shape_dhw": shape_dhw,
                "raw_relative_path": str(raw_path.relative_to(dataset_root)),
                "raw_file_name": file_name,
                "raw_sha256": raw_hash,
                "validation_fold": -1,
            }
        )

    return samples


def load_and_audit_samples(
    dataset_root: Path,
    expected_normal: int,
    expected_defective: int,
) -> list[dict[str, object]]:
    dataset_root = dataset_root.resolve()
    normal_root = dataset_root / NORMAL_DIR_NAME
    defect_root = dataset_root / DEFECT_DIR_NAME

    samples = load_class_samples(normal_root, "normal", 0, dataset_root)
    samples += load_class_samples(defect_root, "defective", 1, dataset_root)

    label_counts = Counter(str(row["label"]) for row in samples)
    expected = {"normal": expected_normal, "defective": expected_defective}
    for label, expected_count in expected.items():
        if expected_count >= 0 and label_counts[label] != expected_count:
            raise ValueError(
                f"Unexpected {label} count: {label_counts[label]}, expected {expected_count}"
            )

    checks = {
        "sample_id": [str(row["sample_id"]) for row in samples],
        "board_sequence": [
            f"{row['board']}|{row['sequence']}" for row in samples
        ],
        "raw_relative_path": [
            str(row["raw_relative_path"]).lower() for row in samples
        ],
        "raw_sha256": [str(row["raw_sha256"]) for row in samples],
    }
    for name, values in checks.items():
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        if duplicates:
            preview = ", ".join(duplicates[:5])
            raise ValueError(f"Duplicate {name} values found: {preview}")

    boards = {str(row["board"]) for row in samples}
    if len(boards) != 15:
        raise ValueError(f"Expected 15 boards, found {len(boards)}")
    return samples


def assign_validation_folds(
    samples: list[dict[str, object]],
    n_splits: int,
    seed: int,
) -> None:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

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
    rng = random.Random(seed)

    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for (board, label_index, shape), group in ordered_groups:
        rng.shuffle(group)
        base, remainder = divmod(len(group), n_splits)
        quotas = [base] * n_splits

        for fold in range(n_splits):
            board_label_counts[(board, label_index)][fold] += base
            board_total_counts[board][fold] += base
            shape_label_counts[(shape, label_index)][fold] += base
            label_counts[label_index][fold] += base
            total_counts[fold] += base

        tie_order = list(range(n_splits))
        rng.shuffle(tie_order)
        tie_rank = {fold: rank for rank, fold in enumerate(tie_order)}
        chosen: set[int] = set()
        for _ in range(remainder):
            candidates = [fold for fold in range(n_splits) if fold not in chosen]
            fold = min(
                candidates,
                key=lambda idx: (
                    board_label_counts[(board, label_index)][idx],
                    board_total_counts[board][idx],
                    shape_label_counts[(shape, label_index)][idx],
                    label_counts[label_index][idx],
                    total_counts[idx],
                    tie_rank[idx],
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


def audit_assignments(samples: list[dict[str, object]], n_splits: int) -> None:
    assigned = Counter(int(row["validation_fold"]) for row in samples)
    if set(assigned) != set(range(n_splits)):
        raise ValueError(f"Invalid validation fold set: {sorted(assigned)}")

    strata = defaultdict(lambda: [0] * n_splits)
    for row in samples:
        key = (row["board"], row["label_index"], row["raw_shape_whd"])
        strata[key][int(row["validation_fold"])] += 1
    for key, counts in strata.items():
        if max(counts) - min(counts) > 1:
            raise ValueError(f"Stratum is not evenly distributed: {key} -> {counts}")

    all_ids = {str(row["sample_id"]) for row in samples}
    for fold in range(n_splits):
        train_ids = {
            str(row["sample_id"])
            for row in samples
            if int(row["validation_fold"]) != fold
        }
        val_ids = {
            str(row["sample_id"])
            for row in samples
            if int(row["validation_fold"]) == fold
        }
        if train_ids & val_ids:
            raise ValueError(f"Train/validation leakage detected in fold {fold}")
        if train_ids | val_ids != all_ids:
            raise ValueError(f"Fold {fold} does not cover all samples")


def count_rows(rows: list[dict[str, object]]) -> dict[str, int]:
    labels = Counter(str(row["label"]) for row in rows)
    return {
        "total": len(rows),
        "normal": labels["normal"],
        "defective": labels["defective"],
    }


def build_summary_rows(samples: list[dict[str, object]], n_splits: int):
    fold_rows = []
    board_rows = []
    shape_rows = []

    boards = sorted({str(row["board"]) for row in samples})
    shapes = sorted({str(row["raw_shape_whd"]) for row in samples})
    for fold in range(n_splits):
        for split in ("train", "val"):
            selected = [
                row
                for row in samples
                if (int(row["validation_fold"]) != fold) == (split == "train")
            ]
            counts = count_rows(selected)
            fold_rows.append({"fold": fold, "split": split, **counts})

            for board in boards:
                board_selected = [row for row in selected if row["board"] == board]
                board_rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "board": board,
                        **count_rows(board_selected),
                    }
                )

            for shape in shapes:
                shape_selected = [
                    row for row in selected if row["raw_shape_whd"] == shape
                ]
                shape_rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "raw_shape_whd": shape,
                        **count_rows(shape_selected),
                    }
                )
    return fold_rows, board_rows, shape_rows


def write_outputs(
    samples: list[dict[str, object]],
    output_root: Path,
    dataset_root: Path,
    n_splits: int,
    seed: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    ordered_samples = sorted(
        samples,
        key=lambda row: (
            str(row["board"]),
            int(row["label_index"]),
            str(row["raw_shape_whd"]),
            str(row["sequence"]),
            str(row["raw_file_name"]),
        ),
    )
    write_csv(
        output_root / "all_samples_with_validation_fold.csv",
        OUTPUT_FIELDS,
        ordered_samples,
    )

    for fold in range(n_splits):
        fold_dir = output_root / f"fold_{fold}"
        train_rows = [
            row for row in ordered_samples if int(row["validation_fold"]) != fold
        ]
        val_rows = [
            row for row in ordered_samples if int(row["validation_fold"]) == fold
        ]
        write_csv(fold_dir / "train.csv", OUTPUT_FIELDS, train_rows)
        write_csv(fold_dir / "val.csv", OUTPUT_FIELDS, val_rows)

    fold_rows, board_rows, shape_rows = build_summary_rows(samples, n_splits)
    write_csv(
        output_root / "fold_split_summary.csv",
        ["fold", "split", "total", "normal", "defective"],
        fold_rows,
    )
    write_csv(
        output_root / "fold_board_summary.csv",
        ["fold", "split", "board", "total", "normal", "defective"],
        board_rows,
    )
    write_csv(
        output_root / "fold_shape_summary.csv",
        ["fold", "split", "raw_shape_whd", "total", "normal", "defective"],
        shape_rows,
    )

    rare_rows = []
    strata = defaultdict(list)
    for row in samples:
        strata[(row["board"], row["label"], row["raw_shape_whd"])].append(row)
    for (board, label, shape), rows in sorted(strata.items()):
        if len(rows) < n_splits:
            rare_rows.append(
                {
                    "board": board,
                    "label": label,
                    "raw_shape_whd": shape,
                    "sample_count": len(rows),
                    "validation_folds_present": ",".join(
                        str(value)
                        for value in sorted(
                            {int(row["validation_fold"]) for row in rows}
                        )
                    ),
                }
            )
    write_csv(
        output_root / "rare_strata_audit.csv",
        [
            "board",
            "label",
            "raw_shape_whd",
            "sample_count",
            "validation_folds_present",
        ],
        rare_rows,
    )

    config = {
        "dataset_relative_layout": str(DATASET_RELATIVE_PATH),
        "portable_raw_path_column": "raw_relative_path",
        "contains_absolute_data_paths": False,
        "normal_manifest_relative": str(Path(NORMAL_DIR_NAME) / MANIFEST_NAME),
        "defective_manifest_relative": str(Path(DEFECT_DIR_NAME) / MANIFEST_NAME),
        "n_splits": n_splits,
        "split_seed": seed,
        "split_method": "within_board_label_shape_round_robin_balanced",
        "train_fraction": (n_splits - 1) / n_splits,
        "validation_fraction": 1 / n_splits,
        "internal_test_set": False,
        "raw_files_copied": False,
        "sample_count": len(samples),
        "normal_count": sum(int(row["label_index"]) == 0 for row in samples),
        "defective_count": sum(int(row["label_index"]) == 1 for row in samples),
        "board_count": len({str(row["board"]) for row in samples}),
    }
    with (output_root / "split_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.n_splits != N_SPLITS:
        raise ValueError(
            f"This dedicated experiment requires exactly {N_SPLITS} folds; "
            f"got {args.n_splits}"
        )
    dataset_root = resolve_dataset_root(args.dataset_root)
    output_root = args.output_root.resolve()

    print("=" * 80)
    print("Multi-board clear binary dataset: 5-fold train/validation split")
    print(f"Dataset (read only): {dataset_root}")
    print(f"Split manifests:     {output_root}")
    print(f"Folds/seed:          {args.n_splits}/{args.seed}")
    print("No internal test set; every sample is validation exactly once.")
    print("=" * 80)

    samples = load_and_audit_samples(
        dataset_root,
        expected_normal=args.expected_normal,
        expected_defective=args.expected_defective,
    )
    assign_validation_folds(samples, args.n_splits, args.seed)
    audit_assignments(samples, args.n_splits)
    write_outputs(
        samples,
        output_root,
        dataset_root,
        args.n_splits,
        args.seed,
    )

    fold_rows, _, _ = build_summary_rows(samples, args.n_splits)
    for row in fold_rows:
        print(
            f"fold_{row['fold']} {row['split']:5s}: total={row['total']:4d}, "
            f"normal={row['normal']:4d}, defective={row['defective']:4d}"
        )
    print("=" * 80)
    print("Split creation and leakage audit passed.")
    print(f"Saved: {output_root}")


if __name__ == "__main__":
    main()
