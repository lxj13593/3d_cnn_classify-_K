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
PROJECT_DATASET_RELATIVE_PATH = Path("datasets")
PROJECT_DATASETS_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_PROJECT_DATASET_ROOT = PROJECT_ROOT / PROJECT_DATASET_RELATIVE_PATH
LEGACY_DATASET_RELATIVE_PATH = (
    Path("325_275_and_sphere_data")
    / "钻孔数据325_275"
    / "clear_binary_dataset"
)
LEGACY_DATASET_PARENT_RELATIVE_PATH = (
    Path("325_275_and_sphere_data")
    / "\u94bb\u5b54\u6570\u636e325_275"
)
DATASET_ROOT_ENV = "DRILL_COMBINED_DATASET_ROOT"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "multiboard_clear_fuzzy_5fold"

CLEAR_DATASET_NAME = "clear_binary_dataset"
FUZZY_DATASET_NAME = "fuzzy_binary_dataset"
NORMAL_DIR_NAME = "normal_samples_by_board"
CLEAR_DEFECT_DIR_NAME = "obvious_defects_by_board"
FUZZY_DEFECT_DIR_NAME = "fuzzy_defects_by_board"
MANIFEST_NAME = "all_boards_manifest.csv"
N_SPLITS = 5
SPLIT_SEED = 42
EXPECTED_CLEAR_NORMAL = 3262
EXPECTED_CLEAR_DEFECTIVE = 1141
EXPECTED_FUZZY_NORMAL = 1169
EXPECTED_FUZZY_DEFECTIVE = 156
RAW_DTYPE_BYTES = 1  # uint8

SHAPE_PATTERN = re.compile(r"_(\d+)_(\d+)_(\d+)-")
OUTPUT_FIELDS = (
    "sample_id",
    "board",
    "sequence",
    "label",
    "label_index",
    "difficulty",
    "source_dataset",
    "source_class_dir",
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
            "combined clear/fuzzy multi-board binary dataset. "
            "No RAW file is copied or modified."
        )
    )
    parser.add_argument(
        "--dataset-parent",
        type=Path,
        default=None,
        help=(
            "Directory containing clear_binary_dataset and fuzzy_binary_dataset. "
            "If omitted, use DRILL_COMBINED_DATASET_ROOT, the project datasets "
            "directory, or mounted-drive discovery."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--expected-clear-normal", type=int, default=EXPECTED_CLEAR_NORMAL
    )
    parser.add_argument(
        "--expected-clear-defective", type=int, default=EXPECTED_CLEAR_DEFECTIVE
    )
    parser.add_argument(
        "--expected-fuzzy-normal", type=int, default=EXPECTED_FUZZY_NORMAL
    )
    parser.add_argument(
        "--expected-fuzzy-defective", type=int, default=EXPECTED_FUZZY_DEFECTIVE
    )
    return parser.parse_args()


def is_dataset_parent(path: Path) -> bool:
    required = (
        path / CLEAR_DATASET_NAME / NORMAL_DIR_NAME / MANIFEST_NAME,
        path / CLEAR_DATASET_NAME / CLEAR_DEFECT_DIR_NAME / MANIFEST_NAME,
        path / FUZZY_DATASET_NAME / NORMAL_DIR_NAME / MANIFEST_NAME,
        path / FUZZY_DATASET_NAME / FUZZY_DEFECT_DIR_NAME / MANIFEST_NAME,
    )
    return all(manifest.is_file() for manifest in required)


def normalize_parent_candidate(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if is_dataset_parent(candidate):
        return candidate
    if candidate.name in {CLEAR_DATASET_NAME, FUZZY_DATASET_NAME}:
        parent = candidate.parent
        if is_dataset_parent(parent):
            return parent
    raise FileNotFoundError(
        "Directory must contain both clear_binary_dataset and "
        f"fuzzy_binary_dataset: {candidate}"
    )


def project_dataset_roots() -> list[Path]:
    if is_dataset_parent(DEFAULT_PROJECT_DATASET_ROOT):
        return [DEFAULT_PROJECT_DATASET_ROOT.resolve()]

    candidates = [PROJECT_DATASETS_ROOT]
    if PROJECT_DATASETS_ROOT.is_dir():
        candidates.extend(
            path for path in PROJECT_DATASETS_ROOT.iterdir() if path.is_dir()
        )
    return sorted(
        {path.resolve() for path in candidates if is_dataset_parent(path)},
        key=lambda path: str(path).lower(),
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


def resolve_dataset_parent(
    explicit_root: Path | None,
    stored_root: str | Path | None = None,
) -> Path:
    if explicit_root is not None:
        return normalize_parent_candidate(explicit_root)

    environment_root = os.environ.get(DATASET_ROOT_ENV, "").strip()
    if environment_root:
        try:
            return normalize_parent_candidate(Path(environment_root))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{DATASET_ROOT_ENV} is not a valid combined dataset parent"
            ) from exc

    project_candidates = project_dataset_roots()
    if len(project_candidates) == 1:
        return project_candidates[0]
    if len(project_candidates) > 1:
        joined = "\n  ".join(str(path) for path in project_candidates)
        raise RuntimeError(
            "Multiple combined datasets were found under the project datasets "
            "directory. Select one with --dataset-parent:\n  " + joined
        )

    if stored_root:
        try:
            return normalize_parent_candidate(Path(stored_root))
        except FileNotFoundError:
            pass

    candidates = []
    for drive_root in mounted_drive_roots():
        candidate = drive_root / LEGACY_DATASET_PARENT_RELATIVE_PATH
        if is_dataset_parent(candidate):
            candidates.append(candidate.resolve())
    candidates = sorted(set(candidates), key=lambda path: str(path).lower())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Could not find a parent containing clear_binary_dataset and "
            "fuzzy_binary_dataset. Put both under "
            f"{DEFAULT_PROJECT_DATASET_ROOT}, pass --dataset-parent, or set "
            f"{DATASET_ROOT_ENV}."
        )
    joined = "\n  ".join(str(path) for path in candidates)
    raise RuntimeError(
        "Multiple matching datasets were found. Select one with "
        "--dataset-parent:\n  "
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


def make_sample_id(
    difficulty: str,
    label: str,
    board: str,
    sequence: str,
    raw_hash: str,
) -> str:
    return (
        f"{difficulty}|{label}|{board}|{sequence}|{raw_hash[:16].lower()}"
    )


def load_class_samples(
    dataset_parent: Path,
    source_dataset: str,
    source_class_dir: str,
    label: str,
    label_index: int,
    difficulty: str,
) -> list[dict[str, object]]:
    class_root = dataset_parent / source_dataset / source_class_dir
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
        if not is_within(raw_path, dataset_parent):
            raise ValueError(f"RAW path escapes dataset parent: {raw_path}")
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
        sample_id = make_sample_id(
            difficulty, label, board, sequence, raw_hash
        )
        samples.append(
            {
                "sample_id": sample_id,
                "board": board,
                "sequence": sequence,
                "label": label,
                "label_index": label_index,
                "difficulty": difficulty,
                "source_dataset": source_dataset,
                "source_class_dir": source_class_dir,
                "raw_shape_whd": shape_whd,
                "input_shape_dhw": shape_dhw,
                "raw_relative_path": str(raw_path.relative_to(dataset_parent)),
                "raw_file_name": file_name,
                "raw_sha256": raw_hash,
                "validation_fold": -1,
            }
        )

    return samples


def load_and_audit_samples(
    dataset_parent: Path,
    expected_counts: dict[tuple[str, str], int],
) -> list[dict[str, object]]:
    dataset_parent = dataset_parent.resolve()
    sources = (
        (CLEAR_DATASET_NAME, NORMAL_DIR_NAME, "normal", 0, "clear"),
        (
            CLEAR_DATASET_NAME,
            CLEAR_DEFECT_DIR_NAME,
            "defective",
            1,
            "clear",
        ),
        (FUZZY_DATASET_NAME, NORMAL_DIR_NAME, "normal", 0, "fuzzy"),
        (
            FUZZY_DATASET_NAME,
            FUZZY_DEFECT_DIR_NAME,
            "defective",
            1,
            "fuzzy",
        ),
    )
    samples: list[dict[str, object]] = []
    for source_dataset, class_dir, label, label_index, difficulty in sources:
        samples.extend(
            load_class_samples(
                dataset_parent,
                source_dataset,
                class_dir,
                label,
                label_index,
                difficulty,
            )
        )

    actual_counts = Counter(
        (str(row["difficulty"]), str(row["label"])) for row in samples
    )
    for key, expected_count in expected_counts.items():
        if expected_count >= 0 and actual_counts[key] != expected_count:
            raise ValueError(
                f"Unexpected {key} count: {actual_counts[key]}, "
                f"expected {expected_count}"
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
    for difficulty in ("clear", "fuzzy"):
        difficulty_boards = {
            str(row["board"])
            for row in samples
            if row["difficulty"] == difficulty
        }
        if len(difficulty_boards) != 15:
            raise ValueError(
                f"Expected all 15 boards in {difficulty}, "
                f"found {len(difficulty_boards)}"
            )
    return samples


def assign_validation_folds(
    samples: list[dict[str, object]],
    n_splits: int,
    seed: int,
) -> None:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    groups: dict[tuple[str, int, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for sample in samples:
        key = (
            str(sample["board"]),
            int(sample["label_index"]),
            str(sample["difficulty"]),
            str(sample["raw_shape_whd"]),
        )
        groups[key].append(sample)

    board_label_counts = defaultdict(lambda: [0] * n_splits)
    board_total_counts = defaultdict(lambda: [0] * n_splits)
    shape_label_counts = defaultdict(lambda: [0] * n_splits)
    difficulty_label_counts = defaultdict(lambda: [0] * n_splits)
    difficulty_counts = defaultdict(lambda: [0] * n_splits)
    label_counts = defaultdict(lambda: [0] * n_splits)
    total_counts = [0] * n_splits
    rng = random.Random(seed)

    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for (board, label_index, difficulty, shape), group in ordered_groups:
        rng.shuffle(group)
        base, remainder = divmod(len(group), n_splits)
        quotas = [base] * n_splits

        for fold in range(n_splits):
            board_label_counts[(board, label_index)][fold] += base
            board_total_counts[board][fold] += base
            shape_label_counts[(shape, label_index)][fold] += base
            difficulty_label_counts[(difficulty, label_index)][fold] += base
            difficulty_counts[difficulty][fold] += base
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
                    difficulty_label_counts[(difficulty, label_index)][idx],
                    difficulty_counts[difficulty][idx],
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
            difficulty_label_counts[(difficulty, label_index)][fold] += 1
            difficulty_counts[difficulty][fold] += 1
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
        key = (
            row["board"],
            row["label_index"],
            row["difficulty"],
            row["raw_shape_whd"],
        )
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
        for split_name, selected in (
            (
                "train",
                [row for row in samples if int(row["validation_fold"]) != fold],
            ),
            (
                "val",
                [row for row in samples if int(row["validation_fold"]) == fold],
            ),
        ):
            boards = {str(row["board"]) for row in selected}
            difficulties = {str(row["difficulty"]) for row in selected}
            if len(boards) != 15:
                raise ValueError(
                    f"Fold {fold} {split_name} contains {len(boards)} boards"
                )
            if difficulties != {"clear", "fuzzy"}:
                raise ValueError(
                    f"Fold {fold} {split_name} difficulty set: {difficulties}"
                )


def count_rows(rows: list[dict[str, object]]) -> dict[str, int]:
    labels = Counter(str(row["label"]) for row in rows)
    difficulty_labels = Counter(
        (str(row["difficulty"]), str(row["label"])) for row in rows
    )
    return {
        "total": len(rows),
        "normal": labels["normal"],
        "defective": labels["defective"],
        "clear_normal": difficulty_labels[("clear", "normal")],
        "clear_defective": difficulty_labels[("clear", "defective")],
        "fuzzy_normal": difficulty_labels[("fuzzy", "normal")],
        "fuzzy_defective": difficulty_labels[("fuzzy", "defective")],
    }


def build_summary_rows(samples: list[dict[str, object]], n_splits: int):
    fold_rows = []
    board_rows = []
    shape_rows = []
    difficulty_rows = []

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

            for difficulty in ("clear", "fuzzy"):
                difficulty_selected = [
                    row for row in selected if row["difficulty"] == difficulty
                ]
                difficulty_rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "difficulty": difficulty,
                        **count_rows(difficulty_selected),
                    }
                )

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
    return fold_rows, board_rows, shape_rows, difficulty_rows


def write_outputs(
    samples: list[dict[str, object]],
    output_root: Path,
    dataset_parent: Path,
    n_splits: int,
    seed: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    ordered_samples = sorted(
        samples,
        key=lambda row: (
            str(row["board"]),
            str(row["difficulty"]),
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

    fold_rows, board_rows, shape_rows, difficulty_rows = build_summary_rows(
        samples, n_splits
    )
    count_fields = [
        "total",
        "normal",
        "defective",
        "clear_normal",
        "clear_defective",
        "fuzzy_normal",
        "fuzzy_defective",
    ]
    write_csv(
        output_root / "fold_split_summary.csv",
        ["fold", "split", *count_fields],
        fold_rows,
    )
    write_csv(
        output_root / "fold_board_summary.csv",
        ["fold", "split", "board", *count_fields],
        board_rows,
    )
    write_csv(
        output_root / "fold_shape_summary.csv",
        ["fold", "split", "raw_shape_whd", *count_fields],
        shape_rows,
    )
    write_csv(
        output_root / "fold_difficulty_summary.csv",
        ["fold", "split", "difficulty", *count_fields],
        difficulty_rows,
    )

    rare_rows = []
    strata = defaultdict(list)
    for row in samples:
        strata[
            (
                row["board"],
                row["label"],
                row["difficulty"],
                row["raw_shape_whd"],
            )
        ].append(row)
    for (board, label, difficulty, shape), rows in sorted(strata.items()):
        if len(rows) < n_splits:
            rare_rows.append(
                {
                    "board": board,
                    "label": label,
                    "difficulty": difficulty,
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
            "difficulty",
            "raw_shape_whd",
            "sample_count",
            "validation_folds_present",
        ],
        rare_rows,
    )

    config = {
        "dataset_parent_relative_layout": str(PROJECT_DATASET_RELATIVE_PATH),
        "portable_raw_path_column": "raw_relative_path",
        "contains_absolute_data_paths": False,
        "source_manifests": {
            "clear_normal": str(
                Path(CLEAR_DATASET_NAME) / NORMAL_DIR_NAME / MANIFEST_NAME
            ),
            "clear_defective": str(
                Path(CLEAR_DATASET_NAME)
                / CLEAR_DEFECT_DIR_NAME
                / MANIFEST_NAME
            ),
            "fuzzy_normal": str(
                Path(FUZZY_DATASET_NAME) / NORMAL_DIR_NAME / MANIFEST_NAME
            ),
            "fuzzy_defective": str(
                Path(FUZZY_DATASET_NAME)
                / FUZZY_DEFECT_DIR_NAME
                / MANIFEST_NAME
            ),
        },
        "n_splits": n_splits,
        "split_seed": seed,
        "split_method": (
            "within_board_label_difficulty_shape_round_robin_balanced"
        ),
        "train_fraction": (n_splits - 1) / n_splits,
        "validation_fraction": 1 / n_splits,
        "internal_test_set": False,
        "raw_files_copied": False,
        "sample_count": len(samples),
        "normal_count": sum(int(row["label_index"]) == 0 for row in samples),
        "defective_count": sum(int(row["label_index"]) == 1 for row in samples),
        "clear_count": sum(row["difficulty"] == "clear" for row in samples),
        "fuzzy_count": sum(row["difficulty"] == "fuzzy" for row in samples),
        "clear_normal_count": sum(
            row["difficulty"] == "clear" and row["label"] == "normal"
            for row in samples
        ),
        "clear_defective_count": sum(
            row["difficulty"] == "clear" and row["label"] == "defective"
            for row in samples
        ),
        "fuzzy_normal_count": sum(
            row["difficulty"] == "fuzzy" and row["label"] == "normal"
            for row in samples
        ),
        "fuzzy_defective_count": sum(
            row["difficulty"] == "fuzzy" and row["label"] == "defective"
            for row in samples
        ),
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
    dataset_parent = resolve_dataset_parent(args.dataset_parent)
    output_root = args.output_root.resolve()

    print("=" * 80)
    print("Multi-board clear + fuzzy dataset: 5-fold train/validation split")
    print(f"Dataset parent (read only): {dataset_parent}")
    print(f"Split manifests:     {output_root}")
    print(f"Folds/seed:          {args.n_splits}/{args.seed}")
    print("No internal test set; every sample is validation exactly once.")
    print("=" * 80)

    samples = load_and_audit_samples(
        dataset_parent,
        expected_counts={
            ("clear", "normal"): args.expected_clear_normal,
            ("clear", "defective"): args.expected_clear_defective,
            ("fuzzy", "normal"): args.expected_fuzzy_normal,
            ("fuzzy", "defective"): args.expected_fuzzy_defective,
        },
    )
    assign_validation_folds(samples, args.n_splits, args.seed)
    audit_assignments(samples, args.n_splits)
    write_outputs(
        samples,
        output_root,
        dataset_parent,
        args.n_splits,
        args.seed,
    )

    fold_rows, _, _, _ = build_summary_rows(samples, args.n_splits)
    for row in fold_rows:
        print(
            f"fold_{row['fold']} {row['split']:5s}: total={row['total']:4d}, "
            f"normal={row['normal']:4d}, defective={row['defective']:4d}, "
            f"clear={row['clear_normal'] + row['clear_defective']:4d}, "
            f"fuzzy={row['fuzzy_normal'] + row['fuzzy_defective']:4d}"
        )
    print("=" * 80)
    print("Split creation and leakage audit passed.")
    print(f"Saved: {output_root}")


if __name__ == "__main__":
    main()
