from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(r"F:\325_275_and_sphere_data\钻孔数据325_275")
CLEAR_DATASET_FOLDER = "clear_binary_dataset"
DEFAULT_SEED = 20260731
MIN_NORMAL_PER_BOARD = 100
NORMALS_PER_OBVIOUS_DEFECT = 2
GRAY_BINS = 5

DRILL_SEQUENCE_RE = re.compile(r"^drill-(\d+)-")
INITIAL_FEATURE_RE = re.compile(r"^(\d+)$")


@dataclass
class Candidate:
    board: str
    sequence: int
    raw_path: Path
    feature_path: Path
    shape: str
    raw_mean: float = 0.0
    gray_bin: int = -1


@dataclass
class BoardData:
    board: str
    board_path: Path
    obvious_defects: int
    quota: int
    source_feature_count: int
    excluded_ng_sequences: int
    excluded_obvious_sequences: int
    ambiguous_feature_sequences: int
    ambiguous_raw_sequences: int
    defect_shape_counts: dict[str, int]
    candidates: list[Candidate]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic normal samples grouped by board."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Transactionally replace an existing normal_samples_by_board dataset.",
    )
    return parser.parse_args()


def sequence_from_drill_name(path: Path) -> int:
    match = DRILL_SEQUENCE_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse drill sequence: {path}")
    return int(match.group(1))


def sequence_from_feature(path: Path, initial_board: bool) -> int:
    if initial_board:
        match = INITIAL_FEATURE_RE.match(path.stem)
        if not match:
            raise ValueError(f"Cannot parse initial-board feature sequence: {path}")
        return int(match.group(1))
    return sequence_from_drill_name(path)


def shape_from_raw_name(path: Path) -> tuple[str, int]:
    parts = path.stem.split("-")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse RAW shape: {path}")
    geometry = parts[2].split("_")
    if len(geometry) < 4:
        raise ValueError(f"Cannot parse RAW geometry: {path}")
    dims = tuple(int(value) for value in geometry[-3:])
    expected_bytes = math.prod(dims)
    return "x".join(str(value) for value in dims), expected_bytes


def load_defect_counts(root: Path) -> dict[str, int]:
    summary_path = root / "obvious_defects_by_board" / "board_summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {
        row["Board"]: int(row["ObviousDefectPairs"])
        for row in rows
        if row["Board"] != "TOTAL"
    }
    if len(counts) != 15:
        raise RuntimeError(f"Expected 15 boards in {summary_path}, found {len(counts)}")
    return counts


def load_obvious_sequences(root: Path) -> dict[str, set[int]]:
    manifest_path = root / "obvious_defects_by_board" / "all_boards_manifest.csv"
    result: dict[str, set[int]] = defaultdict(set)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["Board"]].add(int(row["Sequence"]))
    return result


def load_defect_shape_counts(root: Path) -> dict[str, Counter[str]]:
    manifest_path = root / "obvious_defects_by_board" / "all_boards_manifest.csv"
    result: dict[str, Counter[str]] = defaultdict(Counter)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            shape, _ = shape_from_raw_name(Path(row["RawFileName"]))
            result[row["Board"]][shape] += 1
    return result


def discover_board_paths(root: Path, board_names: set[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {"initial_board": root / "initial_board"}
    for group in ("275", "325"):
        group_path = root / group
        for child in group_path.iterdir():
            if child.is_dir() and child.name in board_names:
                paths[child.name] = child
    missing = sorted(board_names - paths.keys())
    if missing:
        raise RuntimeError(f"Missing board directories: {missing}")
    return paths


def grouped_by_sequence(
    paths: list[Path], *, initial_features: bool = False
) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        sequence = (
            sequence_from_feature(path, True)
            if initial_features
            else sequence_from_drill_name(path)
        )
        grouped[sequence].append(path)
    return grouped


def collect_board_data(
    root: Path,
    board: str,
    board_path: Path,
    obvious_defects: int,
    obvious_sequences: set[int],
    defect_shape_counts: Counter[str],
) -> BoardData:
    initial = board == "initial_board"
    feature_dir = board_path / "drill_feature_minus_ng"
    raw_dir = board_path / ("drill_normal_3d" if initial else "drill")
    if not feature_dir.is_dir() or not raw_dir.is_dir():
        raise RuntimeError(f"Missing source folder for {board}")

    feature_paths = sorted(path for path in feature_dir.iterdir() if path.is_file())
    raw_paths = sorted(raw_dir.glob("*.raw"))
    features = grouped_by_sequence(feature_paths, initial_features=initial)
    raws = grouped_by_sequence(raw_paths)

    ng_sequences: set[int] = set()
    if not initial:
        ng_dir = board_path / "drill_ng"
        ng_sequences = {
            sequence_from_drill_name(path)
            for path in ng_dir.iterdir()
            if path.is_file()
        }

    candidates: list[Candidate] = []
    ambiguous_feature = 0
    ambiguous_raw = 0
    excluded_obvious = 0
    for sequence in sorted(features):
        if sequence in ng_sequences:
            continue
        if sequence in obvious_sequences:
            excluded_obvious += 1
            continue
        feature_matches = features[sequence]
        raw_matches = raws.get(sequence, [])
        if len(feature_matches) != 1:
            ambiguous_feature += 1
            continue
        if len(raw_matches) != 1:
            ambiguous_raw += 1
            continue
        raw_path = raw_matches[0]
        shape, expected_bytes = shape_from_raw_name(raw_path)
        if raw_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"Unexpected RAW size for {raw_path}: "
                f"{raw_path.stat().st_size} != {expected_bytes}"
            )
        candidates.append(
            Candidate(
                board=board,
                sequence=sequence,
                raw_path=raw_path,
                feature_path=feature_matches[0],
                shape=shape,
            )
        )

    quota = max(MIN_NORMAL_PER_BOARD, NORMALS_PER_OBVIOUS_DEFECT * obvious_defects)
    if len(candidates) < quota:
        raise RuntimeError(
            f"Not enough unambiguous normal candidates for {board}: "
            f"{len(candidates)} < {quota}"
        )
    return BoardData(
        board=board,
        board_path=board_path,
        obvious_defects=obvious_defects,
        quota=quota,
        source_feature_count=len(feature_paths),
        excluded_ng_sequences=len(ng_sequences),
        excluded_obvious_sequences=excluded_obvious,
        ambiguous_feature_sequences=ambiguous_feature,
        ambiguous_raw_sequences=ambiguous_raw,
        defect_shape_counts=dict(defect_shape_counts),
        candidates=candidates,
    )


def calculate_raw_means(board_data: BoardData) -> None:
    print(
        f"[{board_data.board}] reading {len(board_data.candidates)} candidate RAW files",
        flush=True,
    )
    for index, candidate in enumerate(board_data.candidates, start=1):
        values = np.fromfile(candidate.raw_path, dtype=np.uint8)
        candidate.raw_mean = float(values.mean(dtype=np.float64))
        if index % 500 == 0:
            print(f"  {index}/{len(board_data.candidates)}", flush=True)


def assign_gray_bins(candidates: list[Candidate]) -> None:
    by_shape: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_shape[candidate.shape].append(candidate)
    for shape_candidates in by_shape.values():
        ordered = sorted(shape_candidates, key=lambda item: (item.raw_mean, item.sequence))
        count = len(ordered)
        for rank, candidate in enumerate(ordered):
            candidate.gray_bin = min(GRAY_BINS - 1, rank * GRAY_BINS // count)


def proportional_allocation(counts: dict[tuple[str, int], int], quota: int) -> dict:
    total = sum(counts.values())
    if quota > total:
        raise ValueError(f"Quota {quota} exceeds available candidates {total}")
    exact = {key: quota * count / total for key, count in counts.items()}
    allocation = {key: min(counts[key], math.floor(value)) for key, value in exact.items()}
    remaining = quota - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - allocation[key], counts[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < counts[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Could not complete proportional allocation")
    return allocation


def weighted_allocation(
    weights: dict[str, int], capacities: dict[str, int], quota: int
) -> dict[str, int]:
    missing = sorted(set(weights) - set(capacities))
    if missing:
        raise RuntimeError(f"No normal candidates for defect shapes: {missing}")
    if quota > sum(capacities[shape] for shape in weights):
        raise RuntimeError("Normal quota exceeds capacity of defect-matched shapes")
    total_weight = sum(weights.values())
    exact = {shape: quota * weight / total_weight for shape, weight in weights.items()}
    allocation = {
        shape: min(capacities[shape], math.floor(value))
        for shape, value in exact.items()
    }
    remaining = quota - sum(allocation.values())
    order = sorted(
        weights,
        key=lambda shape: (
            exact[shape] - allocation[shape],
            weights[shape],
            shape,
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for shape in order:
            if allocation[shape] < capacities[shape]:
                allocation[shape] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Could not complete defect-shape allocation")
    return allocation


def board_seed(seed: int, board: str) -> int:
    digest = hashlib.sha256(board.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:8], "big")


def select_candidates(board_data: BoardData, seed: int) -> list[Candidate]:
    assign_gray_bins(board_data.candidates)
    by_shape: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in board_data.candidates:
        by_shape[candidate.shape].append(candidate)

    rng = random.Random(board_seed(seed, board_data.board))
    selected: list[Candidate] = []
    if not board_data.defect_shape_counts:
        strata: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
        for candidate in board_data.candidates:
            strata[(candidate.shape, candidate.gray_bin)].append(candidate)
        allocation = proportional_allocation(
            {key: len(items) for key, items in strata.items()}, board_data.quota
        )
        for key in sorted(strata):
            items = sorted(
                strata[key], key=lambda item: (item.sequence, item.raw_path.name)
            )
            rng.shuffle(items)
            selected.extend(items[: allocation[key]])
        selected.sort(key=lambda item: item.sequence)
        if len(selected) != board_data.quota:
            raise RuntimeError(
                f"Selection count mismatch for {board_data.board}: "
                f"{len(selected)} != {board_data.quota}"
            )
        return selected

    shape_allocation = weighted_allocation(
        board_data.defect_shape_counts,
        {shape: len(items) for shape, items in by_shape.items()},
        board_data.quota,
    )
    for shape in sorted(shape_allocation):
        gray_strata: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
        for candidate in by_shape[shape]:
            gray_strata[(shape, candidate.gray_bin)].append(candidate)
        gray_allocation = proportional_allocation(
            {key: len(items) for key, items in gray_strata.items()},
            shape_allocation[shape],
        )
        for key in sorted(gray_strata):
            items = sorted(
                gray_strata[key], key=lambda item: (item.sequence, item.raw_path.name)
            )
            rng.shuffle(items)
            selected.extend(items[: gray_allocation[key]])
    selected.sort(key=lambda item: item.sequence)
    if len(selected) != board_data.quota:
        raise RuntimeError(
            f"Selection count mismatch for {board_data.board}: "
            f"{len(selected)} != {board_data.quota}"
        )
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


MANIFEST_FIELDS = [
    "Board",
    "Sequence",
    "Label",
    "ObviousDefectsOnBoard",
    "BoardNormalQuota",
    "SamplingSeed",
    "RawShape",
    "RawMean",
    "GrayBin",
    "RawFileName",
    "FeatureFileName",
    "RawSource",
    "FeatureSource",
    "RawDestination",
    "FeatureDestination",
    "RawSHA256",
    "FeatureSHA256",
]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_board_selection(
    temp_root: Path,
    final_root: Path,
    board_data: BoardData,
    selected: list[Candidate],
    seed: int,
) -> list[dict]:
    board_output = temp_root / board_data.board
    raw_output = board_output / "drill"
    feature_output = board_output / "drill_feature"
    raw_output.mkdir(parents=True)
    feature_output.mkdir()
    rows: list[dict] = []
    for index, candidate in enumerate(selected, start=1):
        raw_destination = raw_output / candidate.raw_path.name
        feature_destination = feature_output / candidate.feature_path.name
        final_raw_destination = final_root / board_data.board / "drill" / candidate.raw_path.name
        final_feature_destination = (
            final_root / board_data.board / "drill_feature" / candidate.feature_path.name
        )
        shutil.copy2(candidate.raw_path, raw_destination)
        shutil.copy2(candidate.feature_path, feature_destination)
        raw_hash = sha256_file(candidate.raw_path)
        feature_hash = sha256_file(candidate.feature_path)
        if sha256_file(raw_destination) != raw_hash:
            raise RuntimeError(f"RAW hash mismatch: {raw_destination}")
        if sha256_file(feature_destination) != feature_hash:
            raise RuntimeError(f"Feature hash mismatch: {feature_destination}")
        rows.append(
            {
                "Board": board_data.board,
                "Sequence": f"{candidate.sequence:04d}",
                "Label": "normal_clear",
                "ObviousDefectsOnBoard": board_data.obvious_defects,
                "BoardNormalQuota": board_data.quota,
                "SamplingSeed": seed,
                "RawShape": candidate.shape,
                "RawMean": f"{candidate.raw_mean:.6f}",
                "GrayBin": candidate.gray_bin,
                "RawFileName": candidate.raw_path.name,
                "FeatureFileName": candidate.feature_path.name,
                "RawSource": str(candidate.raw_path),
                "FeatureSource": str(candidate.feature_path),
                "RawDestination": str(final_raw_destination),
                "FeatureDestination": str(final_feature_destination),
                "RawSHA256": raw_hash,
                "FeatureSHA256": feature_hash,
            }
        )
        if index % 100 == 0:
            print(f"  copied {index}/{len(selected)}", flush=True)
    write_csv(board_output / "manifest.csv", rows, MANIFEST_FIELDS)
    return rows


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_root = root / CLEAR_DATASET_FOLDER
    output = dataset_root / "normal_samples_by_board"
    temp_output = dataset_root / "normal_samples_by_board.__building__"
    backup_output = dataset_root / "normal_samples_by_board.__backup__"
    if temp_output.exists() or backup_output.exists():
        raise RuntimeError(f"Temporary output already exists: {temp_output} or {backup_output}")
    if output.exists() and not args.replace_existing:
        raise RuntimeError(f"Output already exists: {output}")

    defect_counts = load_defect_counts(dataset_root)
    obvious_sequences = load_obvious_sequences(dataset_root)
    defect_shape_counts = load_defect_shape_counts(dataset_root)
    board_paths = discover_board_paths(root, set(defect_counts))
    board_data_list: list[BoardData] = []
    for board, defect_count in defect_counts.items():
        data = collect_board_data(
            root,
            board,
            board_paths[board],
            defect_count,
            obvious_sequences.get(board, set()),
            defect_shape_counts.get(board, Counter()),
        )
        board_data_list.append(data)
        print(
            f"[{board}] candidates={len(data.candidates)}, quota={data.quota}",
            flush=True,
        )

    for data in board_data_list:
        calculate_raw_means(data)

    temp_output.mkdir()
    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for data in board_data_list:
        selected = select_candidates(data, args.seed)
        print(f"[{data.board}] copying {len(selected)} selected holes", flush=True)
        rows = copy_board_selection(temp_output, output, data, selected, args.seed)
        all_rows.extend(rows)
        shape_counts = Counter(candidate.shape for candidate in selected)
        summary_rows.append(
            {
                "Board": data.board,
                "ObviousDefects": data.obvious_defects,
                "RequestedNormal": data.quota,
                "SelectedNormal": len(selected),
                "CandidateHoles": len(data.candidates),
                "SourceFeatureFiles": data.source_feature_count,
                "ExcludedNGSequences": data.excluded_ng_sequences,
                "ExcludedObviousSequencesOutsideNG": data.excluded_obvious_sequences,
                "AmbiguousFeatureSequencesSkipped": data.ambiguous_feature_sequences,
                "AmbiguousRawSequencesSkipped": data.ambiguous_raw_sequences,
                "SelectedShapes": json.dumps(shape_counts, ensure_ascii=True, sort_keys=True),
                "SelectedRawMean": f"{np.mean([item.raw_mean for item in selected]):.6f}",
            }
        )

    expected_total = sum(data.quota for data in board_data_list)
    if len(all_rows) != expected_total:
        raise RuntimeError(f"Total mismatch: {len(all_rows)} != {expected_total}")
    if len({(row["Board"], row["Sequence"]) for row in all_rows}) != expected_total:
        raise RuntimeError("Duplicate board/sequence pair in selected samples")

    write_csv(temp_output / "all_boards_manifest.csv", all_rows, MANIFEST_FIELDS)
    summary_fields = list(summary_rows[0])
    write_csv(temp_output / "board_summary.csv", summary_rows, summary_fields)
    (temp_output / "README.txt").write_text(
        "Normal samples grouped by board.\n"
        f"Rule: max({MIN_NORMAL_PER_BOARD}, "
        f"{NORMALS_PER_OBVIOUS_DEFECT} * obvious defects per board).\n"
        f"Random seed: {args.seed}.\n"
        "Candidates come from drill_feature_minus_ng.\n"
        "Any sequence present in drill_ng or the obvious-defect manifest is excluded.\n"
        "Only holes with exactly one feature image and exactly one RAW file are eligible.\n"
        "For boards with defects, normal shape quotas follow the defect shape ratio.\n"
        "For boards without defects, normal shape quotas follow the candidate pool.\n"
        f"Sampling uses {GRAY_BINS} within-shape gray bins.\n"
        "Each board contains drill, drill_feature, and manifest.csv.\n",
        encoding="ascii",
    )
    if output.exists():
        output.rename(backup_output)
        try:
            temp_output.rename(output)
        except Exception:
            backup_output.rename(output)
            raise
        shutil.rmtree(backup_output)
    else:
        temp_output.rename(output)
    print(f"DONE: {output}", flush=True)
    print(f"TOTAL_SELECTED={expected_total}", flush=True)


if __name__ == "__main__":
    main()
