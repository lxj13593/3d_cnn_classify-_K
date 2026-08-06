from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(r"F:\325_275_and_sphere_data\钻孔数据325_275")
OUTPUT_FOLDER = "fuzzy_binary_dataset"
NORMAL_FOLDER = "normal_samples_by_board"
DEFECT_FOLDER = "fuzzy_defects_by_board"

EXTERNAL_CANDIDATE_FOLDER = "drill_ng_minus_obvious_defects"
EXTERNAL_SCREENED_NORMAL_FOLDER = "drill_ng_minus_obvious_defects_copy"
INITIAL_CANDIDATE_FOLDER = "drill_defect_minus_obvious_defects"
INITIAL_SCREENED_NORMAL_FOLDER = "drill_defect_minus_obvious_defects_copy"

DRILL_SEQUENCE_RE = re.compile(r"^drill-(\d+)-", re.IGNORECASE)
INITIAL_FEATURE_RE = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)
FEATURE_SHAPE_RE = re.compile(r"_(\d+)_(\d+)_(\d+)-")


@dataclass(frozen=True)
class BoardSource:
    category: str
    name: str
    board_path: Path
    candidate_dir: Path
    screened_normal_dir: Path
    raw_dir: Path
    initial_board: bool


@dataclass(frozen=True)
class Sample:
    category: str
    board: str
    sequence: int
    label: str
    raw_path: Path
    feature_path: Path
    original_candidate_path: Path
    raw_shape: str
    raw_mean: float


MANIFEST_FIELDS = [
    "Board",
    "Category",
    "Sequence",
    "Label",
    "Difficulty",
    "RawShape",
    "RawMean",
    "RawFileName",
    "FeatureFileName",
    "RawSource",
    "FeatureSource",
    "OriginalCandidateFeature",
    "RawDestination",
    "FeatureDestination",
    "RawSHA256",
    "FeatureSHA256",
]

AUDIT_FIELDS = [
    "Board",
    "Category",
    "Sequence",
    "DerivedLabel",
    "Status",
    "Reason",
    "CandidateFeature",
    "ScreenedCopyFeature",
    "MatchingFeatureCount",
    "MatchingRawCount",
]

SUMMARY_FIELDS = [
    "Board",
    "Category",
    "CandidateFeatureFiles",
    "CopyRemainingFeatureFiles",
    "DerivedDefectFeatureFiles",
    "IncludedFuzzyNormal",
    "IncludedFuzzyDefect",
    "SkippedAmbiguousNormalFiles",
    "SkippedAmbiguousDefectFiles",
    "FuzzyNormalShapes",
    "FuzzyDefectShapes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fuzzy normal/defect dataset from manually screened copies."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Transactionally replace an existing fuzzy_binary_dataset.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def feature_sequence(path: Path, initial_board: bool) -> int:
    match = (
        INITIAL_FEATURE_RE.match(path.name)
        if initial_board
        else DRILL_SEQUENCE_RE.match(path.name)
    )
    if not match:
        raise ValueError(f"Cannot parse feature sequence: {path}")
    return int(match.group(1))


def raw_sequence(path: Path) -> int:
    match = DRILL_SEQUENCE_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse RAW sequence: {path}")
    return int(match.group(1))


def raw_shape(path: Path) -> tuple[str, tuple[int, int, int]]:
    parts = path.stem.split("-")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse RAW shape: {path}")
    geometry = parts[2].split("_")
    if len(geometry) < 4:
        raise ValueError(f"Cannot parse RAW geometry: {path}")
    dimensions = tuple(int(value) for value in geometry[-3:])
    return "x".join(str(value) for value in dimensions), dimensions


def validate_feature_shape(path: Path, dimensions: tuple[int, int, int]) -> None:
    match = FEATURE_SHAPE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse feature shape: {path}")
    feature_dimensions = tuple(int(value) for value in match.groups())
    if feature_dimensions != dimensions:
        raise RuntimeError(
            f"RAW/feature shape mismatch for {path}: "
            f"{feature_dimensions} != {dimensions}"
        )


def discover_boards(root: Path) -> list[BoardSource]:
    boards: list[BoardSource] = []
    for category in ("275", "325"):
        category_root = root / category
        if not category_root.is_dir():
            raise RuntimeError(f"Missing category folder: {category_root}")
        for board_path in sorted(path for path in category_root.iterdir() if path.is_dir()):
            candidate_dir = board_path / EXTERNAL_CANDIDATE_FOLDER
            screened_dir = board_path / EXTERNAL_SCREENED_NORMAL_FOLDER
            if not candidate_dir.exists() and not screened_dir.exists():
                continue
            if not candidate_dir.is_dir() or not screened_dir.is_dir():
                raise RuntimeError(
                    f"Candidate/copy folder pair is incomplete for {board_path}"
                )
            boards.append(
                BoardSource(
                    category=category,
                    name=board_path.name,
                    board_path=board_path,
                    candidate_dir=candidate_dir,
                    screened_normal_dir=screened_dir,
                    raw_dir=board_path / "drill",
                    initial_board=False,
                )
            )

    initial_path = root / "initial_board"
    boards.append(
        BoardSource(
            category="initial",
            name="initial_board",
            board_path=initial_path,
            candidate_dir=initial_path / INITIAL_CANDIDATE_FOLDER,
            screened_normal_dir=initial_path / INITIAL_SCREENED_NORMAL_FOLDER,
            raw_dir=initial_path / "drill_defect_3d",
            initial_board=True,
        )
    )
    for board in boards:
        for required in (
            board.board_path,
            board.candidate_dir,
            board.screened_normal_dir,
            board.raw_dir,
        ):
            if not required.is_dir():
                raise RuntimeError(f"Missing required folder: {required}")
    if len(boards) != 15:
        raise RuntimeError(f"Expected 15 boards, found {len(boards)}")
    return boards


def load_clear_pairs(root: Path) -> set[tuple[str, int]]:
    clear_root = root / "clear_binary_dataset"
    manifests = (
        clear_root / "normal_samples_by_board" / "all_boards_manifest.csv",
        clear_root / "obvious_defects_by_board" / "all_boards_manifest.csv",
    )
    pairs: set[tuple[str, int]] = set()
    for manifest in manifests:
        if not manifest.is_file():
            raise RuntimeError(f"Missing clear-dataset manifest: {manifest}")
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                pair = (row["Board"], int(row["Sequence"]))
                if pair in pairs:
                    raise RuntimeError(f"Duplicate pair in clear dataset: {pair}")
                pairs.add(pair)
    return pairs


def group_features(
    paths: list[Path], initial_board: bool
) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[feature_sequence(path, initial_board)].append(path)
    return grouped


def group_raws(paths: list[Path]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[raw_sequence(path)].append(path)
    return grouped


def collect_board_samples(
    board: BoardSource,
    clear_pairs: set[tuple[str, int]],
) -> tuple[list[Sample], list[dict], dict]:
    candidate_files = sorted(board.candidate_dir.glob("*.jpg"))
    copy_files = sorted(board.screened_normal_dir.glob("*.jpg"))
    candidate_by_name = {path.name: path for path in candidate_files}
    copy_by_name = {path.name: path for path in copy_files}
    if len(candidate_by_name) != len(candidate_files):
        raise RuntimeError(f"Duplicate candidate filenames for {board.name}")
    if len(copy_by_name) != len(copy_files):
        raise RuntimeError(f"Duplicate copy filenames for {board.name}")

    extra_copy_names = sorted(set(copy_by_name) - set(candidate_by_name))
    if extra_copy_names:
        raise RuntimeError(
            f"Copy contains files outside the candidate set for {board.name}: "
            f"{extra_copy_names[:5]}"
        )
    for name, copy_path in copy_by_name.items():
        if sha256_file(copy_path) != sha256_file(candidate_by_name[name]):
            raise RuntimeError(f"Copy content changed instead of deletion: {copy_path}")

    candidate_groups = group_features(candidate_files, board.initial_board)
    copy_groups = group_features(copy_files, board.initial_board)
    raw_groups = group_raws(sorted(board.raw_dir.glob("*.raw")))
    samples: list[Sample] = []
    audit_rows: list[dict] = []
    normal_shapes: Counter[str] = Counter()
    defect_shapes: Counter[str] = Counter()
    skipped_normal = 0
    skipped_defect = 0

    for sequence in sorted(candidate_groups):
        features = sorted(candidate_groups[sequence])
        copy_matches = sorted(copy_groups.get(sequence, []))
        raw_matches = sorted(raw_groups.get(sequence, []))
        if copy_matches and len(copy_matches) != len(features):
            raise RuntimeError(
                f"Only part of duplicated sequence {sequence:04d} remains in copy "
                f"for {board.name}; classify the whole hole consistently."
            )
        label = "normal_fuzzy" if copy_matches else "defective_fuzzy"
        if (board.name, sequence) in clear_pairs:
            raise RuntimeError(
                f"Fuzzy sample overlaps the clear dataset: {board.name}/{sequence:04d}"
            )

        ambiguous_reasons: list[str] = []
        if len(features) != 1:
            ambiguous_reasons.append(f"feature_count={len(features)}")
        if len(raw_matches) != 1:
            ambiguous_reasons.append(f"raw_count={len(raw_matches)}")
        if ambiguous_reasons:
            if label == "normal_fuzzy":
                skipped_normal += len(features)
            else:
                skipped_defect += len(features)
            for feature in features:
                audit_rows.append(
                    {
                        "Board": board.name,
                        "Category": board.category,
                        "Sequence": f"{sequence:04d}",
                        "DerivedLabel": label,
                        "Status": "skipped_ambiguous",
                        "Reason": ";".join(ambiguous_reasons),
                        "CandidateFeature": str(feature),
                        "ScreenedCopyFeature": str(copy_by_name.get(feature.name, "")),
                        "MatchingFeatureCount": len(features),
                        "MatchingRawCount": len(raw_matches),
                    }
                )
            continue

        candidate_feature = features[0]
        feature_source = (
            copy_by_name[candidate_feature.name]
            if label == "normal_fuzzy"
            else candidate_feature
        )
        raw_path = raw_matches[0]
        shape_name, dimensions = raw_shape(raw_path)
        expected_size = math.prod(dimensions)
        if raw_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Unexpected RAW size for {raw_path}: "
                f"{raw_path.stat().st_size} != {expected_size}"
            )
        if not board.initial_board:
            validate_feature_shape(candidate_feature, dimensions)
        values = np.fromfile(raw_path, dtype=np.uint8)
        sample = Sample(
            category=board.category,
            board=board.name,
            sequence=sequence,
            label=label,
            raw_path=raw_path,
            feature_path=feature_source,
            original_candidate_path=candidate_feature,
            raw_shape=shape_name,
            raw_mean=float(values.mean(dtype=np.float64)),
        )
        samples.append(sample)
        target_shapes = normal_shapes if label == "normal_fuzzy" else defect_shapes
        target_shapes[shape_name] += 1
        audit_rows.append(
            {
                "Board": board.name,
                "Category": board.category,
                "Sequence": f"{sequence:04d}",
                "DerivedLabel": label,
                "Status": "included",
                "Reason": "unique_feature_and_raw",
                "CandidateFeature": str(candidate_feature),
                "ScreenedCopyFeature": str(
                    copy_by_name.get(candidate_feature.name, "")
                ),
                "MatchingFeatureCount": 1,
                "MatchingRawCount": 1,
            }
        )

    summary = {
        "Board": board.name,
        "Category": board.category,
        "CandidateFeatureFiles": len(candidate_files),
        "CopyRemainingFeatureFiles": len(copy_files),
        "DerivedDefectFeatureFiles": len(candidate_files) - len(copy_files),
        "IncludedFuzzyNormal": sum(
            sample.label == "normal_fuzzy" for sample in samples
        ),
        "IncludedFuzzyDefect": sum(
            sample.label == "defective_fuzzy" for sample in samples
        ),
        "SkippedAmbiguousNormalFiles": skipped_normal,
        "SkippedAmbiguousDefectFiles": skipped_defect,
        "FuzzyNormalShapes": json.dumps(
            normal_shapes, ensure_ascii=True, sort_keys=True
        ),
        "FuzzyDefectShapes": json.dumps(
            defect_shapes, ensure_ascii=True, sort_keys=True
        ),
    }
    return samples, audit_rows, summary


def manifest_row(
    sample: Sample,
    final_class_root: Path,
    raw_hash: str,
    feature_hash: str,
) -> dict:
    raw_destination = final_class_root / sample.board / "drill" / sample.raw_path.name
    feature_destination = (
        final_class_root
        / sample.board
        / "drill_feature"
        / sample.feature_path.name
    )
    return {
        "Board": sample.board,
        "Category": sample.category,
        "Sequence": f"{sample.sequence:04d}",
        "Label": sample.label,
        "Difficulty": "fuzzy",
        "RawShape": sample.raw_shape,
        "RawMean": f"{sample.raw_mean:.6f}",
        "RawFileName": sample.raw_path.name,
        "FeatureFileName": sample.feature_path.name,
        "RawSource": str(sample.raw_path),
        "FeatureSource": str(sample.feature_path),
        "OriginalCandidateFeature": str(sample.original_candidate_path),
        "RawDestination": str(raw_destination),
        "FeatureDestination": str(feature_destination),
        "RawSHA256": raw_hash,
        "FeatureSHA256": feature_hash,
    }


def copy_samples(
    temp_root: Path,
    final_root: Path,
    boards: list[BoardSource],
    samples: list[Sample],
) -> tuple[list[dict], list[dict]]:
    normal_temp = temp_root / NORMAL_FOLDER
    defect_temp = temp_root / DEFECT_FOLDER
    normal_final = final_root / NORMAL_FOLDER
    defect_final = final_root / DEFECT_FOLDER
    normal_rows: list[dict] = []
    defect_rows: list[dict] = []
    by_board_label: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for board in boards:
        for class_root in (normal_temp, defect_temp):
            board_root = class_root / board.name
            (board_root / "drill").mkdir(parents=True)
            (board_root / "drill_feature").mkdir()

    for index, sample in enumerate(samples, start=1):
        is_normal = sample.label == "normal_fuzzy"
        class_temp = normal_temp if is_normal else defect_temp
        class_final = normal_final if is_normal else defect_final
        board_temp = class_temp / sample.board
        raw_destination = board_temp / "drill" / sample.raw_path.name
        feature_destination = board_temp / "drill_feature" / sample.feature_path.name
        shutil.copy2(sample.raw_path, raw_destination)
        shutil.copy2(sample.feature_path, feature_destination)
        raw_hash = sha256_file(sample.raw_path)
        feature_hash = sha256_file(sample.feature_path)
        if sha256_file(raw_destination) != raw_hash:
            raise RuntimeError(f"RAW hash mismatch: {raw_destination}")
        if sha256_file(feature_destination) != feature_hash:
            raise RuntimeError(f"Feature hash mismatch: {feature_destination}")
        row = manifest_row(sample, class_final, raw_hash, feature_hash)
        if is_normal:
            normal_rows.append(row)
        else:
            defect_rows.append(row)
        by_board_label[(sample.board, sample.label)].append(row)
        if index % 100 == 0:
            print(f"Copied and verified {index}/{len(samples)} samples", flush=True)

    for board in boards:
        normal_board_rows = by_board_label[(board.name, "normal_fuzzy")]
        defect_board_rows = by_board_label[(board.name, "defective_fuzzy")]
        write_csv(
            normal_temp / board.name / "manifest.csv",
            normal_board_rows,
            MANIFEST_FIELDS,
        )
        write_csv(
            defect_temp / board.name / "manifest.csv",
            defect_board_rows,
            MANIFEST_FIELDS,
        )
    write_csv(normal_temp / "all_boards_manifest.csv", normal_rows, MANIFEST_FIELDS)
    write_csv(defect_temp / "all_boards_manifest.csv", defect_rows, MANIFEST_FIELDS)
    return normal_rows, defect_rows


def summary_rows_for_class(
    summary_rows: list[dict], *, normal: bool
) -> list[dict]:
    count_key = "IncludedFuzzyNormal" if normal else "IncludedFuzzyDefect"
    shape_key = "FuzzyNormalShapes" if normal else "FuzzyDefectShapes"
    return [
        {
            "Board": row["Board"],
            "Category": row["Category"],
            "Count": row[count_key],
            "Shapes": row[shape_key],
        }
        for row in summary_rows
    ]


def chinese_document(summary_rows: list[dict]) -> str:
    normal_total = sum(int(row["IncludedFuzzyNormal"]) for row in summary_rows)
    defect_total = sum(int(row["IncludedFuzzyDefect"]) for row in summary_rows)
    skipped_normal = sum(
        int(row["SkippedAmbiguousNormalFiles"]) for row in summary_rows
    )
    skipped_defect = sum(
        int(row["SkippedAmbiguousDefectFiles"]) for row in summary_rows
    )
    ratio = normal_total / defect_total if defect_total else float("inf")
    lines = [
        "# 模糊二分类数据集说明",
        "",
        "## 1. 标签来源",
        "",
        "- 原候选文件夹保存了排除明显缺陷后的全部待复核二维图。",
        "- 人工从 `*_copy` 文件夹中删除模糊缺陷后，copy 中剩余图像定义为模糊正常。",
        "- 原候选文件夹有、copy 中没有的图像定义为模糊缺陷。",
        "- 构建过程不修改原候选文件夹和 copy 文件夹。",
        "",
        "## 2. 数据规模",
        "",
        f"- 模糊正常：{normal_total} 个",
        f"- 模糊缺陷：{defect_total} 个",
        f"- 合计：{normal_total + defect_total} 个",
        f"- 正常与缺陷数量比：{ratio:.2f}:1",
        f"- 因二维图或 RAW 不是唯一对应而跳过：正常 {skipped_normal} 张、缺陷 {skipped_defect} 张",
        "",
        "## 3. 每块板统计",
        "",
        "| 板号/批次 | 模糊正常 | 模糊缺陷 | 跳过正常 | 跳过缺陷 | 正常尺寸 | 缺陷尺寸 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        normal_shapes = json.loads(row["FuzzyNormalShapes"])
        defect_shapes = json.loads(row["FuzzyDefectShapes"])
        normal_shape_text = "；".join(
            f"{shape}：{count}" for shape, count in sorted(normal_shapes.items())
        ) or "无"
        defect_shape_text = "；".join(
            f"{shape}：{count}" for shape, count in sorted(defect_shapes.items())
        ) or "无"
        lines.append(
            f"| {row['Board']} | {row['IncludedFuzzyNormal']} | "
            f"{row['IncludedFuzzyDefect']} | "
            f"{row['SkippedAmbiguousNormalFiles']} | "
            f"{row['SkippedAmbiguousDefectFiles']} | "
            f"{normal_shape_text} | {defect_shape_text} |"
        )
    lines.extend(
        [
            "",
            "## 4. 目录结构",
            "",
            "```text",
            "fuzzy_binary_dataset/",
            "  normal_samples_by_board/",
            "    <板目录>/drill/",
            "    <板目录>/drill_feature/",
            "    <板目录>/manifest.csv",
            "  fuzzy_defects_by_board/",
            "    <板目录>/drill/",
            "    <板目录>/drill_feature/",
            "    <板目录>/manifest.csv",
            "  board_summary.csv",
            "  selection_audit.csv",
            "```",
            "",
            "## 5. 完整性规则",
            "",
            "- 每个纳入样本必须有且只有一张二维图和一个对应 RAW。",
            "- RAW 文件大小必须与文件名记录的三维尺寸一致。",
            "- 模糊正常与模糊缺陷按板号和孔序号无交集。",
            "- 模糊数据集与明显数据集按板号和孔序号无交集。",
            "- 所有复制文件均使用 SHA-256 校验。",
            "- 非唯一映射样本保留在源目录中，并记录于 selection_audit.csv，但不进入本数据集。",
            "",
        ]
    )
    return "\n".join(lines)


def verify_output(
    temp_root: Path,
    normal_rows: list[dict],
    defect_rows: list[dict],
    samples: list[Sample],
) -> None:
    pairs = [(sample.board, sample.sequence) for sample in samples]
    if len(set(pairs)) != len(pairs):
        raise RuntimeError("Duplicate board/sequence pair in fuzzy dataset")
    normal_pairs = {
        (row["Board"], int(row["Sequence"])) for row in normal_rows
    }
    defect_pairs = {
        (row["Board"], int(row["Sequence"])) for row in defect_rows
    }
    if normal_pairs & defect_pairs:
        raise RuntimeError("Normal and defect manifests overlap")
    checks = (
        (NORMAL_FOLDER, normal_rows),
        (DEFECT_FOLDER, defect_rows),
    )
    for folder, rows in checks:
        class_root = temp_root / folder
        raw_count = len(list(class_root.glob("*/drill/*.raw")))
        feature_count = len(list(class_root.glob("*/drill_feature/*.jpg")))
        if raw_count != len(rows) or feature_count != len(rows):
            raise RuntimeError(
                f"Output count mismatch for {folder}: "
                f"RAW={raw_count}, feature={feature_count}, manifest={len(rows)}"
            )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / OUTPUT_FOLDER
    temp_output = root / f"{OUTPUT_FOLDER}.__building__"
    backup_output = root / f"{OUTPUT_FOLDER}.__backup__"
    if temp_output.exists() or backup_output.exists():
        raise RuntimeError(
            f"Temporary output already exists: {temp_output} or {backup_output}"
        )
    if output.exists() and not args.replace_existing:
        raise RuntimeError(f"Output already exists: {output}")

    boards = discover_boards(root)
    clear_pairs = load_clear_pairs(root)
    samples: list[Sample] = []
    audit_rows: list[dict] = []
    summary_rows: list[dict] = []
    for board in boards:
        board_samples, board_audit, board_summary = collect_board_samples(
            board, clear_pairs
        )
        samples.extend(board_samples)
        audit_rows.extend(board_audit)
        summary_rows.append(board_summary)
        print(
            f"[{board.name}] normal={board_summary['IncludedFuzzyNormal']}, "
            f"defect={board_summary['IncludedFuzzyDefect']}, "
            f"skipped={board_summary['SkippedAmbiguousNormalFiles'] + board_summary['SkippedAmbiguousDefectFiles']}",
            flush=True,
        )

    temp_output.mkdir()
    normal_rows, defect_rows = copy_samples(
        temp_output, output, boards, samples
    )
    write_csv(temp_output / "board_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(temp_output / "selection_audit.csv", audit_rows, AUDIT_FIELDS)
    class_summary_fields = ["Board", "Category", "Count", "Shapes"]
    write_csv(
        temp_output / NORMAL_FOLDER / "board_summary.csv",
        summary_rows_for_class(summary_rows, normal=True),
        class_summary_fields,
    )
    write_csv(
        temp_output / DEFECT_FOLDER / "board_summary.csv",
        summary_rows_for_class(summary_rows, normal=False),
        class_summary_fields,
    )
    (temp_output / "README.txt").write_text(
        "Fuzzy binary drill-hole dataset.\n"
        "Files retained in each manually screened copy are fuzzy normal.\n"
        "Files removed from the copy are fuzzy defective.\n"
        "Only samples with exactly one feature image and one RAW are included.\n"
        "Every copied file is verified with SHA-256.\n"
        "See the Chinese dataset description and CSV manifests for details.\n",
        encoding="ascii",
    )
    (temp_output / "数据集说明.md").write_text(
        chinese_document(summary_rows), encoding="utf-8"
    )
    verify_output(temp_output, normal_rows, defect_rows, samples)

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
    print(f"FUZZY_NORMAL={len(normal_rows)}", flush=True)
    print(f"FUZZY_DEFECT={len(defect_rows)}", flush=True)
    print(f"TOTAL={len(normal_rows) + len(defect_rows)}", flush=True)


if __name__ == "__main__":
    main()
