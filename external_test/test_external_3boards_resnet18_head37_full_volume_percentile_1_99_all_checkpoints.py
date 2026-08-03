import argparse
import os
import random
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_operate.config import (  # noqa: E402
    DROPOUT_RATE,
    FOLD_SEEDS,
    MODEL_BEST_LAST_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    RANDOM_SEED,
    TEST_RESULT_DIR,
    TEST_THRESHOLD,
    USE_DETERMINISTIC_ALGORITHMS,
)
from data_operate.data_load_external_variable_shape import (  # noqa: E402
    RAW_DTYPE,
    build_external_samples,
    parse_raw_shape,
)
from data_operate.data_load_head37_full_volume_percentile_1_99 import (  # noqa: E402
    normalize_head_with_full_volume_bounds,
    sample_percentile_bounds,
    validate_normalization_params,
)
from model.resnet_18_34_50_3d_head_region_37 import (  # noqa: E402
    BasicBlock,
    ResNet3D,
)


MODEL_NAME = "resnet18_head37_full_volume_percentile_1_99_original_size"
EXPERIMENT_NAME = (
    "resnet18_head_region_37_5fold_ce_full_volume_percentile_1_99"
)
CHECKPOINT_FILES = {
    "best_f1": "best_resnet18_head_region_37_3d_full_volume_percentile_1_99.pth",
    "best_loss": "best_loss_resnet18_head_region_37_3d_full_volume_percentile_1_99.pth",
    "last": "last_resnet18_head_region_37_3d_full_volume_percentile_1_99.pth",
}
EXTERNAL_DATA_ROOT = PROJECT_ROOT / "datasets" / "external_test_head37"
EXTERNAL_FULL_DATA_ROOT = PROJECT_ROOT / "datasets" / "external_test"
DEFAULT_OUTPUT_DIR = (
    TEST_RESULT_DIR
    / "external_3boards_resnet18_head37_full_volume_percentile_1_99_original_size_all_checkpoints"
)

FULL_CLASS_DIRECTORY_BY_LABEL = {
    0: "drill_normal_3d",
    1: "drill_defect_3d",
}
_EXTERNAL_FULL_BOUNDS_CACHE = {}

METRIC_COLUMNS = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "AUC",
    "AP",
]

SUMMARY_METRICS = [
    ("Accuracy", "Accuracy"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1-Score", "F1"),
    ("Specificity", "Specificity"),
    ("AUC-ROC", "AUC"),
    ("AP", "AP"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Head37 ResNet18 with full-volume per-sample P1/P99 "
            "on the three external boards without spatial size unification."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=EXTERNAL_DATA_ROOT,
        help="Directory containing the already-cropped external head data.",
    )
    parser.add_argument(
        "--full-data-root",
        type=Path,
        default=EXTERNAL_FULL_DATA_ROOT,
        help="Directory containing corresponding original-size full RAW data.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_BEST_LAST_DIR,
        help="Root directory containing the five-fold model experiment folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for predictions, metrics and confusion matrices.",
    )
    parser.add_argument(
        "--boards",
        nargs="+",
        default=["1022", "1077", "1305"],
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(len(FOLD_SEEDS))),
    )
    parser.add_argument(
        "--checkpoint-kinds",
        nargs="+",
        choices=list(CHECKPOINT_FILES),
        default=list(CHECKPOINT_FILES),
        help="Checkpoint types to evaluate independently.",
    )
    parser.add_argument("--threshold", type=float, default=TEST_THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(USE_DETERMINISTIC_ALGORITHMS)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def build_model():
    return ResNet3D(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
    )


def checkpoint_path(model_root, fold_idx, checkpoint_kind):
    return (
        Path(model_root)
        / EXPERIMENT_NAME
        / f"fold_{fold_idx}"
        / CHECKPOINT_FILES[checkpoint_kind]
    )


class PercentileExternalBoardDataset(Dataset):
    """Normalize each original-size head crop using its full volume's P1/P99."""

    def __init__(self, board_dir, full_board_dir, dataset_kind, norm_params):
        self.board_dir = Path(board_dir)
        self.full_board_dir = Path(full_board_dir)
        self.dataset_kind = dataset_kind
        self.normalization_params = validate_normalization_params(norm_params)
        self.samples = build_external_samples(self.board_dir, dataset_kind)
        self.full_volume_info = []

        for sample in self.samples:
            class_directory = FULL_CLASS_DIRECTORY_BY_LABEL[sample.label]
            full_path = (
                self.full_board_dir / class_directory / sample.source_raw_file
            )
            if not full_path.is_file():
                raise FileNotFoundError(
                    "Prepared head sample has no corresponding original-size full RAW: "
                    f"{sample.raw_path} -> {full_path}"
                )

            full_shape = parse_raw_shape(full_path.name)
            expected_source_shape = "x".join(str(value) for value in full_shape)
            if sample.source_shape != expected_source_shape:
                raise ValueError(
                    f"Head/full source shape mismatch for {sample.raw_path}: "
                    f"manifest={sample.source_shape}, full={expected_source_shape}"
                )
            expected_bytes = int(np.prod(full_shape)) * np.dtype(RAW_DTYPE).itemsize
            if full_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"Full RAW byte count mismatch for {full_path}: "
                    f"got {full_path.stat().st_size}, expected {expected_bytes}"
                )

            cache_key = str(full_path.resolve()).lower()
            if cache_key not in _EXTERNAL_FULL_BOUNDS_CACHE:
                full_volume = np.fromfile(full_path, dtype=RAW_DTYPE).reshape(full_shape)
                _EXTERNAL_FULL_BOUNDS_CACHE[cache_key] = sample_percentile_bounds(
                    full_volume
                )
            lower, upper = _EXTERNAL_FULL_BOUNDS_CACHE[cache_key]
            self.full_volume_info.append(
                {
                    "path": full_path,
                    "shape": full_shape,
                    "lower": lower,
                    "upper": upper,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        full_info = self.full_volume_info[index]
        volume = np.fromfile(sample.raw_path, dtype=RAW_DTYPE).reshape(
            sample.input_shape
        )
        tensor = torch.from_numpy(volume.copy()).float().unsqueeze(0)
        tensor = normalize_head_with_full_volume_bounds(
            tensor,
            full_info["lower"],
            full_info["upper"],
        )

        metadata = {
            "sample_key": sample.sample_key,
            "board": sample.board,
            "sample_order": sample.sample_order,
            "raw_file": sample.source_raw_file,
            "picture_file": sample.source_picture_file,
            "raw_shape": sample.source_shape,
            "input_shape": "x".join(str(value) for value in sample.input_shape),
            "full_raw_file": full_info["path"].name,
            "full_input_shape": "x".join(
                str(value) for value in full_info["shape"]
            ),
            "normalization_p1": full_info["lower"],
            "normalization_p99": full_info["upper"],
            **sample.crop_metadata,
        }
        return tensor, sample.label, metadata


def single_item_collate(batch):
    if len(batch) != 1:
        raise ValueError("Variable-shape external testing requires batch_size=1")
    return batch[0]


def get_external_loader(
    board_dir,
    full_board_dir,
    dataset_kind,
    norm_params,
    num_workers=0,
):
    dataset = PercentileExternalBoardDataset(
        board_dir=board_dir,
        full_board_dir=full_board_dir,
        dataset_kind=dataset_kind,
        norm_params=norm_params,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=single_item_collate,
    )


def validate_inputs(args):
    option_problems = []
    for option_name, values in (
        ("boards", args.boards),
        ("folds", args.folds),
        ("checkpoint kinds", args.checkpoint_kinds),
    ):
        if len(values) != len(set(values)):
            option_problems.append(f"Duplicate {option_name}: {values}")
    if not 0.0 <= args.threshold <= 1.0:
        option_problems.append(f"Threshold must be within [0, 1]: {args.threshold}")
    if args.num_workers < 0:
        option_problems.append(
            f"num_workers must be non-negative: {args.num_workers}"
        )
    if option_problems:
        raise ValueError("\n".join(option_problems))

    problems = []
    for board in args.boards:
        board_dir = args.data_root / board
        if not board_dir.is_dir():
            problems.append(f"Missing prepared head-data board directory: {board_dir}")
        full_board_dir = args.full_data_root / board
        if not full_board_dir.is_dir():
            problems.append(
                f"Missing corresponding full-data board directory: {full_board_dir}"
            )

    for fold_idx in args.folds:
        if fold_idx < 0 or fold_idx >= len(FOLD_SEEDS):
            problems.append(f"Invalid fold index: {fold_idx}")
            continue
        for checkpoint_kind in args.checkpoint_kinds:
            path = checkpoint_path(args.model_root, fold_idx, checkpoint_kind)
            if not path.is_file():
                problems.append(f"Missing checkpoint: {path}")

    if problems:
        raise FileNotFoundError("\n".join(problems))


def load_model_and_norm(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no model_state_dict: {path}")
    if "norm_params" not in checkpoint:
        raise KeyError(
            f"Checkpoint has no training normalization parameters: {path}. "
            "This experiment requires full-volume per-sample P1/P99 metadata."
        )

    try:
        norm_params = validate_normalization_params(checkpoint["norm_params"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint does not normalize the head ROI with corresponding "
            f"full-volume per-sample P1/P99 bounds: {path}"
        ) from exc

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, norm_params


def calculate_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    accuracy = float((predictions == labels).mean())
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(labels, predictions, zero_division=0)

    if np.unique(labels).size == 2:
        auc_score = roc_auc_score(labels, probabilities)
        ap_score = average_precision_score(labels, probabilities)
    else:
        auc_score = float("nan")
        ap_score = float("nan")

    return {
        "Samples": int(labels.size),
        "Normal": int((labels == 0).sum()),
        "Defective": int((labels == 1).sum()),
        "Accuracy": accuracy,
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "F1": float(f1),
        "AUC": float(auc_score),
        "AP": float(ap_score),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def evaluate_fold(
    model,
    loader,
    device,
    threshold,
    checkpoint_kind,
    fold_idx,
    seed,
):
    rows = []
    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for tensor, label, metadata in tqdm(
            loader,
            desc=(
                f"{MODEL_NAME} checkpoint={checkpoint_kind} "
                f"fold={fold_idx} board={loader.dataset.board_dir.name}"
            ),
            file=sys.stdout,
            dynamic_ncols=True,
        ):
            inputs = tensor.unsqueeze(0).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs)
                probability = torch.softmax(logits, dim=1)[0, 1].item()

            prediction = int(probability >= threshold)
            rows.append(
                {
                    "Model": MODEL_NAME,
                    "CheckpointKind": checkpoint_kind,
                    "Fold": fold_idx,
                    "Seed": seed,
                    "Board": metadata["board"],
                    "SampleKey": metadata["sample_key"],
                    "SampleOrder": metadata["sample_order"],
                    "RawFile": metadata["raw_file"],
                    "PictureFile": metadata["picture_file"],
                    "RawShape": metadata["raw_shape"],
                    "InputShape": metadata["input_shape"],
                    "FullRawFile": metadata["full_raw_file"],
                    "FullInputShape": metadata["full_input_shape"],
                    "NormalizationP1": metadata["normalization_p1"],
                    "NormalizationP99": metadata["normalization_p99"],
                    "TrueLabel": int(label),
                    "DefectiveProbability": probability,
                    "PredictedLabel": prediction,
                    "Correct": int(prediction == label),
                    "ForegroundLeft": metadata["foreground_left"],
                    "HeadEnd": metadata["head_end"],
                    "CropStart": metadata["crop_start"],
                    "CropEndExclusive": metadata["crop_end_exclusive"],
                    "LeftPad": metadata["left_pad"],
                }
            )

    labels = [row["TrueLabel"] for row in rows]
    probabilities = [row["DefectiveProbability"] for row in rows]
    metrics = calculate_metrics(labels, probabilities, threshold)
    metrics.update(
        {
            "Model": MODEL_NAME,
            "CheckpointKind": checkpoint_kind,
            "Fold": fold_idx,
            "Seed": seed,
            "Board": loader.dataset.board_dir.name,
        }
    )
    return rows, metrics


def build_ensemble_predictions(predictions):
    group_columns = [
        "Model",
        "CheckpointKind",
        "Board",
        "SampleKey",
        "SampleOrder",
        "RawFile",
        "PictureFile",
        "RawShape",
        "InputShape",
        "TrueLabel",
    ]
    return (
        predictions.groupby(group_columns, as_index=False, dropna=False)[
            "DefectiveProbability"
        ]
        .mean()
        .rename(columns={"DefectiveProbability": "MeanDefectiveProbability"})
    )


def ensemble_metric_rows(ensemble, threshold):
    rows = []
    for (checkpoint_kind, board), group in ensemble.groupby(
        ["CheckpointKind", "Board"]
    ):
        metrics = calculate_metrics(
            group["TrueLabel"], group["MeanDefectiveProbability"], threshold
        )
        metrics.update(
            {
                "Model": MODEL_NAME,
                "CheckpointKind": checkpoint_kind,
                "Board": board,
                "Subset": "Board",
            }
        )
        rows.append(metrics)

    for checkpoint_kind, group in ensemble.groupby("CheckpointKind"):
        metrics = calculate_metrics(
            group["TrueLabel"], group["MeanDefectiveProbability"], threshold
        )
        metrics.update(
            {
                "Model": MODEL_NAME,
                "CheckpointKind": checkpoint_kind,
                "Board": "ALL",
                "Subset": "All boards",
            }
        )
        rows.append(metrics)

    for (checkpoint_kind, raw_shape), group in ensemble.groupby(
        ["CheckpointKind", "RawShape"]
    ):
        metrics = calculate_metrics(
            group["TrueLabel"], group["MeanDefectiveProbability"], threshold
        )
        metrics.update(
            {
                "Model": MODEL_NAME,
                "CheckpointKind": checkpoint_kind,
                "Board": "ALL",
                "Subset": f"Raw {raw_shape}",
            }
        )
        rows.append(metrics)
    return rows


def save_confusion_plot(group, threshold, output_path, title):
    labels = group["TrueLabel"].to_numpy()
    probabilities = group["MeanDefectiveProbability"].to_numpy()
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], labels=["Normal", "Defective"])
    axis.set_yticks([0, 1], labels=["Normal", "Defective"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def add_all_boards_fold_metrics(predictions, fold_metrics, threshold):
    all_board_rows = []
    group_columns = ["Model", "CheckpointKind", "Fold", "Seed"]
    for group_key, group in predictions.groupby(group_columns, sort=False):
        model_name, checkpoint_kind, fold_idx, seed = group_key
        metrics = calculate_metrics(
            group["TrueLabel"], group["DefectiveProbability"], threshold
        )
        metrics.update(
            {
                "Model": model_name,
                "CheckpointKind": checkpoint_kind,
                "Fold": fold_idx,
                "Seed": seed,
                "Board": "ALL",
            }
        )
        all_board_rows.append(metrics)
    return pd.concat(
        [fold_metrics, pd.DataFrame(all_board_rows)], ignore_index=True
    )


def format_mean_std(mean, std):
    if pd.isna(std):
        return f"{mean * 100:.2f}% \u00b1 N/A"
    return f"{mean * 100:.2f}% \u00b1 {std * 100:.2f}%"


def build_board_summary_table(fold_summary, board):
    board_summary = fold_summary[fold_summary["Board"] == board]
    available_kinds = set(board_summary["CheckpointKind"])
    checkpoint_kinds = [
        kind for kind in CHECKPOINT_FILES if kind in available_kinds
    ]
    if not checkpoint_kinds:
        raise ValueError(f"No fold summary rows found for board {board}")

    rows = []
    for display_name, metric in SUMMARY_METRICS:
        row = {"Metric": display_name}
        for checkpoint_kind in checkpoint_kinds:
            selected = board_summary[
                board_summary["CheckpointKind"] == checkpoint_kind
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one summary row for board={board}, "
                    f"checkpoint={checkpoint_kind}; got {len(selected)}"
                )
            summary_row = selected.iloc[0]
            row[checkpoint_kind] = format_mean_std(
                summary_row[f"{metric}_mean"], summary_row[f"{metric}_std"]
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=["Metric", *checkpoint_kinds])


def build_summary_tables(fold_summary, predictions):
    board_order = list(dict.fromkeys(predictions["Board"].tolist()))
    summary_tables = {}
    combined_blocks = []
    for board in [*board_order, "ALL"]:
        table = build_board_summary_table(fold_summary, board)
        summary_tables[board] = table
        block = table.copy()
        block.insert(0, "Board", board)
        combined_blocks.append(block)
    return summary_tables, pd.concat(combined_blocks, ignore_index=True)


def save_outputs(predictions, fold_metrics, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(predictions)
    fold_metrics_df = add_all_boards_fold_metrics(
        predictions_df, pd.DataFrame(fold_metrics), args.threshold
    )
    ensemble_df = build_ensemble_predictions(predictions_df)
    ensemble_df["PredictedLabel"] = (
        ensemble_df["MeanDefectiveProbability"] >= args.threshold
    ).astype(int)
    ensemble_df["Correct"] = (
        ensemble_df["PredictedLabel"] == ensemble_df["TrueLabel"]
    ).astype(int)
    ensemble_metrics_df = pd.DataFrame(
        ensemble_metric_rows(ensemble_df, args.threshold)
    )

    fold_summary_df = (
        fold_metrics_df.groupby(["Model", "CheckpointKind", "Board"])[
            METRIC_COLUMNS
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    fold_summary_df.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in fold_summary_df.columns
    ]
    summary_tables, formatted_fold_summary_df = build_summary_tables(
        fold_summary_df, predictions_df
    )

    predictions_df.to_csv(args.output_dir / "per_fold_predictions.csv", index=False)
    fold_metrics_df.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    formatted_fold_summary_df.to_csv(
        args.output_dir / "fivefold_mean_std.csv", index=False, encoding="utf-8-sig"
    )
    fold_summary_df.to_csv(
        args.output_dir / "fivefold_mean_std_raw.csv", index=False
    )
    ensemble_df.to_csv(args.output_dir / "ensemble_predictions.csv", index=False)
    ensemble_metrics_df.to_csv(
        args.output_dir / "ensemble_metrics_board_and_shape.csv", index=False
    )

    summary_dir = args.output_dir / "fivefold_summary_by_board"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for board, table in summary_tables.items():
        file_name = "all_boards.csv" if board == "ALL" else f"board_{board}.csv"
        table.to_csv(summary_dir / file_name, index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(args.output_dir / "external_test_summary.xlsx") as writer:
            for board, table in summary_tables.items():
                sheet_name = "All boards" if board == "ALL" else f"Board {board}"
                table.to_excel(writer, sheet_name=sheet_name, index=False)
            fold_metrics_df.to_excel(writer, sheet_name="Fold metrics", index=False)
            fold_summary_df.to_excel(
                writer, sheet_name="Mean std raw", index=False
            )
            ensemble_metrics_df.to_excel(
                writer, sheet_name="Ensemble metrics", index=False
            )
            ensemble_df.to_excel(writer, sheet_name="Ensemble predictions", index=False)
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Excel export skipped: {exc}")

    for checkpoint_kind, checkpoint_group in ensemble_df.groupby("CheckpointKind"):
        for board, board_group in checkpoint_group.groupby("Board"):
            save_confusion_plot(
                board_group,
                args.threshold,
                args.output_dir
                / checkpoint_kind
                / f"board_{board}_confusion.png",
                f"{MODEL_NAME} {checkpoint_kind} ensemble - board {board}",
            )
        save_confusion_plot(
            checkpoint_group,
            args.threshold,
            args.output_dir / checkpoint_kind / "all_boards_confusion.png",
            f"{MODEL_NAME} {checkpoint_kind} ensemble - all boards",
        )
    return ensemble_metrics_df


def print_summary(ensemble_metrics):
    rows = ensemble_metrics[ensemble_metrics["Subset"].isin(["Board", "All boards"])]
    columns = [
        "Model",
        "CheckpointKind",
        "Board",
        "Samples",
        "Normal",
        "Defective",
        "Accuracy",
        "Recall",
        "Specificity",
        "F1",
        "AUC",
    ]
    print("\nFive-fold probability ensemble results:")
    print(rows[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.full_data_root = args.full_data_root.resolve()
    args.model_root = args.model_root.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_inputs(args)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Prepared head data: {args.data_root}")
    print(f"Corresponding original-size full data: {args.full_data_root}")
    print(f"Model root: {args.model_root}")
    print(f"Output: {args.output_dir}")
    print(f"Boards: {args.boards}")
    print(f"Folds: {args.folds}")
    print(f"Checkpoint types: {args.checkpoint_kinds}")
    print(f"Threshold: {args.threshold}")
    print(
        "Model: Head37 ResNet18 normalized with per-sample P1/P99 bounds "
        "from the corresponding original-size full volume."
    )
    print("Checkpoint policy: best F1, best loss and last epoch are evaluated separately.")
    print("Testing reads existing head crops and full RAWs; no cropping occurs.")
    print("No spatial resampling or size unification occurs during testing.")
    print("P1/P99 is calculated from the full RAW, then applied to its head crop.")
    print("Testing uses batch_size=1.")

    all_predictions = []
    all_fold_metrics = []

    for checkpoint_kind in args.checkpoint_kinds:
        for fold_idx in args.folds:
            seed = FOLD_SEEDS[fold_idx]
            set_seed(seed)
            path = checkpoint_path(args.model_root, fold_idx, checkpoint_kind)
            print(
                f"\nLoading {MODEL_NAME}, checkpoint={checkpoint_kind}, "
                f"fold={fold_idx}: {path}"
            )
            model, norm_params = load_model_and_norm(path, device)
            print(f"Checkpoint normalization: {norm_params}")

            for board in args.boards:
                loader = get_external_loader(
                    board_dir=args.data_root / board,
                    full_board_dir=args.full_data_root / board,
                    dataset_kind="head",
                    norm_params=norm_params,
                    num_workers=args.num_workers,
                )
                predictions, metrics = evaluate_fold(
                    model=model,
                    loader=loader,
                    device=device,
                    threshold=args.threshold,
                    checkpoint_kind=checkpoint_kind,
                    fold_idx=fold_idx,
                    seed=seed,
                )
                all_predictions.extend(predictions)
                all_fold_metrics.append(metrics)
                print(
                    f"{MODEL_NAME} checkpoint={checkpoint_kind} "
                    f"fold={fold_idx} board={board}: "
                    f"Acc={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}, "
                    f"Recall={metrics['Recall']:.4f}, "
                    f"Specificity={metrics['Specificity']:.4f}"
                )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ensemble_metrics = save_outputs(all_predictions, all_fold_metrics, args)
    print_summary(ensemble_metrics)
    print(f"\nExternal test results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
