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
    get_external_loader,
)
from model.resnet_18_34_50_3d import BasicBlock, ResNet3D  # noqa: E402


MODEL_NAME = "resnet18_full"
EXPERIMENT_NAME = "resnet18_5fold"
BOARD_NAME = "1077_two"
CHECKPOINT_FILES = {
    "best_f1": "best_resnet18_3d.pth",
    "best_loss": "best_loss_resnet18_3d.pth",
    "last": "last_resnet18_3d.pth",
}
EXTERNAL_DATA_ROOT = PROJECT_ROOT / "datasets" / "external_test"
DEFAULT_OUTPUT_DIR = (
    TEST_RESULT_DIR / "external_1077_two_resnet18_all_checkpoints"
)

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
            "Evaluate the original full-volume ResNet18 on external "
            "board 1077_two using best-F1, best-loss and last checkpoints from each fold."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=EXTERNAL_DATA_ROOT,
        help="Directory containing original external board folders.",
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


def validate_inputs(args):
    option_problems = []
    for option_name, values in (
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
    board_dir = args.data_root / BOARD_NAME
    if not board_dir.is_dir():
        problems.append(f"Missing external board directory: {board_dir}")

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
            "External data statistics must not be used as a fallback."
        )

    norm_params = checkpoint["norm_params"]
    global_min = norm_params.get("global_min")
    global_max = norm_params.get("global_max")
    if global_min is None or global_max is None:
        raise ValueError(f"Incomplete norm_params in checkpoint: {path}")

    global_min = float(global_min)
    global_max = float(global_max)
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError(f"Non-finite norm_params in checkpoint: {path}")
    if global_max <= global_min:
        raise ValueError(
            f"Invalid norm_params in checkpoint {path}: "
            f"global_min={global_min}, global_max={global_max}"
        )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, global_min, global_max


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
                    "TrueLabel": int(label),
                    "DefectiveProbability": probability,
                    "PredictedLabel": prediction,
                    "Correct": int(prediction == label),
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
    for checkpoint_kind, group in ensemble.groupby("CheckpointKind"):
        metrics = calculate_metrics(
            group["TrueLabel"], group["MeanDefectiveProbability"], threshold
        )
        metrics.update(
            {
                "Model": MODEL_NAME,
                "CheckpointKind": checkpoint_kind,
                "Board": BOARD_NAME,
                "Subset": "Board",
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


def format_mean_std(mean, std):
    if pd.isna(std):
        return f"{mean * 100:.2f}% ± N/A"
    return f"{mean * 100:.2f}% ± {std * 100:.2f}%"


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
    boards = set(predictions["Board"])
    if boards != {BOARD_NAME}:
        raise ValueError(f"Expected only board {BOARD_NAME}, got {sorted(boards)}")
    table = build_board_summary_table(fold_summary, BOARD_NAME)
    formatted = table.copy()
    formatted.insert(0, "Board", BOARD_NAME)
    return {BOARD_NAME: table}, formatted


def save_outputs(predictions, fold_metrics, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(predictions)
    fold_metrics_df = pd.DataFrame(fold_metrics)
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
        args.output_dir / "ensemble_metrics.csv", index=False
    )

    summary_dir = args.output_dir / "fivefold_summary_by_board"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for board, table in summary_tables.items():
        table.to_csv(
            summary_dir / f"board_{board}.csv", index=False, encoding="utf-8-sig"
        )

    try:
        with pd.ExcelWriter(args.output_dir / "external_test_summary.xlsx") as writer:
            for board, table in summary_tables.items():
                table.to_excel(writer, sheet_name=f"Board {board}", index=False)
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
        save_confusion_plot(
            checkpoint_group,
            args.threshold,
            args.output_dir / checkpoint_kind / f"board_{BOARD_NAME}_confusion.png",
            f"{MODEL_NAME} {checkpoint_kind} ensemble - board {BOARD_NAME}",
        )
    return ensemble_metrics_df


def print_summary(ensemble_metrics):
    rows = ensemble_metrics[ensemble_metrics["Subset"] == "Board"]
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
    args.model_root = args.model_root.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_inputs(args)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Original external data: {args.data_root}")
    print(f"Model root: {args.model_root}")
    print(f"Output: {args.output_dir}")
    print(f"Board: {BOARD_NAME} only")
    print(f"Folds: {args.folds}")
    print(f"Checkpoint types: {args.checkpoint_kinds}")
    print(f"Threshold: {args.threshold}")
    print("Model: original full-volume ResNet18.")
    print("Checkpoint policy: best F1, best loss and last epoch are evaluated separately.")
    print("Testing reads original RAW files and performs no cropping.")
    print("Variable-shape testing uses batch_size=1 and checkpoint train normalization.")

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
            model, global_min, global_max = load_model_and_norm(path, device)
            print(f"Training normalization: min={global_min}, max={global_max}")

            loader = get_external_loader(
                board_dir=args.data_root / BOARD_NAME,
                dataset_kind="full",
                global_min=global_min,
                global_max=global_max,
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
                f"fold={fold_idx} board={BOARD_NAME}: "
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
