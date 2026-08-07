from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_operate.data_load_resnet18_multiboard_direction_unified import (  # noqa: E402
    CLASS_TO_INDEX,
    ShapeBatchSampler,
    TRAIN_DATASET_NAME,
    get_augmentation_params,
    get_fold_loaders,
    get_direction_standardization_params,
    get_normalization_params,
    resolve_dataset_root,
    summarize_dataset,
)
from model.resnet18_3d_multiboard_direction_unified import (  # noqa: E402
    architecture_metadata,
    resnet18_3d,
)


EXPERIMENT_NAME = "resnet18_multiboard_direction_unified_22boards_5fold"
DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT / "datasets" / "resnet18_multiboard_direction_unified_5fold"
)
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "model_best_last" / EXPERIMENT_NAME
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "train_val_result" / EXPERIMENT_NAME

N_SPLITS = 5
FOLD_SEEDS = (42, 123, 2026, 3407, 777)
EPOCHS = 50
BATCH_SIZE = 4
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
LR_ETA_MIN = 1e-7
DROPOUT_RATE = 0.5
DECISION_THRESHOLD = 0.5

CHECKPOINT_FILES = {
    "best_f1": "best_f1_resnet18_3d_direction_unified.pth",
    "best_loss": "best_loss_resnet18_3d_direction_unified.pth",
    "last": "last_resnet18_3d_direction_unified.pth",
}
CHECKPOINT_ORDER = ("best_f1", "best_loss", "last")
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "specificity": "Specificity",
    "auc": "AUC-ROC",
    "ap": "AP",
}


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    predictions: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the full-volume 3D ResNet18 with every drill direction "
            "standardized to the high-D side, using the baseline's exact "
            "22-board five-fold assignments."
        )
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            f"Path to {TRAIN_DATASET_NAME}. If omitted, use the environment, "
            "project datasets, or mounted-drive discovery."
        ),
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="all, a single fold such as 0, or a comma list such as 0,2,4",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--eta-min", type=float, default=LR_ETA_MIN)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-file-check", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def parse_folds(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(N_SPLITS))
    try:
        folds = sorted({int(part.strip()) for part in value.split(",")})
    except ValueError as exc:
        raise ValueError(f"Invalid --folds value: {value}") from exc
    if not folds or any(fold < 0 or fold >= N_SPLITS for fold in folds):
        raise ValueError(f"Folds must be in 0..{N_SPLITS - 1}: {folds}")
    return folds


def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(value: str) -> torch.device:
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def safe_ap(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if not np.any(labels == 1):
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def calculate_metrics(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    average_loss: float,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "auc": safe_auc(y_true, y_prob),
        "ap": safe_ap(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sample_count": int(y_true.size),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    epoch_index: int,
    num_epochs: int,
) -> dict[str, float]:
    model.train()
    batch_sampler = loader.batch_sampler
    if isinstance(batch_sampler, ShapeBatchSampler):
        batch_sampler.set_epoch(epoch_index)

    loss_sum = 0.0
    sample_count = 0
    labels_all: list[int] = []
    probabilities_all: list[float] = []

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch_index + 1:02d}/{num_epochs:02d} train",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for batch in progress:
        inputs = batch["volume"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_count = targets.size(0)
        loss_sum += float(loss.item()) * batch_count
        sample_count += batch_count
        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        labels_all.extend(targets.detach().cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())
        progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}")

    return calculate_metrics(
        labels_all,
        probabilities_all,
        average_loss=loss_sum / sample_count,
    )


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool,
    epoch_index: int,
    num_epochs: int,
) -> EvaluationResult:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    prediction_rows: list[dict[str, object]] = []

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch_index + 1:02d}/{num_epochs:02d} val  ",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    with torch.no_grad():
        for batch in progress:
            inputs = batch["volume"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs)
                sample_losses = nn.functional.cross_entropy(
                    logits, targets, reduction="none"
                )

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predicted = (probabilities >= DECISION_THRESHOLD).long()
            losses_cpu = sample_losses.cpu().tolist()
            labels_cpu = targets.cpu().tolist()
            probabilities_cpu = probabilities.cpu().tolist()
            predictions_cpu = predicted.cpu().tolist()

            batch_count = len(labels_cpu)
            loss_sum += float(sum(losses_cpu))
            sample_count += batch_count
            labels_all.extend(labels_cpu)
            probabilities_all.extend(probabilities_cpu)

            for index in range(batch_count):
                prediction_rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "board": batch["board"][index],
                        "sequence": batch["sequence"][index],
                        "difficulty": batch["difficulty"][index],
                        "raw_shape_whd": batch["raw_shape_whd"][index],
                        "input_shape_dhw": batch["input_shape_dhw"][index],
                        "raw_path": batch["raw_path"][index],
                        "b_head_up": bool(batch["b_head_up"][index]),
                        "direction_flipped": bool(batch["direction_flipped"][index]),
                        "direction_standardized": bool(
                            batch["direction_standardized"][index]
                        ),
                        "standardized_head_side": batch[
                            "standardized_head_side"
                        ][index],
                        "label": int(labels_cpu[index]),
                        "prediction": int(predictions_cpu[index]),
                        "probability_defective": float(probabilities_cpu[index]),
                        "probability_normal": float(1.0 - probabilities_cpu[index]),
                        "sample_loss": float(losses_cpu[index]),
                    }
                )
            progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}")

    metrics = calculate_metrics(
        labels_all,
        probabilities_all,
        average_loss=loss_sum / sample_count,
    )
    return EvaluationResult(metrics=metrics, predictions=prediction_rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler,
    fold: int,
    seed: int,
    epoch: int,
    checkpoint_type: str,
    evaluation: EvaluationResult,
    train_metrics: dict[str, float],
    train_summary: dict[str, object],
    val_summary: dict[str, object],
    run_config: dict[str, object],
    split_config: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment": EXPERIMENT_NAME,
        "checkpoint_type": checkpoint_type,
        "selection_metric": checkpoint_type,
        "fold": fold,
        "fold_seed": seed,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_metadata": architecture_metadata(),
        "normalization_params": get_normalization_params(),
        "norm_params": get_normalization_params(),
        "augmentation_params": get_augmentation_params(),
        "direction_standardization": get_direction_standardization_params(),
        "class_to_index": dict(CLASS_TO_INDEX),
        "decision_threshold": DECISION_THRESHOLD,
        "loss_function": "CrossEntropyLoss_unweighted",
        "validation_metrics": evaluation.metrics,
        "train_metrics": train_metrics,
        "train_dataset_summary": train_summary,
        "validation_dataset_summary": val_summary,
        "run_config": run_config,
        "split_config": split_config,
    }


def save_metrics_csv(path: Path, metrics: dict[str, float]) -> None:
    rows = []
    for key in ("loss", *METRIC_DISPLAY.keys(), "tn", "fp", "fn", "tp", "sample_count"):
        rows.append({"metric": key, "value": metrics[key]})
    write_csv(path, rows)


def save_evaluation_artifacts(
    result: EvaluationResult,
    result_dir: Path,
    checkpoint_type: str,
) -> None:
    rows = sorted(result.predictions, key=lambda row: str(row["sample_id"]))
    write_csv(result_dir / f"validation_predictions_{checkpoint_type}.csv", rows)
    save_metrics_csv(
        result_dir / f"validation_metrics_{checkpoint_type}.csv", result.metrics
    )

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [row["probability_defective"] for row in rows], dtype=np.float64
    )

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Defective"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Validation confusion matrix: {checkpoint_type}")
    fig.tight_layout()
    fig.savefig(
        result_dir / f"confusion_matrix_{checkpoint_type}.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    if np.unique(labels).size == 2:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {result.metrics['auc']:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.set_title(f"Validation ROC: {checkpoint_type}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            result_dir / f"roc_{checkpoint_type}.png",
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(recall, precision, lw=2, label=f"AP = {result.metrics['ap']:.4f}")
        ax.set(xlabel="Recall", ylabel="Precision")
        ax.set_title(f"Validation precision-recall: {checkpoint_type}")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            result_dir / f"pr_{checkpoint_type}.png",
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_history(history: list[dict[str, object]], result_dir: Path) -> None:
    write_csv(result_dir / "training_history.csv", history)
    epochs = [int(row["epoch"]) for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(
        epochs, [row["train_accuracy"] for row in history], label="Train"
    )
    axes[0, 1].plot(
        epochs, [row["val_accuracy"] for row in history], label="Validation"
    )
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["train_f1"] for row in history], label="Train")
    axes[1, 0].plot(epochs, [row["val_f1"] for row in history], label="Validation")
    axes[1, 0].set_title("F1-Score")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, [row["learning_rate"] for row in history])
    axes[1, 1].set_title("Learning rate")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_dir / "training_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def dataset_integrity_check(train_loader, val_loader, fold: int) -> None:
    train_samples = train_loader.dataset.samples
    val_samples = val_loader.dataset.samples
    train_ids = {sample.sample_id for sample in train_samples}
    val_ids = {sample.sample_id for sample in val_samples}
    if train_ids & val_ids:
        raise ValueError(f"Fold {fold}: train/validation sample leakage detected")
    if any(sample.validation_fold == fold for sample in train_samples):
        raise ValueError(f"Fold {fold}: training manifest contains validation samples")
    if any(sample.validation_fold != fold for sample in val_samples):
        raise ValueError(f"Fold {fold}: validation manifest contains wrong fold samples")
    if len({sample.board for sample in train_samples}) != 22:
        raise ValueError(f"Fold {fold}: training set does not contain all 22 boards")
    if len({sample.board for sample in val_samples}) != 22:
        raise ValueError(f"Fold {fold}: validation set does not contain all 22 boards")
    if any(
        sample.standardized_head_side != "high_depth_index"
        for sample in (*train_samples, *val_samples)
    ):
        raise ValueError(f"Fold {fold}: direction standardization metadata mismatch")


def make_history_row(
    epoch: int,
    learning_rate: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {"epoch": epoch, "learning_rate": learning_rate}
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key in ("loss", *METRIC_DISPLAY.keys()):
            row[f"{prefix}_{key}"] = metrics[key]
    return row


def load_split_config(split_root: Path) -> dict[str, object]:
    path = split_root / "split_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run "
            "data_operate/make_resnet18_multiboard_direction_unified_5fold.py first."
        )
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("internal_test_set") is not False:
        raise ValueError("This experiment requires train/validation folds without test sets")
    if int(config.get("n_splits", -1)) != N_SPLITS:
        raise ValueError(f"Expected {N_SPLITS} folds, got {config.get('n_splits')}")
    if config.get("train_dataset_name") != TRAIN_DATASET_NAME:
        raise ValueError(
            f"Split dataset mismatch: expected {TRAIN_DATASET_NAME}, "
            f"got {config.get('train_dataset_name')}"
        )
    expected_assignment = (
        "within_board_label_full_volume_shape_balanced_direction_unified"
    )
    if config.get("assignment") != expected_assignment:
        raise ValueError("Split does not contain the independent direction-unified assignment")
    direction = config.get("direction_standardization", {})
    if (
        config.get("direction_standardized") is not True
        or direction.get("target_head_side") != "high_depth_index"
        or direction.get("bHeadUp_true") != "flip_D"
    ):
        raise ValueError("Split direction-standardization metadata is invalid")
    expected_counts = {
        "sample_count": 4364,
        "normal_count": 3181,
        "defective_count": 1183,
        "board_count": 22,
    }
    for key, expected in expected_counts.items():
        if int(config.get(key, -1)) != expected:
            raise ValueError(
                f"Split config {key} mismatch: expected {expected}, "
                f"got {config.get(key)}"
            )
    return config


def check_overwrite(model_dir: Path, allow_overwrite: bool) -> None:
    existing = [model_dir / file_name for file_name in CHECKPOINT_FILES.values()]
    existing = [path for path in existing if path.exists()]
    if existing and not allow_overwrite:
        raise FileExistsError(
            "Checkpoints already exist. Use --allow-overwrite only when rerunning "
            f"this fold intentionally: {existing}"
        )


def run_fold(
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    split_config: dict[str, object],
) -> dict[str, EvaluationResult]:
    seed = FOLD_SEEDS[fold]
    set_seed(seed, args.deterministic)
    model_dir = args.model_root / f"fold_{fold}"
    result_dir = args.result_root / f"fold_{fold}"
    check_overwrite(model_dir, args.allow_overwrite)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_fold_loaders(
        args.split_root,
        fold,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        verify_files=not args.skip_file_check,
    )
    dataset_integrity_check(train_loader, val_loader, fold)
    train_summary = summarize_dataset(train_loader.dataset)
    val_summary = summarize_dataset(val_loader.dataset)

    model = resnet18_3d(num_classes=2, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min,
    )
    amp_enabled = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    run_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "eta_min": args.eta_min,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropyLoss_unweighted",
        "amp_enabled": amp_enabled,
        "deterministic_algorithms": args.deterministic,
        "device": str(device),
        "dataset_root_runtime": str(args.dataset_root),
        "fold": fold,
        "fold_seed": seed,
    }
    save_json(result_dir / "run_config.json", run_config)
    save_json(
        result_dir / "dataset_summary.json",
        {"train": train_summary, "validation": val_summary},
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("\n" + "=" * 100)
    print(f"Fold {fold} | seed={seed} | device={device} | AMP={amp_enabled}")
    print(f"Model parameters: {parameter_count:,}")
    print(f"Train: {train_summary}")
    print(f"Val:   {val_summary}")
    print("Loss: unweighted CrossEntropyLoss | primary checkpoint: best_loss")
    print("=" * 100)

    history: list[dict[str, object]] = []
    saved_results: dict[str, EvaluationResult] = {}
    best_loss = float("inf")
    best_f1 = -1.0
    best_f1_tie_loss = float("inf")

    for epoch_index in range(args.epochs):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            epoch_index,
            args.epochs,
        )
        val_result = evaluate(
            model,
            val_loader,
            device,
            amp_enabled,
            epoch_index,
            args.epochs,
        )
        scheduler.step()
        val_metrics = val_result.metrics
        history.append(
            make_history_row(
                epoch_index + 1,
                learning_rate,
                train_metrics,
                val_metrics,
            )
        )

        common_payload_args = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "fold": fold,
            "seed": seed,
            "epoch": epoch_index + 1,
            "evaluation": val_result,
            "train_metrics": train_metrics,
            "train_summary": train_summary,
            "val_summary": val_summary,
            "run_config": run_config,
            "split_config": split_config,
        }

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            saved_results["best_loss"] = val_result
            payload = checkpoint_payload(
                checkpoint_type="best_loss", **common_payload_args
            )
            atomic_torch_save(payload, model_dir / CHECKPOINT_FILES["best_loss"])
            save_evaluation_artifacts(val_result, result_dir, "best_loss")
            print(
                f"Fold {fold}: new best_loss at epoch {epoch_index + 1}: "
                f"loss={best_loss:.6f}, F1={val_metrics['f1']:.4f}"
            )

        f1_improved = val_metrics["f1"] > best_f1 + 1e-12
        f1_tie_better_loss = (
            math.isclose(val_metrics["f1"], best_f1, abs_tol=1e-12)
            and val_metrics["loss"] < best_f1_tie_loss
        )
        if f1_improved or f1_tie_better_loss:
            best_f1 = val_metrics["f1"]
            best_f1_tie_loss = val_metrics["loss"]
            saved_results["best_f1"] = val_result
            payload = checkpoint_payload(
                checkpoint_type="best_f1", **common_payload_args
            )
            atomic_torch_save(payload, model_dir / CHECKPOINT_FILES["best_f1"])
            save_evaluation_artifacts(val_result, result_dir, "best_f1")

        print(
            f"Fold {fold} epoch {epoch_index + 1:02d}: "
            f"train loss={train_metrics['loss']:.4f}, F1={train_metrics['f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, "
            f"P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, AUC={val_metrics['auc']:.4f}"
        )

    saved_results["last"] = val_result
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        fold=fold,
        seed=seed,
        epoch=args.epochs,
        checkpoint_type="last",
        evaluation=val_result,
        train_metrics=train_metrics,
        train_summary=train_summary,
        val_summary=val_summary,
        run_config=run_config,
        split_config=split_config,
    )
    atomic_torch_save(payload, model_dir / CHECKPOINT_FILES["last"])
    save_evaluation_artifacts(val_result, result_dir, "last")
    save_history(history, result_dir)

    fold_metric_rows = []
    for checkpoint_type in CHECKPOINT_ORDER:
        row: dict[str, object] = {
            "fold": fold,
            "checkpoint": checkpoint_type,
        }
        row.update(saved_results[checkpoint_type].metrics)
        fold_metric_rows.append(row)
    write_csv(result_dir / "checkpoint_metrics.csv", fold_metric_rows)
    return saved_results


def format_percent(mean: float, std: float) -> str:
    return f"{mean * 100:.2f}% \u00b1 {std * 100:.2f}%"
    return f"{mean * 100:.2f}% ± {std * 100:.2f}%"


def save_grouped_oof_metrics(
    path: Path,
    predictions: list[dict[str, object]],
    group_fields: tuple[str, ...],
) -> None:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        key = tuple(str(row[field]) for field in group_fields)
        groups[key].append(row)
    output_rows: list[dict[str, object]] = []
    for key, rows in sorted(groups.items()):
        metrics = calculate_metrics(
            [int(row["label"]) for row in rows],
            [float(row["probability_defective"]) for row in rows],
            average_loss=float(
                np.mean([float(row["sample_loss"]) for row in rows])
            ),
        )
        output_rows.append({**dict(zip(group_fields, key)), **metrics})
    write_csv(path, output_rows)


def try_write_excel(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import pandas as pd

        pd.DataFrame(rows).to_excel(path, index=False)
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Excel export skipped ({exc}); CSV output is complete.")


def save_cross_validation_summary(
    fold_results: dict[int, dict[str, EvaluationResult]],
    result_root: Path,
    completed_folds: list[int],
) -> None:
    raw_rows: list[dict[str, object]] = []
    for fold in completed_folds:
        for checkpoint_type in CHECKPOINT_ORDER:
            row: dict[str, object] = {
                "fold": fold,
                "checkpoint": checkpoint_type,
            }
            row.update(fold_results[fold][checkpoint_type].metrics)
            raw_rows.append(row)
    write_csv(result_root / "fivefold_checkpoint_metrics.csv", raw_rows)
    try_write_excel(result_root / "fivefold_checkpoint_metrics.xlsx", raw_rows)

    formatted_rows: list[dict[str, object]] = []
    for metric_key, metric_name in METRIC_DISPLAY.items():
        row: dict[str, object] = {"Metric": metric_name}
        for checkpoint_type in CHECKPOINT_ORDER:
            values = np.asarray(
                [
                    fold_results[fold][checkpoint_type].metrics[metric_key]
                    for fold in completed_folds
                ],
                dtype=np.float64,
            )
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
            row[checkpoint_type] = format_percent(mean, std)
        formatted_rows.append(row)
    write_csv(result_root / "fivefold_mean_std.csv", formatted_rows)
    try_write_excel(result_root / "fivefold_mean_std.xlsx", formatted_rows)

    oof_rows: list[dict[str, object]] = []
    for fold in completed_folds:
        for prediction in fold_results[fold]["best_loss"].predictions:
            oof_rows.append({"fold": fold, **prediction})
    ids = [str(row["sample_id"]) for row in oof_rows]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate best-loss OOF predictions: {duplicates[:5]}")
    oof_rows.sort(key=lambda row: str(row["sample_id"]))
    write_csv(result_root / "oof_predictions_best_loss.csv", oof_rows)

    oof_metrics = calculate_metrics(
        [int(row["label"]) for row in oof_rows],
        [float(row["probability_defective"]) for row in oof_rows],
        average_loss=float(np.mean([float(row["sample_loss"]) for row in oof_rows])),
    )
    save_metrics_csv(result_root / "oof_metrics_best_loss.csv", oof_metrics)
    save_json(result_root / "oof_metrics_best_loss.json", oof_metrics)
    save_grouped_oof_metrics(
        result_root / "oof_metrics_best_loss_by_board.csv",
        oof_rows,
        ("board",),
    )
    save_grouped_oof_metrics(
        result_root / "oof_metrics_best_loss_by_board_shape.csv",
        oof_rows,
        ("board", "raw_shape_whd"),
    )

    if completed_folds == list(range(N_SPLITS)) and len(oof_rows) != 4364:
        raise ValueError(
            f"Complete 5-fold run must contain 4364 OOF predictions, "
            f"got {len(oof_rows)}"
        )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    args.split_root = args.split_root.resolve()
    args.model_root = args.model_root.resolve()
    args.result_root = args.result_root.resolve()
    folds = parse_folds(args.folds)
    device = choose_device(args.device)
    split_config = load_split_config(args.split_root)
    args.dataset_root = resolve_dataset_root(
        explicit_root=args.dataset_root,
        stored_root=split_config.get("train_dataset_root_at_creation"),
        expected_name=TRAIN_DATASET_NAME,
    )
    split_config = {
        **split_config,
        "dataset_root_runtime": str(args.dataset_root),
        "raw_paths_resolved_from_relative_manifest_paths": True,
    }

    for fold in folds:
        check_overwrite(args.model_root / f"fold_{fold}", args.allow_overwrite)

    args.model_root.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.result_root / "experiment_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "folds": folds,
            "fold_seeds": list(FOLD_SEEDS),
            "model": architecture_metadata(),
            "normalization": get_normalization_params(),
            "augmentation": get_augmentation_params(),
            "direction_standardization": get_direction_standardization_params(),
            "primary_checkpoint": "best_loss",
            "internal_test_set": False,
            "external_test_used_for_training_or_selection": False,
            "dataset_root_runtime": str(args.dataset_root),
            "split_config": split_config,
        },
    )

    print("=" * 100)
    print("Full-volume direction-unified 3D ResNet18: 22-board 5-fold CV")
    print(f"Folds: {folds}")
    print(f"Dataset root (read only): {args.dataset_root}")
    print(f"Split root:  {args.split_root}")
    print(f"Model root:  {args.model_root}")
    print(f"Result root: {args.result_root}")
    print("Train/validation = 80%/20%; locked 3-board test is not used here.")
    print("Direction: bHeadUp=true flips D; every drill head is at high D index.")
    print("=" * 100)

    start_time = time.time()
    fold_results: dict[int, dict[str, EvaluationResult]] = {}
    for fold in folds:
        fold_results[fold] = run_fold(fold, args, device, split_config)
        save_cross_validation_summary(fold_results, args.result_root, sorted(fold_results))

    elapsed_minutes = (time.time() - start_time) / 60.0
    print("\n" + "=" * 100)
    print(f"Completed folds: {folds}")
    print(f"Elapsed: {elapsed_minutes:.1f} minutes")
    print(f"Primary results: {args.result_root / 'fivefold_mean_std.csv'}")
    print(f"Best-loss OOF: {args.result_root / 'oof_metrics_best_loss.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
