from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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

from data_operate.data_load_resnet18_head40_toml_direction_unified_multiboard import (  # noqa: E402
    TEST_DATASET_NAME,
    get_normalization_params,
    get_direction_standardization_params,
    make_loader,
    resolve_dataset_root,
    validate_normalization_params,
)
from model.resnet18_3d_head40_toml_direction_unified_stem533_layer3_e2_multiboard import (  # noqa: E402
    architecture_metadata,
    resnet18_3d,
)


EXPERIMENT_NAME = (
    "resnet18_head40_toml_direction_unified_stem533_layer3_e2_multiboard_22boards_5fold"
)
DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "resnet18_head40_toml_direction_unified_multiboard_5fold"
)
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "model_best_last" / EXPERIMENT_NAME
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test_result" / f"{EXPERIMENT_NAME}_locked_3boards"
CHECKPOINT_FILES = {
    "best_f1": "best_f1_resnet18_head40_toml_direction_unified_stem533_layer3_e2.pth",
    "best_loss": "best_loss_resnet18_head40_toml_direction_unified_stem533_layer3_e2.pth",
    "last": "last_resnet18_head40_toml_direction_unified_stem533_layer3_e2.pth",
}
CHECKPOINT_ORDER = ("best_f1", "best_loss", "last")
METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "specificity": "Specificity",
    "auc": "AUC-ROC",
    "ap": "AP",
}
N_FOLDS = 5
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all five Head40-533 Layer3 E2 HWAvg-k1-k3-DMax folds "
            "on the locked three-board test set."
        )
    )
    parser.add_argument("--test-dataset-root", type=Path, default=None)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--checkpoint-kinds",
        nargs="+",
        choices=list(CHECKPOINT_ORDER),
        default=list(CHECKPOINT_ORDER),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--skip-file-check", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def calculate_metrics(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    losses: list[float] | np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    auc = float("nan")
    if np.unique(labels).size == 2:
        auc = float(roc_auc_score(labels, probabilities))
    ap = float("nan")
    if np.any(labels == 1):
        ap = float(average_precision_score(labels, probabilities))
    return {
        "loss": float(np.mean(np.asarray(losses, dtype=np.float64))),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "auc": auc,
        "ap": ap,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sample_count": int(labels.size),
    }


def checkpoint_path(model_root: Path, fold: int, kind: str) -> Path:
    return model_root / f"fold_{fold}" / CHECKPOINT_FILES[kind]


def load_checkpoint_model(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no model_state_dict: {path}")
    if checkpoint.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(
            f"Checkpoint experiment mismatch in {path}: "
            f"{checkpoint.get('experiment')}"
        )
    model_metadata = checkpoint.get("model_metadata", {})
    if model_metadata != architecture_metadata():
        raise ValueError(f"Checkpoint Layer3 E2 architecture metadata mismatch: {path}")
    if (
        model_metadata.get("layer3_aux_variant") != "e2"
        or model_metadata.get("classifier_feature_dim") != 640
    ):
        raise ValueError(f"Checkpoint is not the required Layer3 E2 model: {path}")
    if checkpoint.get("direction_standardization") != get_direction_standardization_params():
        raise ValueError(f"Checkpoint direction metadata mismatch in {path}")
    normalization = checkpoint.get("normalization_params", checkpoint.get("norm_params"))
    validate_normalization_params(normalization)
    if checkpoint.get("class_to_index") != {"normal": 0, "defective": 1}:
        raise ValueError(f"Class mapping mismatch in {path}")
    model = resnet18_3d(num_classes=2, dropout_rate=0.5)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool,
    fold: int,
    checkpoint_kind: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    progress = tqdm(
        loader,
        desc=f"{checkpoint_kind} fold_{fold}",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    with torch.no_grad():
        for batch in progress:
            inputs = batch["volume"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs)
                losses = F.cross_entropy(logits, targets, reduction="none")
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            labels = targets.cpu().tolist()
            probs = probabilities.cpu().tolist()
            sample_losses = losses.cpu().tolist()
            for index in range(len(labels)):
                rows.append(
                    {
                        "checkpoint": checkpoint_kind,
                        "fold": fold,
                        "sample_id": batch["sample_id"][index],
                        "board": batch["board"][index],
                        "sequence": batch["sequence"][index],
                        "raw_shape_whd": batch["raw_shape_whd"][index],
                        "raw_path": batch["raw_path"][index],
                        "b_head_up": bool(batch["b_head_up"][index]),
                        "direction_flipped": bool(batch["direction_flipped"][index]),
                        "direction_standardized": bool(
                            batch["direction_standardized"][index]
                        ),
                        "standardized_head_side": batch[
                            "standardized_head_side"
                        ][index],
                        "label": int(labels[index]),
                        "probability_defective": float(probs[index]),
                        "sample_loss": float(sample_losses[index]),
                    }
                )
    return rows


def grouped_metric_rows(
    predictions: list[dict[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        groups[(str(row["checkpoint"]), int(row["fold"]), str(row["board"]))].append(row)
        groups[(str(row["checkpoint"]), int(row["fold"]), "ALL")].append(row)
    for (kind, fold, board), rows in sorted(groups.items()):
        metrics = calculate_metrics(
            [int(row["label"]) for row in rows],
            [float(row["probability_defective"]) for row in rows],
            [float(row["sample_loss"]) for row in rows],
            threshold,
        )
        output.append({"checkpoint": kind, "fold": fold, "board": board, **metrics})
    return output


def ensemble_predictions(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        groups[(str(row["checkpoint"]), str(row["sample_id"]))].append(row)
    output = []
    for (kind, sample_id), rows in sorted(groups.items()):
        if len(rows) != N_FOLDS:
            raise ValueError(f"Expected {N_FOLDS} folds for {kind}/{sample_id}")
        first = rows[0]
        if len({int(row["label"]) for row in rows}) != 1:
            raise ValueError(f"Label disagreement across folds for {sample_id}")
        output.append(
            {
                "checkpoint": kind,
                "sample_id": sample_id,
                "board": first["board"],
                "sequence": first["sequence"],
                "raw_shape_whd": first["raw_shape_whd"],
                "b_head_up": bool(first["b_head_up"]),
                "direction_flipped": bool(first["direction_flipped"]),
                "direction_standardized": bool(first["direction_standardized"]),
                "standardized_head_side": first["standardized_head_side"],
                "label": int(first["label"]),
                "probability_defective": float(
                    np.mean([float(row["probability_defective"]) for row in rows])
                ),
                "sample_loss": float(np.mean([float(row["sample_loss"]) for row in rows])),
            }
        )
    return output


def ensemble_metric_rows(
    ensemble: list[dict[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in ensemble:
        groups[(str(row["checkpoint"]), str(row["board"]))].append(row)
        groups[(str(row["checkpoint"]), "ALL")].append(row)
    output = []
    for (kind, board), rows in sorted(groups.items()):
        metrics = calculate_metrics(
            [int(row["label"]) for row in rows],
            [float(row["probability_defective"]) for row in rows],
            [float(row["sample_loss"]) for row in rows],
            threshold,
        )
        output.append({"checkpoint": kind, "board": board, **metrics})
    return output


def format_percent(mean: float, std: float) -> str:
    return f"{mean * 100:.2f}% \u00b1 {std * 100:.2f}%"


def mean_std_tables(
    fold_metrics: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    boards = sorted({str(row["board"]) for row in fold_metrics if row["board"] != "ALL"})
    boards.append("ALL")
    formatted: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    for board in boards:
        for metric_key, metric_name in METRICS.items():
            formatted_row: dict[str, object] = {"Board": board, "Metric": metric_name}
            for kind in CHECKPOINT_ORDER:
                values = np.asarray(
                    [
                        float(row[metric_key])
                        for row in fold_metrics
                        if row["board"] == board and row["checkpoint"] == kind
                    ],
                    dtype=np.float64,
                )
                if values.size == 0:
                    continue
                mean = float(np.nanmean(values))
                std = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
                formatted_row[kind] = format_percent(mean, std)
                raw.append(
                    {
                        "Board": board,
                        "Metric": metric_name,
                        "Checkpoint": kind,
                        "Mean": mean,
                        "Std": std,
                    }
                )
            formatted.append(formatted_row)
    return formatted, raw


def save_excel(path: Path, formatted: list[dict[str, object]]) -> None:
    try:
        import pandas as pd

        used_sheet_names: set[str] = set()
        with pd.ExcelWriter(path) as writer:
            frame = pd.DataFrame(formatted)
            for board, group in frame.groupby("Board", sort=False):
                if board == "ALL":
                    base_name = "ALL"
                else:
                    parts = str(board).split("_")
                    scanner = parts[0].removeprefix("VT-X750-")
                    batch = (parts[1].lstrip("0") or "0") if len(parts) > 1 else "board"
                    base_name = f"{scanner}-{batch}"
                base_name = "".join(
                    "_" if character in "[]:*?/\\" else character
                    for character in base_name
                ).strip("'")[:31] or "Sheet"
                sheet_name = base_name
                suffix = 2
                while sheet_name.lower() in used_sheet_names:
                    suffix_text = f"_{suffix}"
                    sheet_name = f"{base_name[: 31 - len(suffix_text)]}{suffix_text}"
                    suffix += 1
                used_sheet_names.add(sheet_name.lower())
                group.drop(columns=["Board"]).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                )
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Excel export skipped: {exc}")


def save_curves(
    rows: list[dict[str, object]],
    threshold: float,
    output_dir: Path,
    title: str,
) -> None:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [float(row["probability_defective"]) for row in rows], dtype=np.float64
    )
    predictions = (probabilities >= threshold).astype(np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Defective"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    if np.unique(labels).size == 2:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "roc.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ap = average_precision_score(labels, probabilities)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(recall, precision, label=f"AP = {ap:.4f}")
        ax.set(xlabel="Recall", ylabel="Precision", title=title)
        ax.legend(loc="lower left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "pr.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def validate_inputs(args: argparse.Namespace, test_manifest: Path) -> None:
    if not test_manifest.is_file():
        raise FileNotFoundError(
            f"Missing shared Head40 locked-test manifest: {test_manifest}. "
            "Pass the prepared Head40 direction-unified split with --split-root."
        )
    missing = []
    for kind in args.checkpoint_kinds:
        for fold in range(N_FOLDS):
            path = checkpoint_path(args.model_root, fold, kind)
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n  " + "\n  ".join(missing))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    set_seed(RANDOM_SEED)
    device = choose_device(args.device)
    args.split_root = args.split_root.resolve()
    args.model_root = args.model_root.resolve()
    args.output_root = args.output_root.resolve()
    test_root = resolve_dataset_root(
        args.test_dataset_root,
        expected_name=TEST_DATASET_NAME,
    )
    test_manifest = args.split_root / "locked_test_manifest.csv"
    validate_inputs(args, test_manifest)
    amp_enabled = device.type == "cuda" and not args.disable_amp

    loader = make_loader(
        test_manifest,
        dataset_root=test_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        seed=RANDOM_SEED,
        verify_files=not args.skip_file_check,
        expected_dataset_name=TEST_DATASET_NAME,
    )
    if len(loader.dataset) != 778:
        raise ValueError(f"Locked test manifest must contain 778 samples, got {len(loader.dataset)}")
    if len({sample.board for sample in loader.dataset.samples}) != 3:
        raise ValueError("Locked test manifest must contain exactly three boards")

    print("=" * 100)
    print("Head40-533 Layer3 E2 HWAvg-k1-k3-DMax: locked three-board test")
    print(f"Test dataset: {test_root}")
    print(f"Models: {args.model_root}")
    print(f"Checkpoints: {args.checkpoint_kinds}")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print("Normalization: per-sample Head40 P1/P99, identical to training")
    print("Direction: bHeadUp=true flips D; every drill head is at high D index")
    print("Inputs are pre-cropped Head40; no runtime augmentation or interpolation")
    print("=" * 100)

    predictions: list[dict[str, object]] = []
    checkpoint_meta = []
    for kind in args.checkpoint_kinds:
        for fold in range(N_FOLDS):
            path = checkpoint_path(args.model_root, fold, kind)
            model, checkpoint = load_checkpoint_model(path, device)
            predictions.extend(
                evaluate_model(model, loader, device, amp_enabled, fold, kind)
            )
            checkpoint_meta.append(
                {
                    "checkpoint": kind,
                    "fold": fold,
                    "path": str(path),
                    "epoch": checkpoint.get("epoch"),
                    "fold_seed": checkpoint.get("fold_seed"),
                }
            )

    fold_metrics = grouped_metric_rows(predictions, args.threshold)
    ensemble = ensemble_predictions(predictions)
    ensemble_metrics = ensemble_metric_rows(ensemble, args.threshold)
    formatted, raw_mean_std = mean_std_tables(fold_metrics)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "predictions_all_folds.csv", predictions)
    write_csv(args.output_root / "fold_metrics_by_board.csv", fold_metrics)
    write_csv(args.output_root / "ensemble_predictions.csv", ensemble)
    write_csv(args.output_root / "ensemble_metrics_by_board.csv", ensemble_metrics)
    write_csv(args.output_root / "fivefold_mean_std.csv", formatted)
    write_csv(args.output_root / "fivefold_mean_std_raw.csv", raw_mean_std)
    save_excel(args.output_root / "fivefold_mean_std.xlsx", formatted)
    save_json(
        args.output_root / "test_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "test_dataset_root_runtime": str(test_root),
            "test_manifest": str(test_manifest),
            "test_sample_count": len(loader.dataset),
            "test_board_count": 3,
            "checkpoint_kinds": args.checkpoint_kinds,
            "fold_count": N_FOLDS,
            "threshold": args.threshold,
            "normalization": get_normalization_params(),
            "direction_standardization": get_direction_standardization_params(),
            "augmentation": False,
            "interpolation": False,
            "runtime_cropping": False,
            "runtime_direction_flip": True,
            "input_pre_cropped_head40_by_toml": True,
            "checkpoint_metadata": checkpoint_meta,
        },
    )

    for kind in args.checkpoint_kinds:
        kind_rows = [row for row in ensemble if row["checkpoint"] == kind]
        boards = sorted({str(row["board"]) for row in kind_rows})
        for board in [*boards, "ALL"]:
            rows = kind_rows if board == "ALL" else [row for row in kind_rows if row["board"] == board]
            safe_board = "all_boards" if board == "ALL" else board
            save_curves(
                rows,
                args.threshold,
                args.output_root / "plots" / kind / safe_board,
                f"{kind} ensemble - {board}",
            )

    print(f"Saved: {args.output_root}")
    print(f"Main table: {args.output_root / 'fivefold_mean_std.xlsx'}")
    print("best_f1, best_loss and last are reported separately; best_loss remains primary.")


if __name__ == "__main__":
    main()
