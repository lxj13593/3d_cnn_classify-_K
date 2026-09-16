"""Five-fold Head40-533 clear-only training with fuzzy samples weighted by 0.0.

The binary task, folds, preprocessing, optimizer, checkpoint selection, and
locked test protocol are unchanged.  Fuzzy samples have zero supervised-loss
weight, so they are excluded before the model forward pass during training.
Validation always uses ordinary unweighted cross-entropy.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_operate.data_load_resnet18_head40_toml_direction_unified_multiboard_ambiguous_weight00 import (  # noqa: E402
    AMBIGUOUS_DIFFICULTY,
    CLASS_TO_INDEX,
    ShapeBatchSampler,
    TRAIN_DATASET_NAME,
    get_augmentation_params,
    get_direction_standardization_params,
    get_fold_loaders,
    get_normalization_params,
    is_ambiguous_difficulty,
    resolve_dataset_root,
    summarize_dataset,
)
from model.resnet18_3d_head40_toml_direction_unified_stem533_ambiguous_weight00_multiboard import (  # noqa: E402
    architecture_metadata,
    resnet18_3d,
)
from train_val import main_resnet18_head40_toml_direction_unified_stem533_multiboard_5fold as baseline  # noqa: E402


EXPERIMENT_NAME = (
    "resnet18_head40_toml_direction_unified_stem533_"
    "ambiguous_weight00_multiboard_22boards_5fold"
)
DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT / "datasets" / "resnet18_head40_toml_direction_unified_multiboard_5fold"
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
AMBIGUOUS_WEIGHT = 0.0

CHECKPOINT_FILES = {
    "best_f1": "best_f1_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00.pth",
    "best_loss": "best_loss_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00.pth",
    "last": "last_resnet18_head40_toml_direction_unified_stem533_ambiguous_weight00.pth",
}
CHECKPOINT_ORDER = ("best_f1", "best_loss", "last")

EvaluationResult = baseline.EvaluationResult


def ambiguity_weighting_metadata() -> dict[str, object]:
    return {
        "enabled_training_only": True,
        "label_task": "normal=0, defective=1",
        "ambiguous_definition": f"difficulty == {AMBIGUOUS_DIFFICULTY!r}",
        "clear_weight": 1.0,
        "ambiguous_weight": AMBIGUOUS_WEIGHT,
        "loss_formula": "sum(sample_weight * CE_i) / sum(sample_weight)",
        "zero_weight_execution": (
            "fuzzy samples are excluded before model forward; an all-fuzzy "
            "batch is skipped without optimizer or BatchNorm update"
        ),
        "supervision_scope": "difficulty != 'fuzzy' (clear-only)",
        "validation_loss": "ordinary unweighted CrossEntropyLoss",
        "external_test_loss": "ordinary unweighted CrossEntropyLoss",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Head40-533 on the fixed 22-board folds with fuzzy samples "
            "assigned zero training-loss weight (clear-only supervision)."
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


def check_overwrite(model_dir: Path, allow_overwrite: bool) -> None:
    existing = [model_dir / name for name in CHECKPOINT_FILES.values()]
    existing = [path for path in existing if path.exists()]
    if existing and not allow_overwrite:
        raise FileExistsError(
            "Checkpoints already exist. Use --allow-overwrite only when rerunning "
            f"this fold intentionally: {existing}"
        )


def validate_ambiguity_metadata(*loaders) -> None:
    for loader in loaders:
        for sample in loader.dataset.samples:
            is_ambiguous_difficulty(sample.difficulty)


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
    """Train one epoch with fuzzy samples assigned exactly zero CE weight.

    Zero-weight samples are removed before the forward pass.  This makes the
    weighted-loss formula well-defined for all-fuzzy batches and prevents those
    batches from changing gradients or BatchNorm running statistics.
    """
    model.train()
    batch_sampler = loader.batch_sampler
    if isinstance(batch_sampler, ShapeBatchSampler):
        batch_sampler.set_epoch(epoch_index)

    weighted_loss_numerator = 0.0
    unweighted_loss_numerator = 0.0
    effective_weight_sum = 0.0
    raw_sample_count = 0
    supervised_sample_count = 0
    ambiguous_sample_count = 0
    skipped_all_ambiguous_batch_count = 0
    optimizer_step_count = 0
    labels_all: list[int] = []
    probabilities_all: list[float] = []

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch_index + 1:02d}/{num_epochs:02d} train",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for batch in progress:
        targets_cpu = batch["target"]
        ambiguous = batch["ambiguous"].bool()
        if ambiguous.ndim != 1 or ambiguous.numel() != targets_cpu.numel():
            raise ValueError("Ambiguity flags do not align with the training targets")

        raw_batch_count = targets_cpu.size(0)
        raw_sample_count += raw_batch_count
        ambiguous_sample_count += int(ambiguous.sum().item())
        weights = (~ambiguous).to(dtype=torch.float32)
        supervised_mask = weights > 0
        if not bool(supervised_mask.any().item()):
            skipped_all_ambiguous_batch_count += 1
            if effective_weight_sum > 0:
                progress.set_postfix(
                    loss=f"{weighted_loss_numerator / effective_weight_sum:.4f}",
                    skipped=skipped_all_ambiguous_batch_count,
                )
            else:
                progress.set_postfix(loss="n/a", skipped=skipped_all_ambiguous_batch_count)
            continue

        inputs = batch["volume"][supervised_mask].to(device, non_blocking=True)
        targets = targets_cpu[supervised_mask].to(device, non_blocking=True)
        weights = weights[supervised_mask].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
            loss_each = criterion(logits, targets)
            if loss_each.ndim != 1 or loss_each.numel() != targets.numel():
                raise ValueError("Weighted CE requires one loss value per sample")
            weighted_numerator = (loss_each * weights).sum()
            weight_denominator = weights.sum()
            loss = weighted_numerator / weight_denominator

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_count += 1

        batch_count = targets.size(0)
        weighted_loss_numerator += float(weighted_numerator.detach().item())
        unweighted_loss_numerator += float(loss_each.detach().sum().item())
        effective_weight_sum += float(weight_denominator.detach().item())
        supervised_sample_count += batch_count
        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        labels_all.extend(targets.detach().cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())
        progress.set_postfix(
            loss=f"{weighted_loss_numerator / effective_weight_sum:.4f}",
            skipped=skipped_all_ambiguous_batch_count,
        )

    if effective_weight_sum <= 0 or supervised_sample_count <= 0:
        raise RuntimeError("No non-fuzzy samples were available for clear-only training")
    metrics = baseline.calculate_metrics(
        labels_all,
        probabilities_all,
        average_loss=weighted_loss_numerator / effective_weight_sum,
    )
    metrics["unweighted_ce_loss"] = unweighted_loss_numerator / supervised_sample_count
    metrics["effective_weight_sum"] = effective_weight_sum
    metrics["raw_sample_count"] = float(raw_sample_count)
    metrics["supervised_sample_count"] = float(supervised_sample_count)
    metrics["ambiguous_sample_count"] = float(ambiguous_sample_count)
    metrics["skipped_all_ambiguous_batch_count"] = float(
        skipped_all_ambiguous_batch_count
    )
    metrics["optimizer_step_count"] = float(optimizer_step_count)
    return metrics


def checkpoint_payload(**kwargs) -> dict[str, object]:
    payload = baseline.checkpoint_payload(**kwargs)
    payload["experiment"] = EXPERIMENT_NAME
    payload["model_metadata"] = architecture_metadata()
    payload["loss_function"] = "CrossEntropyLoss_per_sample_ambiguous_weight00_clear_only"
    payload["ambiguous_weighting"] = ambiguity_weighting_metadata()
    return payload


def run_fold(
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    split_config: dict[str, object],
) -> dict[str, EvaluationResult]:
    seed = FOLD_SEEDS[fold]
    baseline.set_seed(seed, args.deterministic)
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
    baseline.dataset_integrity_check(train_loader, val_loader, fold)
    validate_ambiguity_metadata(train_loader, val_loader)
    train_summary = summarize_dataset(train_loader.dataset)
    val_summary = summarize_dataset(val_loader.dataset)

    model = resnet18_3d(num_classes=2, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
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
        "loss": "per-sample CrossEntropyLoss with fuzzy weight 0.0 (clear-only)",
        "ambiguous_weighting": ambiguity_weighting_metadata(),
        "amp_enabled": amp_enabled,
        "deterministic_algorithms": args.deterministic,
        "device": str(device),
        "dataset_root_runtime": str(args.dataset_root),
        "fold": fold,
        "fold_seed": seed,
    }
    baseline.save_json(result_dir / "run_config.json", run_config)
    baseline.save_json(
        result_dir / "dataset_summary.json",
        {"train": train_summary, "validation": val_summary},
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("\n" + "=" * 100)
    print(f"Fold {fold} | seed={seed} | device={device} | AMP={amp_enabled}")
    print(f"Model parameters: {parameter_count:,}")
    print(f"Train: {train_summary}")
    print(f"Val:   {val_summary}")
    print(
        "Loss: per-sample CE; fuzzy difficulty weight=0.0 (clear-only); "
        "primary checkpoint=best_loss"
    )
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
        val_result = baseline.evaluate(
            model, val_loader, device, amp_enabled, epoch_index, args.epochs
        )
        scheduler.step()
        val_metrics = val_result.metrics
        history_row = baseline.make_history_row(
            epoch_index + 1, learning_rate, train_metrics, val_metrics
        )
        history_row.update(
            {
                "train_source_sample_count": train_metrics["raw_sample_count"],
                "train_supervised_sample_count": train_metrics[
                    "supervised_sample_count"
                ],
                "train_ignored_ambiguous_sample_count": train_metrics[
                    "ambiguous_sample_count"
                ],
                "train_skipped_all_ambiguous_batch_count": train_metrics[
                    "skipped_all_ambiguous_batch_count"
                ],
                "train_optimizer_step_count": train_metrics["optimizer_step_count"],
            }
        )
        history.append(history_row)

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
            baseline.atomic_torch_save(
                checkpoint_payload(checkpoint_type="best_loss", **common_payload_args),
                model_dir / CHECKPOINT_FILES["best_loss"],
            )
            baseline.save_evaluation_artifacts(val_result, result_dir, "best_loss")
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
            baseline.atomic_torch_save(
                checkpoint_payload(checkpoint_type="best_f1", **common_payload_args),
                model_dir / CHECKPOINT_FILES["best_f1"],
            )
            baseline.save_evaluation_artifacts(val_result, result_dir, "best_f1")

        print(
            f"Fold {fold} epoch {epoch_index + 1:02d}: "
            f"train clear-only loss={train_metrics['loss']:.4f}, "
            f"F1={train_metrics['f1']:.4f} | val loss={val_metrics['loss']:.4f}, "
            f"Acc={val_metrics['accuracy']:.4f}, P={val_metrics['precision']:.4f}, "
            f"R={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}, "
            f"AUC={val_metrics['auc']:.4f}"
        )

    saved_results["last"] = val_result
    baseline.atomic_torch_save(
        checkpoint_payload(
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
        ),
        model_dir / CHECKPOINT_FILES["last"],
    )
    baseline.save_evaluation_artifacts(val_result, result_dir, "last")
    baseline.save_history(history, result_dir)

    rows = []
    for checkpoint_type in CHECKPOINT_ORDER:
        row: dict[str, object] = {"fold": fold, "checkpoint": checkpoint_type}
        row.update(saved_results[checkpoint_type].metrics)
        rows.append(row)
    baseline.write_csv(result_dir / "checkpoint_metrics.csv", rows)
    return saved_results


def save_ambiguity_oof_summaries(
    fold_results: dict[int, dict[str, EvaluationResult]],
    result_root: Path,
) -> None:
    rows = [
        {"fold": fold, **prediction}
        for fold in sorted(fold_results)
        for prediction in fold_results[fold]["best_loss"].predictions
    ]
    baseline.save_grouped_oof_metrics(
        result_root / "oof_metrics_best_loss_by_difficulty.csv",
        rows,
        ("difficulty",),
    )
    baseline.save_grouped_oof_metrics(
        result_root / "oof_metrics_best_loss_by_difficulty_label.csv",
        rows,
        ("difficulty", "label"),
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    args.split_root = args.split_root.resolve()
    args.model_root = args.model_root.resolve()
    args.result_root = args.result_root.resolve()
    folds = parse_folds(args.folds)
    device = baseline.choose_device(args.device)
    split_config = baseline.load_split_config(args.split_root)
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
    baseline.save_json(
        args.result_root / "experiment_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "folds": folds,
            "fold_seeds": list(FOLD_SEEDS),
            "model": architecture_metadata(),
            "normalization": get_normalization_params(),
            "augmentation": get_augmentation_params(),
            "direction_standardization": get_direction_standardization_params(),
            "ambiguous_weighting": ambiguity_weighting_metadata(),
            "primary_checkpoint": "best_loss",
            "internal_test_set": False,
            "external_test_used_for_training_or_selection": False,
            "dataset_root_runtime": str(args.dataset_root),
            "split_config": split_config,
        },
    )

    print("=" * 100)
    print("Head40-533: fuzzy samples use zero training-loss weight (clear-only)")
    print(f"Folds: {folds}")
    print(f"Dataset root (read only): {args.dataset_root}")
    print(f"Split root:  {args.split_root}")
    print(f"Model root:  {args.model_root}")
    print(f"Result root: {args.result_root}")
    print("Validation and locked external testing use ordinary unweighted metrics.")
    print("=" * 100)

    start_time = time.time()
    fold_results: dict[int, dict[str, EvaluationResult]] = {}
    for fold in folds:
        fold_results[fold] = run_fold(fold, args, device, split_config)
        baseline.save_cross_validation_summary(
            fold_results, args.result_root, sorted(fold_results)
        )
        save_ambiguity_oof_summaries(fold_results, args.result_root)

    elapsed_minutes = (time.time() - start_time) / 60.0
    print("\n" + "=" * 100)
    print(f"Completed folds: {folds}")
    print(f"Elapsed: {elapsed_minutes:.1f} minutes")
    print(f"Primary results: {args.result_root / 'fivefold_mean_std.csv'}")
    print(f"Best-loss OOF: {args.result_root / 'oof_metrics_best_loss.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
