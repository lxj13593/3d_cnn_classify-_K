"""Five-fold Head40-533 RACW training.

RACW keeps the original hard labels.  Epochs 1--5 use ordinary CE.  From
epoch 6 onward, only fuzzy samples receive a feature-space reliability weight.
Class centers are built exclusively from high-confidence clear training samples
in eval/no-grad mode with augmentation disabled.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the established ambiguity-aware loader.  No new data-loading pipeline
# is introduced for RACW.
from data_operate.data_load_resnet18_head40_toml_direction_unified_multiboard_ambiguous_weight05 import (  # noqa: E402
    AMBIGUOUS_DIFFICULTY,
    CLASS_TO_INDEX,
    ShapeBatchSampler,
    TRAIN_DATASET_NAME,
    get_augmentation_params,
    get_direction_standardization_params,
    get_fold_loaders,
    get_normalization_params,
    is_ambiguous_difficulty,
    make_loader,
    resolve_dataset_root,
    summarize_dataset,
)
from model.resnet18_3d_head40_toml_direction_unified_stem533_racw_multiboard import (  # noqa: E402
    architecture_metadata,
    racw_metadata,
    resnet18_3d,
)
from train_val import main_resnet18_head40_toml_direction_unified_stem533_multiboard_5fold as baseline  # noqa: E402


EXPERIMENT_NAME = (
    "resnet18_head40_toml_direction_unified_stem533_"
    "racw_multiboard_22boards_5fold"
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

CLEAR_DIFFICULTY = "clear"
FEATURE_DIMENSION = 512
MIN_CENTER_SAMPLES = 10
RACW_PROTOCOL = racw_metadata()
WARMUP_EPOCHS = int(RACW_PROTOCOL["warmup_epochs"])
RELIABILITY_P_THRESHOLD = float(RACW_PROTOCOL["reliability_probability_threshold"])
CENTER_EMA_MOMENTUM = float(RACW_PROTOCOL["center_ema_momentum"])
RELIABILITY_TEMPERATURE = float(RACW_PROTOCOL["reliability_temperature"])
FUZZY_MIN_WEIGHT = float(RACW_PROTOCOL["fuzzy_min_weight"])

CHECKPOINT_FILES = {
    "best_f1": "best_f1_resnet18_head40_toml_direction_unified_stem533_racw.pth",
    "best_loss": "best_loss_resnet18_head40_toml_direction_unified_stem533_racw.pth",
    "last": "last_resnet18_head40_toml_direction_unified_stem533_racw.pth",
}
CHECKPOINT_ORDER = ("best_f1", "best_loss", "last")

EvaluationResult = baseline.EvaluationResult
STANDARD_METRIC_KEYS = frozenset({"loss", *baseline.METRIC_DISPLAY.keys()})


@dataclass
class RACWState:
    """Fold-local detached center state; never part of model state_dict."""

    centers: torch.Tensor | None = None
    initialized_epoch: int | None = None
    last_center_epoch: int | None = None
    reliable_clear_counts: tuple[int, int] = (0, 0)

    @property
    def initialized(self) -> bool:
        return self.centers is not None

    def center_cosine_similarity(self) -> float:
        if self.centers is None:
            return float("nan")
        return float(
            functional.cosine_similarity(
                self.centers[0].unsqueeze(0),
                self.centers[1].unsqueeze(0),
                dim=1,
            ).item()
        )

    def log_values(self) -> dict[str, float | int]:
        return {
            "racw_center_initialized": int(self.initialized),
            "racw_center_initialized_epoch": (
                self.initialized_epoch if self.initialized_epoch is not None else 0
            ),
            "racw_center_last_update_epoch": (
                self.last_center_epoch if self.last_center_epoch is not None else 0
            ),
            "reliable_clear_normal_count": int(self.reliable_clear_counts[0]),
            "reliable_clear_defective_count": int(self.reliable_clear_counts[1]),
            "center_cosine_similarity": self.center_cosine_similarity(),
        }

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "initialized": self.initialized,
            "initialized_epoch": self.initialized_epoch,
            "last_center_epoch": self.last_center_epoch,
            "reliable_clear_counts": {
                "normal": int(self.reliable_clear_counts[0]),
                "defective": int(self.reliable_clear_counts[1]),
            },
            "centers": (
                self.centers.detach().cpu().clone() if self.centers is not None else None
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Head40-533 RACW on the fixed 22-board folds.  The first five "
            "epochs use ordinary CE; later epochs dynamically weight fuzzy samples."
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


def make_center_loader(
    split_root: Path,
    fold: int,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    verify_files: bool,
):
    """Return the same training fold with augmentation disabled for centers."""
    return make_loader(
        split_root / f"fold_{fold}" / "train.csv",
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        seed=seed,
        verify_files=verify_files,
        expected_dataset_name=TRAIN_DATASET_NAME,
    )


def validate_center_loader(train_loader, center_loader, fold: int) -> None:
    if center_loader.dataset.augment:
        raise ValueError(f"Fold {fold}: RACW center loader must disable augmentation")
    train_ids = {sample.sample_id for sample in train_loader.dataset.samples}
    center_ids = {sample.sample_id for sample in center_loader.dataset.samples}
    if train_ids != center_ids:
        raise ValueError(
            f"Fold {fold}: RACW center loader must contain exactly the training fold"
        )


def difficulty_mask(
    difficulties,
    expected_difficulty: str,
    device: torch.device,
) -> torch.Tensor:
    normalized = [str(value).strip().lower() for value in difficulties]
    for value in normalized:
        is_ambiguous_difficulty(value)
    return torch.tensor(
        [value == expected_difficulty for value in normalized],
        device=device,
        dtype=torch.bool,
    )


def validate_binary_targets(targets: torch.Tensor) -> None:
    if targets.ndim != 1 or not bool(torch.all((targets == 0) | (targets == 1))):
        raise ValueError("RACW requires one-dimensional binary targets encoded as 0/1")


@torch.no_grad()
def collect_clear_center_candidates(
    model: nn.Module,
    center_loader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[torch.Tensor | None], tuple[int, int]]:
    """Collect high-reliability clear-only mean embeddings from this train fold."""
    was_training = model.training
    model.eval()
    sums = torch.zeros(2, FEATURE_DIMENSION, device=device, dtype=torch.float32)
    counts = torch.zeros(2, device=device, dtype=torch.long)

    try:
        for batch in center_loader:
            inputs = batch["volume"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            validate_binary_targets(targets)
            clear = difficulty_mask(batch["difficulty"], CLEAR_DIFFICULTY, device)
            if clear.numel() != targets.numel():
                raise ValueError("Difficulty values do not align with center targets")

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits, pooled = model.forward_with_pooled_features(inputs)
            if pooled.ndim != 2 or pooled.shape[1] != FEATURE_DIMENSION:
                raise ValueError(
                    "RACW expects a pre-dropout Layer4 GAP feature of shape [N, 512]"
                )

            probabilities = torch.softmax(logits.float(), dim=1)
            predicted = probabilities.argmax(dim=1)
            target_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
            reliable = (
                clear
                & predicted.eq(targets)
                & target_probabilities.ge(RELIABILITY_P_THRESHOLD)
            )
            normalized_features = functional.normalize(pooled.detach().float(), p=2, dim=1)

            for class_index in (0, 1):
                selected = reliable & targets.eq(class_index)
                if bool(selected.any()):
                    sums[class_index] += normalized_features[selected].sum(dim=0)
                    counts[class_index] += int(selected.sum().item())
    finally:
        model.train(was_training)

    candidates: list[torch.Tensor | None] = []
    count_values = (int(counts[0].item()), int(counts[1].item()))
    for class_index, count in enumerate(count_values):
        if count == 0:
            candidates.append(None)
        else:
            candidates.append(
                functional.normalize(sums[class_index] / count, p=2, dim=0).detach()
            )
    return candidates, count_values


def initialize_centers(
    model: nn.Module,
    center_loader,
    state: RACWState,
    device: torch.device,
    amp_enabled: bool,
    epoch_number: int,
) -> None:
    candidates, counts = collect_clear_center_candidates(
        model, center_loader, device, amp_enabled
    )
    missing = [
        ("normal", counts[0]) if counts[0] < MIN_CENTER_SAMPLES else None,
        ("defective", counts[1]) if counts[1] < MIN_CENTER_SAMPLES else None,
    ]
    missing = [item for item in missing if item is not None]
    if missing:
        raise RuntimeError(
            "RACW center initialization failed after warm-up: insufficient "
            f"high-reliability clear samples {missing}. No zero or synthetic center "
            "will be used."
        )
    if candidates[0] is None or candidates[1] is None:
        raise RuntimeError("RACW center initialization produced an unexpected empty center")

    state.centers = torch.stack((candidates[0], candidates[1])).detach()
    state.initialized_epoch = epoch_number
    state.last_center_epoch = epoch_number
    state.reliable_clear_counts = counts


def refresh_centers(
    model: nn.Module,
    center_loader,
    state: RACWState,
    device: torch.device,
    amp_enabled: bool,
    epoch_number: int,
) -> None:
    """EMA-refresh detached centers; an empty class keeps its prior center."""
    if state.centers is None:
        raise RuntimeError("RACW centers must be initialized before they are refreshed")
    candidates, counts = collect_clear_center_candidates(
        model, center_loader, device, amp_enabled
    )
    updated = state.centers.detach().clone()
    for class_index, candidate in enumerate(candidates):
        if candidate is not None:
            updated[class_index] = functional.normalize(
                CENTER_EMA_MOMENTUM * updated[class_index]
                + (1.0 - CENTER_EMA_MOMENTUM) * candidate,
                p=2,
                dim=0,
            )
    state.centers = updated.detach()
    state.last_center_epoch = epoch_number
    state.reliable_clear_counts = counts


@torch.no_grad()
def compute_sample_weights(
    pooled_features: torch.Tensor,
    targets: torch.Tensor,
    ambiguous: torch.Tensor,
    state: RACWState,
    racw_active: bool,
) -> torch.Tensor:
    """Return detached per-sample loss weights for one training batch."""
    validate_binary_targets(targets)
    if ambiguous.ndim != 1 or ambiguous.numel() != targets.numel():
        raise ValueError("Ambiguity flags do not align with the training targets")

    weights = torch.ones(targets.numel(), device=targets.device, dtype=torch.float32)
    if not racw_active:
        return weights
    if state.centers is None:
        raise RuntimeError("RACW is active but fold centers have not been initialized")
    if pooled_features.ndim != 2 or pooled_features.shape[1] != FEATURE_DIMENSION:
        raise ValueError("RACW pooled features must have shape [batch, 512]")

    fuzzy = ambiguous.bool()
    if bool(fuzzy.any()):
        normalized_features = functional.normalize(
            pooled_features.detach().float(),
            p=2,
            dim=1,
        )
        similarities = normalized_features @ state.centers.detach().float().T
        own_similarity = similarities.gather(1, targets.unsqueeze(1)).squeeze(1)
        other_targets = 1 - targets
        other_similarity = similarities.gather(1, other_targets.unsqueeze(1)).squeeze(1)
        reliability = torch.sigmoid(
            (own_similarity - other_similarity) / RELIABILITY_TEMPERATURE
        )
        fuzzy_weights = FUZZY_MIN_WEIGHT + (1.0 - FUZZY_MIN_WEIGHT) * reliability
        # AMP may produce fp16 similarities/reliabilities, while the loss
        # weights intentionally stay fp32.  Index assignment requires both
        # sides to have the same dtype.
        weights[fuzzy] = fuzzy_weights[fuzzy].to(dtype=weights.dtype)
    return weights.detach()


def summarize_weight_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
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
    state: RACWState,
) -> dict[str, float]:
    """Train with ordinary CE during warm-up and RACW after center initialization."""
    model.train()
    batch_sampler = loader.batch_sampler
    if isinstance(batch_sampler, ShapeBatchSampler):
        batch_sampler.set_epoch(epoch_index)

    racw_active = epoch_index >= WARMUP_EPOCHS
    if racw_active and not state.initialized:
        raise RuntimeError("RACW training began before center initialization")

    weighted_loss_numerator = 0.0
    unweighted_loss_numerator = 0.0
    effective_weight_sum = 0.0
    sample_count = 0
    fuzzy_count = 0
    fuzzy_weight_values: list[float] = []
    fuzzy_normal_weight_values: list[float] = []
    fuzzy_defective_weight_values: list[float] = []
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
        ambiguous = batch["ambiguous"].to(device, non_blocking=True).bool()
        validate_binary_targets(targets)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, pooled = model.forward_with_pooled_features(inputs)
            loss_each = criterion(logits, targets)
            if loss_each.ndim != 1 or loss_each.numel() != targets.numel():
                raise ValueError("RACW requires one CE loss value per sample")
            weights = compute_sample_weights(
                pooled,
                targets,
                ambiguous,
                state,
                racw_active,
            ).to(dtype=loss_each.dtype)
            weighted_numerator = (loss_each * weights).sum()
            weight_denominator = weights.sum()
            if not bool(weight_denominator.gt(0)):
                raise RuntimeError("RACW produced a non-positive loss-weight denominator")
            loss = weighted_numerator / weight_denominator

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_count = targets.size(0)
        weighted_loss_numerator += float(weighted_numerator.detach().item())
        unweighted_loss_numerator += float(loss_each.detach().sum().item())
        effective_weight_sum += float(weight_denominator.detach().item())
        sample_count += batch_count

        fuzzy_indices = ambiguous.detach().cpu().tolist()
        weights_cpu = weights.detach().float().cpu().tolist()
        targets_cpu = targets.detach().cpu().tolist()
        for is_fuzzy, weight, target in zip(fuzzy_indices, weights_cpu, targets_cpu):
            if is_fuzzy:
                fuzzy_count += 1
                fuzzy_weight_values.append(float(weight))
                if target == 0:
                    fuzzy_normal_weight_values.append(float(weight))
                else:
                    fuzzy_defective_weight_values.append(float(weight))

        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        labels_all.extend(targets_cpu)
        probabilities_all.extend(probabilities.cpu().tolist())
        progress.set_postfix(loss=f"{weighted_loss_numerator / effective_weight_sum:.4f}")

    if sample_count == 0 or effective_weight_sum <= 0:
        raise RuntimeError("Training loader produced no usable samples")

    metrics = baseline.calculate_metrics(
        labels_all,
        probabilities_all,
        average_loss=weighted_loss_numerator / effective_weight_sum,
    )
    metrics["unweighted_ce_loss"] = unweighted_loss_numerator / sample_count
    metrics["effective_weight_sum"] = effective_weight_sum
    metrics["fuzzy_sample_count"] = float(fuzzy_count)
    metrics["racw_active"] = float(racw_active)
    for prefix, values in (
        ("fuzzy_weight", fuzzy_weight_values),
        ("fuzzy_normal_weight", fuzzy_normal_weight_values),
        ("fuzzy_defective_weight", fuzzy_defective_weight_values),
    ):
        for key, value in summarize_weight_values(values).items():
            metrics[f"{prefix}_{key}"] = value
    return metrics


def make_racw_group_rows(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build the four requested clear/fuzzy and normal/defective summaries."""
    rows: list[dict[str, object]] = []
    for difficulty in (CLEAR_DIFFICULTY, AMBIGUOUS_DIFFICULTY):
        for label, label_name in ((0, "normal"), (1, "defective")):
            group = [
                row
                for row in predictions
                if str(row["difficulty"]).strip().lower() == difficulty
                and int(row["label"]) == label
            ]
            sample_count = len(group)
            if sample_count == 0:
                raise ValueError(
                    f"Validation output has no samples for {difficulty}-{label_name}"
                )
            correct_count = sum(
                int(row["prediction"]) == int(row["label"]) for row in group
            )
            error_count = sample_count - correct_count
            row: dict[str, object] = {
                "group": f"{difficulty}-{label_name}",
                "difficulty": difficulty,
                "true_label": label,
                "sample_count": sample_count,
                "correct_count": correct_count,
                "error_count": error_count,
                "normal_correct_rate": None,
                "false_positive_count": None,
                "defective_recall": None,
                "false_negative_count": None,
            }
            if label == 0:
                row["normal_correct_rate"] = correct_count / sample_count
                row["false_positive_count"] = error_count
            else:
                row["defective_recall"] = correct_count / sample_count
                row["false_negative_count"] = error_count
            rows.append(row)
    return rows


def save_validation_group_metrics(
    result: EvaluationResult,
    result_dir: Path,
    checkpoint_type: str,
) -> None:
    """Save clear/fuzzy normal and defective validation counts separately."""
    baseline.write_csv(
        result_dir / f"validation_group_metrics_{checkpoint_type}.csv",
        make_racw_group_rows(result.predictions),
    )


def save_racw_evaluation_artifacts(
    result: EvaluationResult,
    result_dir: Path,
    checkpoint_type: str,
) -> None:
    baseline.save_evaluation_artifacts(result, result_dir, checkpoint_type)
    save_validation_group_metrics(result, result_dir, checkpoint_type)


def checkpoint_payload(*, racw_state: RACWState, **kwargs) -> dict[str, object]:
    payload = baseline.checkpoint_payload(**kwargs)
    payload["experiment"] = EXPERIMENT_NAME
    payload["model_metadata"] = architecture_metadata()
    payload["loss_function"] = "CrossEntropyLoss_per_sample_RACW_hard_label"
    payload["racw_metadata"] = racw_metadata()
    payload["racw_center_state"] = racw_state.checkpoint_state()
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
    center_loader = make_center_loader(
        args.split_root,
        fold,
        args.dataset_root,
        args.batch_size,
        args.num_workers,
        seed,
        not args.skip_file_check,
    )
    baseline.dataset_integrity_check(train_loader, val_loader, fold)
    validate_center_loader(train_loader, center_loader, fold)
    validate_ambiguity_metadata(train_loader, val_loader, center_loader)
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
    racw_state = RACWState()

    run_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "eta_min": args.eta_min,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "per-sample hard-label CrossEntropyLoss with RACW fuzzy weights",
        "racw_metadata": racw_metadata(),
        "center_min_reliable_samples": MIN_CENTER_SAMPLES,
        "center_loader": "training fold only; augment=False; eval+no_grad",
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
        "RACW: epochs 1-5 ordinary CE; epoch 6+ fuzzy weights from clear-only "
        "Layer4-GAP centers; primary checkpoint=best_loss"
    )
    print("=" * 100)

    history: list[dict[str, object]] = []
    saved_results: dict[str, EvaluationResult] = {}
    best_loss = float("inf")
    best_f1 = -1.0
    best_f1_tie_loss = float("inf")

    for epoch_index in range(args.epochs):
        epoch_number = epoch_index + 1
        if epoch_index >= WARMUP_EPOCHS:
            refresh_centers(
                model,
                center_loader,
                racw_state,
                device,
                amp_enabled,
                epoch_number,
            )

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
            racw_state,
        )
        val_result = baseline.evaluate(
            model, val_loader, device, amp_enabled, epoch_index, args.epochs
        )
        scheduler.step()

        if epoch_number == WARMUP_EPOCHS:
            initialize_centers(
                model,
                center_loader,
                racw_state,
                device,
                amp_enabled,
                epoch_number,
            )

        val_metrics = val_result.metrics
        history_row = baseline.make_history_row(
            epoch_number, learning_rate, train_metrics, val_metrics
        )
        history_row.update(
            {
                f"train_{key}": value
                for key, value in train_metrics.items()
                if key not in STANDARD_METRIC_KEYS
            }
        )
        history_row.update(racw_state.log_values())
        history.append(history_row)

        common_payload_args = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "fold": fold,
            "seed": seed,
            "epoch": epoch_number,
            "evaluation": val_result,
            "train_metrics": train_metrics,
            "train_summary": train_summary,
            "val_summary": val_summary,
            "run_config": run_config,
            "split_config": split_config,
            "racw_state": racw_state,
        }
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            saved_results["best_loss"] = val_result
            baseline.atomic_torch_save(
                checkpoint_payload(checkpoint_type="best_loss", **common_payload_args),
                model_dir / CHECKPOINT_FILES["best_loss"],
            )
            save_racw_evaluation_artifacts(val_result, result_dir, "best_loss")
            print(
                f"Fold {fold}: new best_loss at epoch {epoch_number}: "
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
            save_racw_evaluation_artifacts(val_result, result_dir, "best_f1")

        print(
            f"Fold {fold} epoch {epoch_number:02d}: "
            f"train weighted loss={train_metrics['loss']:.4f}, "
            f"fuzzy weight={train_metrics['fuzzy_weight_mean']:.3f}, "
            f"F1={train_metrics['f1']:.4f} | val loss={val_metrics['loss']:.4f}, "
            f"Acc={val_metrics['accuracy']:.4f}, P={val_metrics['precision']:.4f}, "
            f"R={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}, "
            f"AUC={val_metrics['auc']:.4f}, "
            f"centers={'on' if racw_state.initialized else 'warmup'}"
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
            racw_state=racw_state,
        ),
        model_dir / CHECKPOINT_FILES["last"],
    )
    save_racw_evaluation_artifacts(val_result, result_dir, "last")
    baseline.save_history(history, result_dir)

    rows = []
    for checkpoint_type in CHECKPOINT_ORDER:
        row: dict[str, object] = {"fold": fold, "checkpoint": checkpoint_type}
        row.update(saved_results[checkpoint_type].metrics)
        rows.append(row)
    baseline.write_csv(result_dir / "checkpoint_metrics.csv", rows)
    return saved_results


def save_racw_oof_summaries(
    fold_results: dict[int, dict[str, EvaluationResult]],
    result_root: Path,
) -> None:
    rows = [
        {"fold": fold, **prediction}
        for fold in sorted(fold_results)
        for prediction in fold_results[fold]["best_loss"].predictions
    ]
    baseline.write_csv(
        result_root / "oof_group_metrics_best_loss.csv",
        make_racw_group_rows(rows),
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
            "racw_metadata": racw_metadata(),
            "center_min_reliable_samples": MIN_CENTER_SAMPLES,
            "primary_checkpoint": "best_loss",
            "internal_test_set": False,
            "external_test_used_for_training_or_selection": False,
            "dataset_root_runtime": str(args.dataset_root),
            "split_config": split_config,
        },
    )

    print("=" * 100)
    print("Head40-533 RACW: clear-center reliability weighting for fuzzy samples")
    print(f"Folds: {folds}")
    print(f"Dataset root (read only): {args.dataset_root}")
    print(f"Split root:  {args.split_root}")
    print(f"Model root:  {args.model_root}")
    print(f"Result root: {args.result_root}")
    print(
        f"Fixed RACW: warmup={WARMUP_EPOCHS}, p_thr={RELIABILITY_P_THRESHOLD}, "
        f"mu={CENTER_EMA_MOMENTUM}, tau={RELIABILITY_TEMPERATURE}, "
        f"fuzzy_min_weight={FUZZY_MIN_WEIGHT}"
    )
    print("Validation and locked external testing use ordinary unweighted metrics.")
    print("=" * 100)

    start_time = time.time()
    fold_results: dict[int, dict[str, EvaluationResult]] = {}
    for fold in folds:
        fold_results[fold] = run_fold(fold, args, device, split_config)
        baseline.save_cross_validation_summary(
            fold_results, args.result_root, sorted(fold_results)
        )
        save_racw_oof_summaries(fold_results, args.result_root)

    elapsed_minutes = (time.time() - start_time) / 60.0
    print("\n" + "=" * 100)
    print(f"Completed folds: {folds}")
    print(f"Elapsed: {elapsed_minutes:.1f} minutes")
    print(f"Primary results: {args.result_root / 'fivefold_mean_std.csv'}")
    print(f"Best-loss OOF: {args.result_root / 'oof_metrics_best_loss.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
