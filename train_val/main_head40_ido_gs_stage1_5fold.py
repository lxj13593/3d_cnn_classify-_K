"""Standalone Head40 IDO-GS Stage 1: ordinary CE, trajectories, and diagnostics.

This entry point never starts Stage 2 and never opens the external test set.
Run directly from an IDE or use --help. Diagnostics-only does not require torch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import random
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_val.ido_gs_stage1_diagnostics import diagnose_fold

EXPERIMENT = "head40_stem533_ido_gs_stage1_22boards_5fold"
VERSION = "IDO-GS-stage1-2026-09-21-standalone-v1"
FOLD_SEEDS = (42, 123, 2026, 3407, 777)
EPOCHS = 50
FIRST_OBSERVATION = 6
METRIC_KEYS = ("loss", "accuracy", "precision", "recall", "f1",
               "specificity", "auc", "ap")
CHECKPOINT_ORDER = ("best_f1", "best_loss", "last")
METRIC_DISPLAY = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1-Score", "specificity": "Specificity", "auc": "AUC-ROC", "ap": "AP",
}
PREDICTION_FIELDS = (
    "sample_id", "board", "sequence", "difficulty", "raw_shape_whd",
    "input_shape_dhw", "raw_path", "b_head_up", "direction_flipped",
    "direction_standardized", "standardized_head_side", "label", "prediction",
    "probability_defective", "probability_normal", "sample_loss",
)
SETTINGS = {
    "version": VERSION, "stage": 1, "epochs": EPOCHS, "batch_size": 4,
    "num_workers": 0, "optimizer": "AdamW", "lr": 1e-4,
    "weight_decay": 1e-3, "betas": [0.9, 0.999],
    "scheduler": "CosineAnnealingLR", "eta_min": 1e-7,
    "dropout": 0.5, "threshold": 0.5, "fold_seeds": list(FOLD_SEEDS),
    "first_observation_epoch": FIRST_OBSERVATION,
    "last_observation_epoch": EPOCHS, "bootstrap_repetitions": 100,
    "supervision_mean_min": 0.8, "low_supervision_fraction_max": 0.1,
    "loss": "unweighted_cross_entropy", "automatic_stage2": False,
}
SOURCE_FILES = (
    "train_val/main_head40_ido_gs_stage1_5fold.py",
    "train_val/ido_gs_stage1_diagnostics.py",
    "data_operate/data_load_head40_ido_gs.py",
    "model/resnet18_3d_head40_stem533_ido_gs.py",
)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2,
                  allow_nan=False)
    temporary.replace(path)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows) -> None:
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        raise ValueError(f"Refusing an empty results table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(json_safe(first))
        writer.writerows(json_safe(row) for row in iterator)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_metrics_csv(path: Path, metrics: dict) -> None:
    write_csv(path, (
        {"metric": key, "value": metrics[key]}
        for key in ("loss", *METRIC_DISPLAY, "tn", "fp", "fn", "tp", "sample_count")
    ))


def try_write_excel(path: Path, rows: list[dict]) -> None:
    try:
        import pandas as pd
        pd.DataFrame(rows).to_excel(path, index=False)
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Excel export skipped ({exc}); CSV output is complete: {path}")


def digest(value) -> str:
    serialized = json.dumps(json_safe(value), ensure_ascii=False,
                            sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "resnet18_head40_toml_direction_unified_multiboard_5fold_clear_fuzzy_20260924",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model-root", type=Path,
                        default=PROJECT_ROOT / "model_best_last" / EXPERIMENT)
    parser.add_argument("--result-root", type=Path,
                        default=PROJECT_ROOT / "train_val_result" / EXPERIMENT)
    parser.add_argument("--folds", default="all", help="all, 0, or 0,1,2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume owned checkpoints; completed folds are reused.")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Use saved Stage-1 trajectories; no raw data/GPU needed.")
    return parser.parse_args()


def parse_folds(text: str) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(5))
    folds = sorted({int(part.strip()) for part in text.split(",")})
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("--folds must contain numbers from 0 to 4.")
    return folds


def sample_metadata(sample) -> dict:
    return {
        "sample_id": str(sample.sample_id), "label": int(sample.label_index),
        "difficulty": str(sample.difficulty), "board": str(sample.board),
        "sequence": str(sample.sequence), "raw_sha256": str(sample.raw_sha256),
        "raw_shape_whd": "x".join(map(str, sample.shape_whd)),
        "input_shape_dhw": "x".join(map(str, sample.shape_dhw)),
        "b_head_up": bool(sample.b_head_up),
        "direction_flipped": bool(sample.direction_flip_required),
        "standardized_head_side": str(sample.standardized_head_side),
    }


class TrainingTrajectory:
    """SampleID-aligned float32 probabilities; predictions are always p >= .5."""

    def __init__(self, metadata: list[dict], probabilities=None):
        self.metadata = metadata
        self.sample_ids = [str(row["sample_id"]) for row in metadata]
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Training SampleIDs must be nonempty and unique.")
        self.labels = np.asarray([row["label"] for row in metadata], dtype=np.int64)
        self.difficulties = np.asarray([row["difficulty"] for row in metadata])
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Only normal=0 and defect=1 are allowed.")
        if not np.isin(self.difficulties,
                       ["clear", "fuzzy", "reviewed_defect", "supplemental"]).all():
            raise ValueError("Missing or unknown sample difficulty metadata.")
        self.probabilities = (
            np.empty((0, len(metadata)), dtype=np.float32) if probabilities is None
            else np.asarray(probabilities, dtype=np.float32)
        )
        if (self.probabilities.ndim != 2
                or self.probabilities.shape[1] != len(metadata)
                or self.probabilities.shape[0] > 45
                or not np.isfinite(self.probabilities).all()
                or not ((self.probabilities >= 0) & (self.probabilities <= 1)).all()):
            raise ValueError("Invalid Stage-1 trajectory probabilities.")

    @property
    def predictions(self):
        return (self.probabilities >= 0.5).astype(np.int8)

    @property
    def n_observed(self):
        return int(self.probabilities.shape[0])

    def append(self, epoch: int, rows: list[dict]) -> None:
        if epoch != FIRST_OBSERVATION + self.n_observed:
            raise ValueError("Wrong-event epochs must be consecutive, starting at 6.")
        indexed = {}
        for row in rows:
            key = str(row["sample_id"])
            if key in indexed:
                raise ValueError(f"Duplicate train-eval SampleID: {key}")
            indexed[key] = row
        if set(indexed) != set(self.sample_ids):
            raise ValueError("Train-eval must contain each training SampleID once.")
        ordered = [indexed[key] for key in self.sample_ids]
        if any(int(row["label"]) != int(label)
               or str(row["difficulty"]) != difficulty
               for row, label, difficulty
               in zip(ordered, self.labels, self.difficulties)):
            raise ValueError("Train-eval labels/difficulty changed for a SampleID.")
        values = np.asarray([row["probability_defective"] for row in ordered],
                            dtype=np.float32)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Invalid probabilities in deterministic train-eval.")
        if any(int(row["prediction"]) != int(prob >= 0.5)
               for row, prob in zip(ordered, values)):
            raise ValueError("Prediction threshold differs from fixed p >= 0.5.")
        self.probabilities = np.vstack((self.probabilities, values))

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, sample_ids=np.asarray(self.sample_ids), labels=self.labels,
                difficulties=self.difficulties, probabilities=self.probabilities,
                epochs=np.arange(6, 6 + self.n_observed, dtype=np.int64),
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, metadata: list[dict]):
        with np.load(path, allow_pickle=False) as archive:
            trace = cls(metadata, archive["probabilities"])
            expected = (np.asarray(trace.sample_ids), trace.labels, trace.difficulties,
                        np.arange(6, 6 + trace.n_observed))
            for key, value in zip(("sample_ids", "labels", "difficulties", "epochs"),
                                  expected):
                if not np.array_equal(archive[key], value):
                    raise ValueError(f"Trajectory/metadata mismatch: {key}")
        return trace

    def export(self, folder: Path, fold: int) -> None:
        def rows():
            for offset, probabilities in enumerate(self.probabilities):
                for index, value in enumerate(probabilities):
                    yield {
                        "fold": fold, "sample_id": self.sample_ids[index],
                        "epoch": 6 + offset, "label": int(self.labels[index]),
                        "difficulty": str(self.difficulties[index]),
                        "prediction": int(value >= 0.5),
                        "probability_defective": float(value),
                    }
        if self.n_observed:
            write_csv(folder / "stage1_epoch_predictions.csv", rows())
        if self.n_observed >= 2:
            changes = np.mean(self.predictions[1:] != self.predictions[:-1], axis=1)
            write_csv(folder / "stage1_pc_curve.csv", (
                {"fold": fold, "epoch": 7 + index, "prediction_change": float(value)}
                for index, value in enumerate(changes)
            ))


def metrics_from_rows(rows: list[dict]) -> dict:
    labels = np.asarray([int(row["label"]) for row in rows])
    probabilities = np.asarray([float(row["probability_defective"]) for row in rows])
    predictions = (probabilities >= 0.5).astype(np.int64)
    losses = np.asarray([float(row["sample_loss"]) for row in rows])
    if not len(rows) or not np.isfinite(probabilities).all() or not np.isfinite(losses).all():
        raise ValueError("Evaluation contains no samples or non-finite values.")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "loss": float(losses.mean()), "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None,
        "ap": float(average_precision_score(labels, probabilities)) if np.any(labels == 1) else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sample_count": len(rows),
    }


def group_rows(rows: list[dict], fold) -> list[dict]:
    result = []
    groups = sorted({(int(row["label"]), str(row["difficulty"])) for row in rows})
    for label, difficulty in groups:
        selected = [row for row in rows
                    if int(row["label"]) == label and row["difficulty"] == difficulty]
        correct = sum(int(float(row["probability_defective"]) >= 0.5) == label
                      for row in selected)
        result.append({
            "fold": fold, "label": label, "difficulty": difficulty,
            "sample_count": len(selected), "correct": correct,
            "tn": correct if label == 0 else 0,
            "fp": len(selected) - correct if label == 0 else 0,
            "tp": correct if label == 1 else 0,
            "fn": len(selected) - correct if label == 1 else 0,
            "normal_correct_rate": correct / len(selected) if label == 0 else None,
            "defect_recall": correct / len(selected) if label == 1 else None,
        })
    return result


def load_training_dependencies():
    global torch, data, make_model, architecture_metadata
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training requires the PyTorch environment on your training PC. "
                           "--diagnose-only works without PyTorch.") from exc
    from data_operate import data_load_head40_ido_gs as data
    from model.resnet18_3d_head40_stem533_ido_gs import (
        resnet18_3d as make_model, architecture_metadata,
    )


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def preserved_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def evaluate(model, loader, device, amp_enabled: bool, description: str):
    """Read-only evaluation; additional passes cannot perturb training RNG."""
    metadata = {row["sample_id"]: row
                for row in map(sample_metadata, loader.dataset.samples)}
    original_mode = model.training
    rows = []
    try:
        with preserved_rng(), torch.no_grad():
            model.eval()
            for batch in tqdm(loader, desc=description, file=sys.stdout, dynamic_ncols=True):
                targets = batch["target"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(batch["volume"].to(device, non_blocking=True))
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    logits = logits.float()
                    losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                    probabilities = torch.softmax(logits, dim=1)[:, 1]
                for key, label, probability, loss in zip(
                    batch["sample_id"], targets.cpu().tolist(),
                    probabilities.cpu().tolist(), losses.cpu().tolist(),
                ):
                    if key not in metadata or metadata[key]["label"] != label:
                        raise ValueError("Evaluation returned an unexpected SampleID/label.")
                    rows.append({
                        **metadata[key], "prediction": int(probability >= 0.5),
                        "probability_defective": probability, "sample_loss": loss,
                    })
    finally:
        model.train(original_mode)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(metadata) or len(set(ids)) != len(ids) or set(ids) != set(metadata):
        raise ValueError("Evaluation did not cover each manifest SampleID exactly once.")
    rows.sort(key=lambda row: row["sample_id"])
    return metrics_from_rows(rows), rows


def train_epoch(model, loader, optimizer, scaler, device, amp_enabled: bool, epoch: int):
    model.train()
    loader.batch_sampler.set_epoch(epoch - 1)
    rows = []
    progress = tqdm(loader, desc=f"Epoch {epoch:02d}/50 train", file=sys.stdout,
                    dynamic_ncols=True)
    total_loss, total_n = 0.0, 0
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(batch["volume"].to(device, non_blocking=True))
        with torch.amp.autocast(device_type=device.type, enabled=False):
            sample_losses = torch.nn.functional.cross_entropy(
                logits.float(), targets, reduction="none")
            loss = sample_losses.mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite CE training loss; last checkpoint retained.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        probabilities = torch.softmax(logits.detach().float(), dim=1)[:, 1]
        for label, probability, sample_loss in zip(
            targets.cpu().tolist(), probabilities.cpu().tolist(),
            sample_losses.detach().cpu().tolist(),
        ):
            rows.append({"label": label, "probability_defective": probability,
                         "sample_loss": sample_loss})
        total_loss += float(loss.detach()) * len(targets)
        total_n += len(targets)
        progress.set_postfix(loss=f"{total_loss / total_n:.4f}")
    if total_n != len(loader.dataset):
        raise ValueError("Training sampler changed the number of samples.")
    return metrics_from_rows(rows)


def validate_split(split_root: Path) -> dict:
    config = read_json(split_root / "split_config.json")
    expected = {
        "internal_test_set": False, "n_splits": 5,
        "train_dataset_name": "multiboard_head40_toml",
        "assignment": "within_board_label_full_volume_shape_balanced_direction_unified",
        "direction_standardized": True, "sample_count": 4364,
        "normal_count": 3181, "defective_count": 1183, "board_count": 22,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Split config {key}: expected {value!r}, got {config.get(key)!r}")
    direction = config.get("direction_standardization", {})
    if direction.get("target_head_side") != "high_depth_index" or direction.get("bHeadUp_true") != "flip_D":
        raise ValueError("Split direction standardization differs from Head40 protocol.")
    return config


def validate_datasets(train_dataset, val_dataset, fold: int):
    train = list(map(sample_metadata, train_dataset.samples))
    val = list(map(sample_metadata, val_dataset.samples))
    train_ids = {row["sample_id"] for row in train}
    val_ids = {row["sample_id"] for row in val}
    if len(train_ids) != len(train) or len(val_ids) != len(val) or train_ids & val_ids:
        raise ValueError("Duplicate IDs or train/validation leakage in manifests.")
    if len(train) + len(val) != 4364:
        raise ValueError("The fold must partition all 4364 training-pool samples.")
    if Counter(row["label"] for row in train + val) != Counter({0: 3181, 1: 1183}):
        raise ValueError("The normal/defect counts do not match the fixed data pool.")
    for samples, is_train in ((train_dataset.samples, True), (val_dataset.samples, False)):
        if len({sample.board for sample in samples}) != 22:
            raise ValueError("Each fold subset must contain the original 22 boards.")
        for sample in samples:
            if ((sample.validation_fold == fold) == is_train
                    or sample.standardized_head_side != "high_depth_index"):
                raise ValueError("Fold assignment or direction metadata mismatch.")
    return train, val


def atomic_checkpoint(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_fold(fold: int, args, device, split_config: dict):
    folder = args.result_root / f"fold_{fold}"
    model_dir = args.model_root / f"fold_{fold}"
    last_path = model_dir / "stage1_last.pth"
    epoch50_path = model_dir / "stage1_epoch50.pth"
    resume_path = epoch50_path if epoch50_path.exists() else last_path
    if not args.resume and any(path.exists() and any(path.iterdir())
                               for path in (folder, model_dir)):
        raise FileExistsError(f"Fold {fold} already has files. Use --resume or a new output directory.")
    if args.resume and not resume_path.exists() and (
        (folder / "stage1_trajectory.npz").exists()
        or (model_dir / "stage1_best_loss.pth").exists()
    ):
        raise FileNotFoundError(f"Fold {fold} has results but no recoverable last checkpoint.")
    folder.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    seed = FOLD_SEEDS[fold]
    set_seed(seed, args.deterministic)
    train_loader, val_loader = data.get_fold_loaders(
        args.split_root, fold, dataset_root=args.dataset_root,
        batch_size=4, num_workers=0, seed=seed, verify_files=True,
    )
    train_meta, val_meta = validate_datasets(train_loader.dataset, val_loader.dataset, fold)
    train_eval = data.make_loader(
        args.split_root / f"fold_{fold}" / "train.csv",
        dataset_root=args.dataset_root, batch_size=4, num_workers=0,
        augment=False, seed=seed + 1000, verify_files=False,
    )
    if list(map(sample_metadata, train_eval.dataset.samples)) != train_meta:
        raise ValueError("Deterministic evaluation does not match the training manifest.")
    amp_enabled = device.type == "cuda" and not args.disable_amp
    identity = {
        "settings": SETTINGS, "fold": fold, "seed": seed,
        "train_identity": digest(train_meta), "validation_identity": digest(val_meta),
        "amp_enabled": amp_enabled, "deterministic": args.deterministic,
        "model": architecture_metadata(),
    }
    identity_hash = digest(identity)
    config_path = folder / "fold_config.json"
    if config_path.exists() and read_json(config_path)["identity_hash"] != identity_hash:
        raise ValueError("Saved run and current data/protocol differ; use a separate output directory.")
    model = make_model(num_classes=2, dropout_rate=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                 weight_decay=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-7)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    loaders = {"train": train_loader, "val": val_loader, "train_eval": train_eval}
    trace = TrainingTrajectory(train_meta)
    start_epoch, best_loss, best_epoch = 0, float("inf"), None
    best_f1, best_f1_loss, best_f1_epoch = -1.0, float("inf"), None
    history, best_rows, best_metrics = [], [], None
    best_f1_rows, best_f1_metrics, last_rows, last_metrics = [], None, [], None
    if args.resume and resume_path.exists():
        # Only load checkpoints produced by this experiment.
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("identity_hash") != identity_hash:
            raise ValueError("Checkpoint data identity or Stage-1 settings changed.")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"])
        trace = TrainingTrajectory(train_meta, state["trajectory_probabilities"])
        if trace.n_observed != max(0, start_epoch - 5) or start_epoch not in range(51):
            raise ValueError("Checkpoint epoch/trajectory mismatch.")
        history = state["history"]
        best_loss, best_epoch = state["best_loss"], state["best_epoch"]
        best_rows, best_metrics = state["best_rows"], state["best_metrics"]
        best_f1 = state.get("best_f1", -1.0)
        best_f1_loss = state.get("best_f1_loss", float("inf"))
        best_f1_epoch = state.get("best_f1_epoch")
        best_f1_rows = state.get("best_f1_rows", [])
        best_f1_metrics = state.get("best_f1_metrics")
        last_rows = state.get("last_rows", [])
        last_metrics = state.get("last_metrics")
        for name, loader in loaders.items():
            loader.generator.set_state(state["loader_rng_states"][name].cpu())
        restore_rng(state["rng_state"])
        print(f"Fold {fold}: restored epoch {start_epoch}, observations={trace.n_observed}")
    save_json(config_path, {**identity, "identity_hash": identity_hash,
                           "dataset_root_runtime": args.dataset_root,
                           "model_directory_runtime": model_dir,
                           "device": str(device)})
    save_json(folder / "sample_metadata.json", train_meta)
    save_json(folder / "validation_metadata.json", val_meta)
    print(f"\nFold {fold} | train={len(train_meta)} val={len(val_meta)} | "
          f"device={device} AMP={amp_enabled} | ordinary CE, all weights=1")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_epoch(model, train_loader, optimizer, scaler,
                                    device, amp_enabled, epoch)
        val_metrics, val_rows = evaluate(model, val_loader, device, amp_enabled,
                                        f"Epoch {epoch:02d}/50 validation")
        if epoch >= FIRST_OBSERVATION:
            _, train_rows = evaluate(model, train_eval, device, amp_enabled,
                                     f"Epoch {epoch:02d}/50 train-eval")
            trace.append(epoch, train_rows)
        scheduler.step()
        history.append({
            "epoch": epoch, "learning_rate": learning_rate,
            **{f"train_{key}": train_metrics[key] for key in METRIC_KEYS},
            **{f"val_{key}": val_metrics[key] for key in METRIC_KEYS},
        })
        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss, best_epoch = val_metrics["loss"], epoch
            best_rows, best_metrics = val_rows, val_metrics
        f1_improved = (val_metrics["f1"] > best_f1 + 1e-12
                       or (math.isclose(val_metrics["f1"], best_f1, abs_tol=1e-12)
                           and val_metrics["loss"] < best_f1_loss))
        if f1_improved:
            best_f1, best_f1_loss, best_f1_epoch = (
                val_metrics["f1"], val_metrics["loss"], epoch)
            best_f1_rows, best_f1_metrics = val_rows, val_metrics
        if epoch == EPOCHS:
            last_rows, last_metrics = val_rows, val_metrics
        payload = {
            "experiment": EXPERIMENT, "version": VERSION, "stage": 1,
            "identity_hash": identity_hash, "fold": fold, "fold_seed": seed,
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_metadata": architecture_metadata(), "settings": SETTINGS,
            "normalization_params": data.get_normalization_params(),
            "augmentation_params": data.get_augmentation_params(),
            "direction_standardization": data.get_direction_standardization_params(),
            "decision_threshold": 0.5, "loss_function": "CrossEntropyLoss_unweighted",
            "class_to_index": {"normal": 0, "defective": 1},
            "validation_metrics": val_metrics, "best_metrics": best_metrics,
            "best_loss": best_loss, "best_epoch": best_epoch, "best_rows": best_rows,
            "best_f1": best_f1, "best_f1_loss": best_f1_loss,
            "best_f1_epoch": best_f1_epoch, "best_f1_rows": best_f1_rows,
            "best_f1_metrics": best_f1_metrics, "last_rows": last_rows,
            "last_metrics": last_metrics,
            "history": history, "sample_ids": trace.sample_ids,
            "trajectory_probabilities": trace.probabilities,
            "wrong_count": (trace.predictions != trace.labels[None, :]).sum(axis=0),
            "n_observed": trace.n_observed, "bmm_parameters": None,
            "rng_state": capture_rng(),
            "loader_rng_states": {name: loader.generator.get_state()
                                  for name, loader in loaders.items()},
        }
        # The last checkpoint is authoritative if a process stops during CSV export.
        if improved:
            atomic_checkpoint(model_dir / "stage1_best_loss.pth",
                              {**payload, "checkpoint_type": "best_loss"})
        if f1_improved:
            atomic_checkpoint(model_dir / "stage1_best_f1.pth",
                              {**payload, "checkpoint_type": "best_f1"})
        atomic_checkpoint(last_path, {**payload, "checkpoint_type": "last"})
        trace.save(folder / "stage1_trajectory.npz")
        write_csv(folder / "stage1_history.csv", history)
        print(f"Fold {fold} epoch {epoch:02d}: val loss={val_metrics['loss']:.6f}, "
              f"F1={val_metrics['f1']:.4f}, best epoch={best_epoch}, "
              f"wrong-event observations={trace.n_observed}")

    if trace.n_observed != 45 or best_metrics is None:
        raise RuntimeError("Stage 1 did not finish all 50 epochs and 45 observations.")
    if last_path.exists():
        last_path.replace(epoch50_path)
    trace.save(folder / "stage1_trajectory.npz")
    trace.export(folder, fold)
    write_csv(folder / "stage1_history.csv", history)
    write_csv(folder / "stage1_best_loss_predictions.csv", best_rows)
    write_csv(folder / "stage1_best_loss_groups.csv", group_rows(best_rows, fold))
    save_json(folder / "stage1_best_loss_metrics.json",
              {"fold": fold, "best_epoch": best_epoch, **best_metrics})
    if best_f1_metrics is not None and best_f1_rows:
        write_csv(folder / "stage1_best_f1_predictions.csv", best_f1_rows)
        save_json(folder / "stage1_best_f1_metrics.json",
                  {"fold": fold, "best_epoch": best_f1_epoch, **best_f1_metrics})
    if last_metrics is not None and last_rows:
        write_csv(folder / "stage1_last_predictions.csv", last_rows)
        save_json(folder / "stage1_last_metrics.json",
                  {"fold": fold, "best_epoch": EPOCHS, **last_metrics})
    save_json(folder / "training_complete.json",
              {"version": VERSION, "fold": fold, "epoch": 50,
               "identity_hash": identity_hash, "n_observed": 45})
    del model, optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()


def diagnose_saved_fold(folder: Path, fold: int):
    complete = read_json(folder / "training_complete.json")
    config = read_json(folder / "fold_config.json")
    metadata = read_json(folder / "sample_metadata.json")
    trace_path = folder / "stage1_trajectory.npz"
    trace = TrainingTrajectory.load(trace_path, metadata)
    if (complete.get("version") != VERSION or complete.get("epoch") != 50
            or complete.get("fold") != fold or trace.n_observed != 45
            or complete.get("identity_hash") != config["identity_hash"]
            or config["train_identity"] != digest(metadata)):
        raise ValueError("Diagnostics require a complete, matching Stage-1 record.")
    fingerprint = diagnostics_fingerprint(trace_path, metadata)
    output = folder / "stage1_bmm_diagnostics.json"
    if output.exists():
        cached = read_json(output)
        if cached.get("diagnostics_fingerprint") == fingerprint:
            print(f"Fold {fold}: using existing matching diagnostics.")
            return cached
    print(f"\nFold {fold}: CPU BMM diagnostics (the network is not trained here).")
    summary, arrays = diagnose_fold(
        trace.predictions, trace.labels, trace.difficulties.tolist(), FOLD_SEEDS[fold],
        progress=lambda message: print(f"  Fold {fold}: {message}", flush=True),
    )
    summary = {**summary, "fold": fold, "version": VERSION,
               "identity_hash": config["identity_hash"],
               "diagnostics_fingerprint": fingerprint}
    sample_rows = [
        {"fold": fold, **row,
         **{key: values[index] for key, values in arrays.items()}}
        for index, row in enumerate(metadata)
    ]
    write_csv(folder / "stage1_sample_diagnostics.csv", sample_rows)
    group_diagnostics = []
    for label, difficulty in sorted({(row["label"], row["difficulty"]) for row in metadata}):
        mask = (trace.labels == label) & (trace.difficulties == difficulty)
        pc = np.asarray(arrays["pc"])[mask]
        valid = np.isfinite(pc).all()
        group_diagnostics.append({
            "fold": fold, "label": label, "difficulty": difficulty,
            "sample_count": int(mask.sum()), "valid": bool(valid),
            "mean_pc": float(pc.mean()) if valid else None,
            "median_pc": float(np.median(pc)) if valid else None,
            "pc_below_half_count": int(np.sum(pc < 0.5)) if valid else None,
            "pc_below_half_fraction": float(np.mean(pc < 0.5)) if valid else None,
        })
    write_csv(folder / "stage1_group_diagnostics.csv", group_diagnostics)
    save_json(output, summary)
    print(f"Fold {fold}: gate_pass={summary['gate_pass']}; "
          f"reasons={summary.get('failure_reasons', [])}")
    return summary


def diagnostics_fingerprint(trace_path: Path, metadata: list[dict]) -> str:
    return digest({
        "trajectory": file_hash(trace_path), "train_identity": digest(metadata),
        "implementation": file_hash(PROJECT_ROOT / SOURCE_FILES[1]), "version": VERSION,
    })


def legacy_prediction_rows(rows: list[dict], fold: int, split_root: Path,
                           dataset_root: Path | None) -> list[dict]:
    manifest_path = split_root / f"fold_{fold}" / "val.csv"
    manifest = ({row["sample_id"]: row for row in read_csv(manifest_path)}
                if manifest_path.exists() else {})
    if manifest and set(manifest) != {row["sample_id"] for row in rows}:
        raise ValueError(f"Fold {fold}: validation manifest differs from saved predictions.")
    converted = []
    for row in rows:
        source = manifest.get(row["sample_id"], {})
        if source and (int(source["label_index"]) != int(row["label"])
                       or source["difficulty"] != row["difficulty"]):
            raise ValueError(f"Fold {fold}: saved label/difficulty differs from manifest.")
        relative_path = source.get("raw_relative_path", "")
        raw_path = row.get("raw_path", "")
        if not raw_path and relative_path and dataset_root is not None:
            raw_path = str(dataset_root / Path(relative_path))
        probability = float(row["probability_defective"])
        if int(row["prediction"]) != int(probability >= 0.5):
            raise ValueError(f"Fold {fold}: prediction threshold mismatch.")
        converted.append({
            "sample_id": row["sample_id"], "board": row["board"],
            "sequence": row["sequence"], "difficulty": row["difficulty"],
            "raw_shape_whd": row["raw_shape_whd"],
            "input_shape_dhw": row["input_shape_dhw"], "raw_path": raw_path,
            "b_head_up": str(row["b_head_up"]).lower() == "true",
            "direction_flipped": str(row["direction_flipped"]).lower() == "true",
            "direction_standardized": True,
            "standardized_head_side": row["standardized_head_side"],
            "label": int(row["label"]), "prediction": int(row["prediction"]),
            "probability_defective": probability,
            "probability_normal": 1.0 - probability,
            "sample_loss": float(row["sample_loss"]),
        })
    if any(tuple(row) != PREDICTION_FIELDS for row in converted):
        raise ValueError("Legacy prediction fields do not match the historical output.")
    return converted


def grouped_oof_metrics(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, []).append(row)
    return [
        {**dict(zip(fields, key)), **metrics_from_rows(group)}
        for key, group in sorted(groups.items())
    ]


def dataset_summary(rows: list[dict]) -> dict:
    shapes = Counter(row["raw_shape_whd"] for row in rows)
    return {
        "total": len(rows), "normal": sum(int(row["label"]) == 0 for row in rows),
        "defective": sum(int(row["label"]) == 1 for row in rows),
        "boards": len({row["board"] for row in rows}),
        "raw_shape_whd_counts": dict(sorted(shapes.items())),
        "b_head_up_false": sum(not row["b_head_up"] for row in rows),
        "b_head_up_true": sum(bool(row["b_head_up"]) for row in rows),
        "direction_flipped": sum(bool(row["direction_flipped"]) for row in rows),
        "standardized_head_side": "high_depth_index",
    }


def save_legacy_plots(folder: Path, checkpoint: str, rows: list[dict],
                      metrics: dict) -> None:
    expected = (folder / f"confusion_matrix_{checkpoint}.png",
                folder / f"roc_{checkpoint}.png", folder / f"pr_{checkpoint}.png")
    if all(path.exists() for path in expected):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (ConfusionMatrixDisplay, roc_curve,
                                     precision_recall_curve)
    except ImportError:
        print("Plot export skipped: matplotlib is unavailable.")
        return
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    probabilities = np.asarray([row["probability_defective"] for row in rows])
    display = ConfusionMatrixDisplay(
        confusion_matrix(labels, predictions, labels=[0, 1]),
        display_labels=["Normal", "Defective"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Validation confusion matrix: {checkpoint}")
    fig.tight_layout()
    fig.savefig(folder / f"confusion_matrix_{checkpoint}.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)
    if np.unique(labels).size != 2:
        return
    fpr, tpr, _ = roc_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_title(f"Validation ROC: {checkpoint}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / f"roc_{checkpoint}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, lw=2, label=f"AP = {metrics['ap']:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision")
    ax.set_title(f"Validation precision-recall: {checkpoint}")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / f"pr_{checkpoint}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_legacy_training_plot(folder: Path, history: list[dict]) -> None:
    if (folder / "training_curves.png").exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, key, title in (
        (axes[0, 0], "loss", "Loss"),
        (axes[0, 1], "accuracy", "Accuracy"),
        (axes[1, 0], "f1", "F1-Score"),
    ):
        axis.plot(epochs, [float(row[f"train_{key}"]) for row in history],
                  label="Train")
        axis.plot(epochs, [float(row[f"val_{key}"]) for row in history],
                  label="Validation")
        axis.set_title(title)
        axis.legend()
    axes[1, 1].plot(epochs, [float(row["learning_rate"]) for row in history])
    axes[1, 1].set_title("Learning rate")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "training_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_legacy_outputs(result_root: Path, split_root: Path,
                        dataset_root: Path | None, training_folds: list[int]) -> None:
    """Keep the prior Stem533 result filenames/columns; never invent a missing checkpoint."""
    all_metrics, oof_rows = [], []
    missing_raw_paths = 0
    available = {checkpoint: [] for checkpoint in CHECKPOINT_ORDER}
    for fold in training_folds:
        folder = result_root / f"fold_{fold}"
        history = read_csv(folder / "stage1_history.csv")
        write_csv(folder / "training_history.csv", history)
        save_legacy_training_plot(folder, history)
        train_meta = read_json(folder / "sample_metadata.json")
        val_meta = read_json(folder / "validation_metadata.json")
        save_json(folder / "dataset_summary.json", {
            "train": dataset_summary(train_meta),
            "validation": dataset_summary(val_meta),
        })
        config = read_json(folder / "fold_config.json")
        save_json(folder / "run_config.json", {
            "epochs": EPOCHS, "batch_size": SETTINGS["batch_size"],
            "num_workers": SETTINGS["num_workers"],
            "learning_rate": SETTINGS["lr"],
            "weight_decay": SETTINGS["weight_decay"],
            "eta_min": SETTINGS["eta_min"], "optimizer": SETTINGS["optimizer"],
            "scheduler": SETTINGS["scheduler"],
            "loss": "CrossEntropyLoss_unweighted",
            "amp_enabled": config["amp_enabled"],
            "deterministic_algorithms": config["deterministic"],
            "device": config.get("device", "unknown_in_saved_stage1"),
            "dataset_root_runtime": str(dataset_root) if dataset_root else None,
            "fold": fold, "fold_seed": FOLD_SEEDS[fold],
        })
        fold_metrics = []
        for checkpoint in CHECKPOINT_ORDER:
            metrics_path = folder / f"stage1_{checkpoint}_metrics.json"
            predictions_path = folder / f"stage1_{checkpoint}_predictions.csv"
            if not (metrics_path.exists() and predictions_path.exists()):
                continue
            metrics = read_json(metrics_path)
            converted = legacy_prediction_rows(
                read_csv(predictions_path), fold, split_root, dataset_root)
            missing_raw_paths += sum(not row["raw_path"] for row in converted)
            if len(converted) != int(metrics["sample_count"]):
                raise ValueError(f"Fold {fold} {checkpoint}: prediction count mismatch.")
            calculated = metrics_from_rows(converted)
            for key in ("loss", "accuracy", "precision", "recall", "f1", "specificity", "auc", "ap",
                        "tn", "fp", "fn", "tp", "sample_count"):
                if not math.isclose(float(calculated[key]), float(metrics[key]),
                                    rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError(f"Fold {fold} {checkpoint}: {key} does not match predictions.")
            metric_row = {"fold": fold, "checkpoint": checkpoint}
            metric_row.update({key: metrics[key] for key in
                               (*METRIC_KEYS, "tn", "fp", "fn", "tp", "sample_count")})
            fold_metrics.append(metric_row)
            all_metrics.append(metric_row)
            available[checkpoint].append(fold)
            write_csv(folder / f"validation_predictions_{checkpoint}.csv", converted)
            save_metrics_csv(folder / f"validation_metrics_{checkpoint}.csv", metrics)
            save_legacy_plots(folder, checkpoint, converted, metrics)
            if checkpoint == "best_loss":
                oof_rows.extend({"fold": fold, **row} for row in converted)
        if fold_metrics:
            write_csv(folder / "checkpoint_metrics.csv", fold_metrics)
    if not all_metrics:
        return
    protocol = read_json(result_root / "protocol.json")
    save_json(result_root / "experiment_config.json", {
        "experiment": EXPERIMENT, "folds": training_folds,
        "fold_seeds": list(FOLD_SEEDS), "model": protocol["model"],
        "normalization": protocol["normalization"],
        "augmentation": protocol["augmentation"],
        "direction_standardization": protocol["direction"],
        "primary_checkpoint": "best_loss", "internal_test_set": False,
        "external_test_used_for_training_or_selection": False,
        "dataset_root_runtime": str(dataset_root) if dataset_root else None,
        "split_config": protocol["split_config"],
    })
    write_csv(result_root / "fivefold_checkpoint_metrics.csv", all_metrics)
    try_write_excel(result_root / "fivefold_checkpoint_metrics.xlsx", all_metrics)
    formatted = []
    for key, display in METRIC_DISPLAY.items():
        row = {"Metric": display}
        for checkpoint in CHECKPOINT_ORDER:
            values = np.asarray([float(item[key]) for item in all_metrics
                                 if item["checkpoint"] == checkpoint])
            if len(values) == len(training_folds):
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[checkpoint] = f"{mean * 100:.2f}% ± {std * 100:.2f}%"
            else:
                row[checkpoint] = ""
        formatted.append(row)
    write_csv(result_root / "fivefold_mean_std.csv", formatted)
    try_write_excel(result_root / "fivefold_mean_std.xlsx", formatted)
    if oof_rows:
        oof_rows.sort(key=lambda row: row["sample_id"])
        if len({row["sample_id"] for row in oof_rows}) != len(oof_rows):
            raise ValueError("Duplicate OOF SampleIDs in legacy export.")
        write_csv(result_root / "oof_predictions_best_loss.csv", oof_rows)
        oof_metrics = metrics_from_rows(oof_rows)
        save_metrics_csv(result_root / "oof_metrics_best_loss.csv", oof_metrics)
        save_json(result_root / "oof_metrics_best_loss.json", oof_metrics)
        write_csv(result_root / "oof_metrics_best_loss_by_board.csv",
                  grouped_oof_metrics(oof_rows, ("board",)))
        write_csv(result_root / "oof_metrics_best_loss_by_board_shape.csv",
                  grouped_oof_metrics(oof_rows, ("board", "raw_shape_whd")))
    save_json(result_root / "legacy_export_status.json", {
        "available_folds_by_checkpoint": available,
        "missing_raw_path_fields": missing_raw_paths,
        "missing_checkpoints_cannot_be_reconstructed_from_stage1_history": {
            checkpoint: sorted(set(training_folds) - set(folds))
            for checkpoint, folds in available.items()
        },
    })


def summarize_available(result_root: Path, split_root: Path | None = None,
                        dataset_root: Path | None = None) -> None:
    raw_rows, oof, groups, diagnoses = [], [], [], {}
    training_folds = []
    for fold in range(5):
        folder = result_root / f"fold_{fold}"
        if not (folder / "training_complete.json").exists():
            continue
        marker = read_json(folder / "training_complete.json")
        config = read_json(folder / "fold_config.json")
        train_meta = read_json(folder / "sample_metadata.json")
        val_meta = read_json(folder / "validation_metadata.json")
        if (marker.get("version") != VERSION or marker.get("epoch") != 50
                or marker.get("fold") != fold
                or marker.get("identity_hash") != config.get("identity_hash")
                or config.get("settings") != SETTINGS
                or config.get("train_identity") != digest(train_meta)
                or config.get("validation_identity") != digest(val_meta)):
            raise ValueError(f"Fold {fold} has a different or incomplete protocol.")
        training_folds.append(fold)
        metrics = read_json(folder / "stage1_best_loss_metrics.json")
        raw_rows.append({"method": "Stage1_CE", **metrics})
        predictions = read_csv(folder / "stage1_best_loss_predictions.csv")
        expected = {row["sample_id"]: row for row in val_meta}
        predicted_ids = [row["sample_id"] for row in predictions]
        if (set(predicted_ids) != set(expected)
                or len(predicted_ids) != len(expected)
                or any(int(row["label"]) != expected[row["sample_id"]]["label"]
                       or row["difficulty"] != expected[row["sample_id"]]["difficulty"]
                       for row in predictions)):
            raise ValueError(f"Fold {fold} validation predictions do not match its manifest.")
        oof.extend({"fold": fold, **row} for row in predictions)
        groups.extend(group_rows(predictions, fold))
        diag_path = folder / "stage1_bmm_diagnostics.json"
        if diag_path.exists():
            diagnosis = read_json(diag_path)
            fingerprint = diagnostics_fingerprint(folder / "stage1_trajectory.npz", train_meta)
            if (diagnosis.get("identity_hash") != marker["identity_hash"]
                    or diagnosis.get("diagnostics_fingerprint") != fingerprint):
                raise ValueError(f"Fold {fold} has stale diagnostics.")
            diagnoses[fold] = diagnosis
    if raw_rows:
        write_csv(result_root / "comparison_folds.csv", raw_rows)
        write_csv(result_root / "comparison_groups.csv", groups)
    ids = [str(row["sample_id"]) for row in oof]
    if len(set(ids)) != len(ids):
        raise ValueError("OOF SampleIDs occur in more than one validation fold.")
    if training_folds == list(range(5)):
        if len(oof) != 4364:
            raise ValueError("Full five-fold OOF must contain 4364 unique samples.")
        fuzzy_defect_count = sum(int(row["label"]) == 1 and row["difficulty"] == "fuzzy"
                                 for row in oof)
        if fuzzy_defect_count != 144:
            raise ValueError("Expected exactly 144 OOF fuzzy-defect samples.")
        means = {"method": "Stage1_CE", "fold_count": 5}
        for key in METRIC_KEYS:
            values = [row[key] for row in raw_rows]
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError(f"Cannot report a five-fold mean with invalid {key}.")
            means[key] = float(np.mean(values))
        write_csv(result_root / "comparison_mean.csv", [means])
        write_csv(result_root / "oof_predictions_best_loss.csv",
                  sorted(oof, key=lambda row: str(row["sample_id"])))
        write_csv(result_root / "oof_groups_best_loss.csv", group_rows(oof, "OOF"))
    all_diagnosed = set(diagnoses) == set(range(5))
    stage2_allowed = all_diagnosed and all(row["gate_pass"] for row in diagnoses.values())
    status = ("pass" if stage2_allowed else "fail") if all_diagnosed else "incomplete"
    report = {
        "version": VERSION, "status": status, "stage2_allowed": stage2_allowed,
        "training_completed_folds": training_folds,
        "diagnosed_folds": sorted(diagnoses),
        "fold_results": {str(fold): {
            "gate_pass": row["gate_pass"], "failure_reasons": row.get("failure_reasons", [])
        } for fold, row in diagnoses.items()},
        "automatic_stage2_started": False,
    }
    save_json(result_root / "stage1_gate.json", report)
    if training_folds:
        protocol_path = result_root / "protocol.json"
        if dataset_root is None and protocol_path.exists():
            stored_root = read_json(protocol_path).get("dataset_root_runtime")
            dataset_root = Path(stored_root) if stored_root else None
        save_legacy_outputs(result_root, split_root or Path(),
                            dataset_root, training_folds)
    print(f"\nStage 1 status: {status}; Stage 2 allowed: {stage2_allowed}")
    print(f"Decision: {result_root / 'stage1_gate.json'}")


def main() -> None:
    args = parse_args()
    args.result_root = args.result_root.resolve()
    args.model_root = args.model_root.resolve()
    args.split_root = args.split_root.resolve()
    folds = parse_folds(args.folds)
    if args.diagnose_only:
        for fold in folds:
            diagnose_saved_fold(args.result_root / f"fold_{fold}", fold)
        summarize_available(args.result_root, args.split_root, args.dataset_root)
        return
    load_training_dependencies()
    split_config = validate_split(args.split_root)
    args.dataset_root = data.resolve_dataset_root(
        explicit_root=args.dataset_root,
        stored_root=split_config.get("train_dataset_root_at_creation"),
        expected_name="multiboard_head40_toml",
    )
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment.")
    args.result_root.mkdir(parents=True, exist_ok=True)
    args.model_root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.result_root / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path)["settings"] != SETTINGS:
        raise ValueError("Output root belongs to a different protocol.")
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch"):
        versions[package] = importlib.metadata.version(package)
    if not protocol_path.exists():
        save_json(protocol_path, {
            "settings": SETTINGS, "software": versions,
            "source_sha256": {name: file_hash(PROJECT_ROOT / name) for name in SOURCE_FILES},
            "split_config": split_config, "dataset_root_runtime": args.dataset_root,
            "model": architecture_metadata(),
            "normalization": data.get_normalization_params(),
            "augmentation": data.get_augmentation_params(),
            "direction": data.get_direction_standardization_params(),
        })
    print(f"Standalone IDO-GS Stage 1 | folds={folds}")
    print(f"Dataset: {args.dataset_root}\nResults: {args.result_root}")
    for fold in folds:
        run_fold(fold, args, device, split_config)
        diagnose_saved_fold(args.result_root / f"fold_{fold}", fold)
        summarize_available(args.result_root, args.split_root, args.dataset_root)
    print("Stage 1 finished. Read stage1_gate.json before considering Stage 2.")


if __name__ == "__main__":
    main()
