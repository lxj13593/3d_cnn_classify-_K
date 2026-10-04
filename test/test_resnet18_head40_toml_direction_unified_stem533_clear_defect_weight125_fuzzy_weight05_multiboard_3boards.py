"""Locked three-board evaluation for the clear-defect-1.25/fuzzy-0.5 run.

Evaluation intentionally uses ordinary unweighted cross-entropy and the normal
binary metrics.  Sample weights affect training only.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.resnet18_3d_head40_toml_direction_unified_stem533_clear_defect_weight125_fuzzy_weight05_multiboard import (  # noqa: E402
    architecture_metadata,
    resnet18_3d,
)


def load_shared_evaluator():
    """Load the project evaluator without colliding with Python's test package."""
    path = (
        PROJECT_ROOT
        / "test"
        / "test_resnet18_head40_toml_direction_unified_stem533_multiboard_3boards.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_head40_stem533_shared_locked_test", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = load_shared_evaluator()


EXPERIMENT_NAME = (
    "resnet18_head40_toml_direction_unified_stem533_"
    "clear_defect_weight125_fuzzy_weight05_multiboard_22boards_5fold"
)
DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "resnet18_head40_toml_direction_unified_multiboard_5fold_clear_fuzzy_20260924"
)
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "model_best_last" / EXPERIMENT_NAME
# Keep the output folder short: per-board plot folders append the complete board
# identifier and otherwise exceed the legacy Windows MAX_PATH limit.
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "test_result" / "head40_stem533_cd125_fz05_locked_3boards"
)
CHECKPOINT_FILES = {
    "best_f1": "best_f1_stem533_cd125_fz05.pth",
    "best_loss": "best_loss_stem533_cd125_fz05.pth",
    "last": "last_stem533_cd125_fz05.pth",
}
CHECKPOINT_ORDER = ("best_f1", "best_loss", "last")


def sample_weighting_metadata() -> dict[str, object]:
    return {
        "enabled_training_only": True,
        "label_task": "normal=0, defective=1",
        "ambiguous_definition": "difficulty == 'fuzzy'",
        "clear_normal_weight": 1.0,
        "clear_defective_weight": 1.25,
        "fuzzy_normal_weight": 0.5,
        "fuzzy_defective_weight": 0.5,
        "loss_formula": "sum(sample_weight * CE_i) / sum(sample_weight)",
        "validation_loss": "ordinary unweighted CrossEntropyLoss",
        "external_test_loss": "ordinary unweighted CrossEntropyLoss",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all five clear-defect-1.25/fuzzy-0.5 folds on the locked "
            "three-board test set with ordinary unweighted binary metrics."
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


_shared_save_json = shared.save_json


def load_checkpoint_model(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no model_state_dict: {path}")
    if checkpoint.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(
            f"Checkpoint experiment mismatch in {path}: "
            f"{checkpoint.get('experiment')}"
        )
    if checkpoint.get("model_metadata") != architecture_metadata():
        raise ValueError(
            "Checkpoint is not the standalone Head40-533 "
            f"clear-defect-1.25/fuzzy-0.5 model: {path}"
        )
    if checkpoint.get("direction_standardization") != shared.get_direction_standardization_params():
        raise ValueError(f"Checkpoint direction metadata mismatch in {path}")
    normalization = checkpoint.get("normalization_params", checkpoint.get("norm_params"))
    shared.validate_normalization_params(normalization)
    if checkpoint.get("class_to_index") != {"normal": 0, "defective": 1}:
        raise ValueError(f"Class mapping mismatch in {path}")

    expected = sample_weighting_metadata()
    expected_loss = "CrossEntropyLoss_per_sample_clear_defect_weight125_fuzzy_weight05"
    if checkpoint.get("loss_function") != expected_loss:
        raise ValueError(
            "Checkpoint is not the clear-defect-1.25/fuzzy-0.5 "
            f"experiment: {path}"
        )
    if checkpoint.get("sample_weighting") != expected:
        raise ValueError(f"Checkpoint sample-weight metadata mismatch: {path}")

    model = resnet18_3d(num_classes=2, dropout_rate=0.5)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def save_json(path: Path, value: object) -> None:
    if path.name == "test_config.json" and isinstance(value, dict):
        value = {
            **value,
            "experiment": EXPERIMENT_NAME,
            "sample_weighting": sample_weighting_metadata(),
            "evaluation_loss": "ordinary unweighted CrossEntropyLoss",
        }
    _shared_save_json(path, value)


def main() -> None:
    # The shared evaluator supplies the unchanged locked-test protocol.  Patch
    # only its experiment-specific inputs and validate checkpoint provenance.
    shared.EXPERIMENT_NAME = EXPERIMENT_NAME
    shared.DEFAULT_SPLIT_ROOT = DEFAULT_SPLIT_ROOT
    shared.DEFAULT_MODEL_ROOT = DEFAULT_MODEL_ROOT
    shared.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    shared.CHECKPOINT_FILES = CHECKPOINT_FILES
    shared.CHECKPOINT_ORDER = CHECKPOINT_ORDER
    shared.parse_args = parse_args
    shared.load_checkpoint_model = load_checkpoint_model
    shared.save_json = save_json
    shared.main()


if __name__ == "__main__":
    main()
