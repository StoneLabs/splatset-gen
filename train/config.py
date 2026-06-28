"""Load training settings from training_config.yaml."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

_TRAIN_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _TRAIN_ROOT / "training_config.yaml"


def _resolve_path(value: str | None, base: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def load_training_config(path: Path | str | None = None) -> SimpleNamespace:
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    paths = raw.get("paths", {})
    dataset = raw.get("dataset", {})
    model = raw.get("model", {})
    training = raw.get("training", {})
    checkpointing = raw.get("checkpointing", {})
    early_stopping = raw.get("early_stopping", {})
    evaluation = raw.get("evaluation", {})
    inference = raw.get("inference", {})

    return SimpleNamespace(
        CONFIG_PATH=str(config_path),
        TRAINING_DATA_DIR=_resolve_path(
            paths.get("training_data_dir", "../outputs"),
            _TRAIN_ROOT,
        ),
        CHECKPOINT_DIR=_resolve_path(paths.get("checkpoint_dir", "checkpoints"), _TRAIN_ROOT),
        LOG_DIR=_resolve_path(paths.get("log_dir", "logs"), _TRAIN_ROOT),
        PREDICTIONS_DIR=_resolve_path(paths.get("predictions_dir", "predictions"), _TRAIN_ROOT),
        BEST_MODEL_NAME=paths.get("best_model_name", "best_by_val_loss.pth"),
        BACKUP_NAME=paths.get("backup_name", "latest_periodic.pth"),
        INTERRUPT_NAME=paths.get("interrupt_name", "interrupted.pth"),
        TRAIN_RUNS=list(dataset.get("train_runs") or []),
        TRAIN_RATIO=float(dataset.get("train_ratio", 0.70)),
        VAL_RATIO=float(dataset.get("val_ratio", 0.20)),
        TEST_RATIO=float(dataset.get("test_ratio", 0.10)),
        SEED=int(dataset.get("seed", 42)),
        BASE_CHANNELS=int(model.get("base_channels", 32)),
        EPOCHS=int(training.get("epochs", 100)),
        BATCH_SIZE=int(training.get("batch_size", 8)),
        NUM_WORKERS=int(training.get("num_workers", 0)),
        LR=float(training.get("lr", 1e-3)),
        LR_MIN=float(training.get("lr_min", 1e-5)),
        WEIGHT_DECAY=float(training.get("weight_decay", 1e-4)),
        GRAD_CLIP=training.get("grad_clip", 1.0),
        USE_AMP=bool(training.get("use_amp", False)),
        BALANCE_RUNS=bool(training.get("balance_runs", True)),
        BCE_WEIGHT=float(training.get("bce_weight", 0.4)),
        BACKUP_EVERY_EPOCHS=int(checkpointing.get("backup_every_epochs", 5)),
        PATIENCE=int(early_stopping.get("patience", 20)),
        DIVERGENCE_GAP=float(early_stopping.get("divergence_gap", 0.10)),
        DIVERGENCE_STREAK=int(early_stopping.get("divergence_streak", 10)),
        MASK_THRESHOLD=float(evaluation.get("mask_threshold", 0.50)),
        SAVE_PREDICTIONS=bool(evaluation.get("save_predictions", False)),
        INFERENCE_CHECKPOINT=_resolve_path(inference.get("checkpoint"), _TRAIN_ROOT),
        DEVICE=str(raw.get("device", "auto")),
    )


cfg = load_training_config()
