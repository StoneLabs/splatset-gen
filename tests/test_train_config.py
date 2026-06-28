"""Tests for train/training_config.yaml loading."""

from __future__ import annotations

import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
sys.path.insert(0, str(_TRAIN_DIR))

from config import load_training_config  # noqa: E402


def test_load_training_config_defaults() -> None:
    cfg = load_training_config(_TRAIN_DIR / "training_config.yaml")
    assert cfg.EPOCHS == 100
    assert cfg.BATCH_SIZE == 8
    assert cfg.MASK_THRESHOLD == 0.5
    assert cfg.INFERENCE_CHECKPOINT is None
    assert cfg.TRAINING_DATA_DIR.endswith("outputs")
