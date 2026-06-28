"""Tests for binary + soft alpha training metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
sys.path.insert(0, str(_TRAIN_DIR))

from train import _batch_metrics_cpu  # noqa: E402


def test_binary_perfect_match() -> None:
    logits = torch.full((1, 1, 8, 8), 10.0)
    masks = torch.ones(1, 1, 8, 8)
    metrics, _ = _batch_metrics_cpu(logits, masks)
    assert metrics["bin_f1"][0] > 0.99
    assert metrics["soft_f1"][0] > 0.99
    assert metrics["alpha_mae"][0] < 0.01


def test_soft_alpha_perfect_match_uses_mae() -> None:
    """Identical soft alpha and target → zero MAE (soft F1 is not 1 on partial alphas)."""
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, 1:3, 1:3] = 0.5
    logits = torch.full((1, 1, 4, 4), -10.0)
    logits[0, 0, 1:3, 1:3] = 0.0  # sigmoid(0) = 0.5

    metrics, _ = _batch_metrics_cpu(logits, gt)
    assert metrics["alpha_mae"][0] < 0.01


def test_wrong_alpha_hurts_both_binary_and_soft() -> None:
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, 1:3, 1:3] = 1.0
    logits = torch.full((1, 1, 4, 4), -10.0)  # sigmoid ~ 0 everywhere

    metrics, _ = _batch_metrics_cpu(logits, gt)
    assert metrics["bin_f1"][0] < 0.5
    assert metrics["soft_f1"][0] < 0.5
    assert metrics["alpha_mae"][0] > 0.2
