"""Tests for inference alpha PNG encoding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
sys.path.insert(0, str(TRAIN_DIR))

from inference import ModelRunner  # noqa: E402


def test_encode_alpha_png_transparent_rgba() -> None:
    alpha = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    rgba = ModelRunner.encode_alpha_png(alpha, background="transparent")

    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] == 128
    assert np.all(rgba[..., :3] == 255)
    assert ModelRunner.alpha_png_mode("transparent") == "RGBA"


def test_encode_alpha_png_black_grayscale() -> None:
    alpha = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    gray = ModelRunner.encode_alpha_png(alpha, background="black")

    assert gray.shape == (2, 2)
    assert gray[1, 0] == 255
    assert ModelRunner.alpha_png_mode("black") == "L"


def test_encode_alpha_png_rejects_unknown_background() -> None:
    alpha = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown alpha background"):
        ModelRunner.encode_alpha_png(alpha, background="white")
