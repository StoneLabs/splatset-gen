"""Tests for inference PNG encoding (all format × visualization × background combos)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
sys.path.insert(0, str(TRAIN_DIR))

from inference import ModelRunner  # noqa: E402

ALPHA = np.array(
    [
        [0, 64, 128],
        [192, 255, 32],
    ],
    dtype=np.uint8,
)
GT = np.array(
    [
        [0, 128, 128],
        [255, 0, 64],
    ],
    dtype=np.uint8,
)


@pytest.mark.parametrize(
    ("output_format", "visualization", "background", "expected_mode"),
    [
        ("alpha", "raw", "transparent", "RGBA"),
        ("alpha", "raw", "black", "L"),
        ("alpha", "compare", "transparent", "RGBA"),
        ("alpha", "compare", "black", "RGB"),
        ("binary", "raw", "transparent", "RGBA"),
        ("binary", "raw", "black", "L"),
        ("binary", "compare", "transparent", "RGBA"),
        ("binary", "compare", "black", "RGB"),
    ],
)
def test_encode_prediction_png_all_combinations(
    output_format: str,
    visualization: str,
    background: str,
    expected_mode: str,
) -> None:
    gt = GT if visualization == "compare" else None
    encoded, mode = ModelRunner.encode_prediction_png(
        ALPHA,
        output_format=output_format,
        visualization=visualization,
        background=background,
        gt_u8=gt,
        threshold=0.5,
    )

    assert mode == expected_mode
    if mode == "L":
        assert encoded.shape == ALPHA.shape
    elif mode == "RGB":
        assert encoded.shape == (*ALPHA.shape, 3)
    else:
        assert encoded.shape == (*ALPHA.shape, 4)


def test_encode_alpha_png_transparent_rgba() -> None:
    rgba = ModelRunner.encode_alpha_png(ALPHA, background="transparent")
    assert rgba.shape == (2, 3, 4)
    assert rgba[0, 0, 3] == 0
    assert np.all(rgba[..., :3] == 255)


def test_encode_alpha_png_black_grayscale() -> None:
    gray = ModelRunner.encode_alpha_png(ALPHA, background="black")
    assert gray.shape == (2, 3)
    assert gray[1, 1] == 255


def test_encode_binary_transparent_white_detect() -> None:
    encoded, mode = ModelRunner.encode_prediction_png(
        ALPHA,
        output_format="binary",
        visualization="raw",
        background="transparent",
        threshold=0.5,
    )
    assert mode == "RGBA"
    assert encoded[1, 1, 3] == 255
    assert encoded[0, 0, 3] == 0


def test_encode_compare_black_has_colored_pixels() -> None:
    encoded, mode = ModelRunner.encode_compare_png(
        ALPHA,
        GT,
        output_format="alpha",
        background="black",
        threshold=0.5,
    )
    assert mode == "RGB"
    assert encoded[0, 0].tolist() == [0, 0, 0]
    assert encoded.max() > 0
    assert np.any((encoded != 0).any(axis=-1))


def test_encode_compare_black_alpha_uses_soft_strength() -> None:
    encoded, mode = ModelRunner.encode_compare_png(
        ALPHA,
        GT,
        output_format="alpha",
        background="black",
        threshold=0.5,
    )
    assert mode == "RGB"
    signal = encoded[(encoded != 0).any(axis=-1)]
    assert signal.size > 0
    full_tp = np.array([56, 203, 92], dtype=np.uint8)
    assert np.any(signal != full_tp)


def test_encode_compare_transparent_tn_is_clear() -> None:
    encoded, mode = ModelRunner.encode_compare_png(
        ALPHA,
        GT,
        output_format="alpha",
        background="transparent",
        threshold=0.5,
    )
    assert mode == "RGBA"
    assert encoded[0, 0, 3] == 0


def test_encode_compare_transparent_alpha_uses_soft_strength() -> None:
    encoded, mode = ModelRunner.encode_compare_png(
        ALPHA,
        GT,
        output_format="alpha",
        background="transparent",
        threshold=0.5,
    )
    assert mode == "RGBA"
    signal_alphas = encoded[..., 3][encoded[..., 3] > 0]
    assert signal_alphas.size > 0
    assert np.any((signal_alphas > 0) & (signal_alphas < 255))


def test_encode_prediction_png_rejects_unknown_background() -> None:
    with pytest.raises(ValueError, match="Unknown background"):
        ModelRunner.encode_prediction_png(ALPHA, background="white")


def test_encode_prediction_png_compare_requires_gt() -> None:
    with pytest.raises(ValueError, match="requires gt_u8"):
        ModelRunner.encode_prediction_png(
            ALPHA,
            visualization="compare",
            background="black",
        )
