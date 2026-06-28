"""Tests that train-time augmentation keeps the click on the object mask."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
sys.path.insert(0, str(_TRAIN_DIR))

from dataset import (  # noqa: E402
    apply_augment,
    click_on_mask,
    transform_crop_resize,
    transform_hflip,
    transform_rotate,
    transform_vflip,
)


def _synthetic_sample(size: int = 128) -> tuple[Image.Image, Image.Image, float, float]:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    for y in range(40, 88):
        for x in range(40, 88):
            img.putpixel((x, y), (220, 40, 40))
            mask.putpixel((x, y), 255)
    click_x, click_y = 63.0, 64.0
    assert click_on_mask(mask, click_x, click_y)
    return img, mask, click_x, click_y


def test_hflip_keeps_click_on_mask() -> None:
    img, mask, x, y = _synthetic_sample()
    img2, mask2, x2, y2 = transform_hflip(img, mask, x, y)
    assert click_on_mask(mask2, x2, y2)


def test_vflip_keeps_click_on_mask() -> None:
    img, mask, x, y = _synthetic_sample()
    img2, mask2, x2, y2 = transform_vflip(img, mask, x, y)
    assert click_on_mask(mask2, x2, y2)


def test_rotate_keeps_click_on_mask() -> None:
    img, mask, x, y = _synthetic_sample()
    for angle in (-20.0, -7.5, 0.0, 9.0, 19.0):
        img2, mask2, x2, y2 = transform_rotate(img, mask, x, y, angle)
        assert click_on_mask(mask2, x2, y2), f"click left object at angle={angle}"


def test_crop_resize_keeps_click_on_mask() -> None:
    img, mask, x, y = _synthetic_sample()
    result = transform_crop_resize(img, mask, x, y, scale=0.82, x0=20, y0=18)
    assert result is not None
    img2, mask2, x2, y2 = result
    assert click_on_mask(mask2, x2, y2)


def test_crop_skipped_when_click_would_leave_object() -> None:
    img, mask, x, y = _synthetic_sample()
    result = transform_crop_resize(img, mask, x, y, scale=0.5, x0=0, y0=0)
    assert result is None


def test_color_only_ops_do_not_move_click() -> None:
    img, mask, x, y = _synthetic_sample()
    mask_before = np.array(mask)
    out_img, out_mask, x2, y2 = apply_augment(
        img,
        mask,
        x,
        y,
        rng=random.Random(0),
        overrides={"hflip": False, "vflip": False, "rotate": False, "crop": False, "jitter": True, "blur": True},
    )
    assert x2 == x and y2 == y
    assert np.array_equal(np.array(out_mask), mask_before)
    assert not np.array_equal(np.array(out_img), np.array(img))


def test_apply_augment_spatial_chain_keeps_click_on_mask() -> None:
    img, mask, x, y = _synthetic_sample()
    _, mask2, x2, y2 = apply_augment(
        img,
        mask,
        x,
        y,
        overrides={
            "hflip": True,
            "vflip": True,
            "rotate": 12.0,
            "crop": {"scale": 0.85, "x0": 10, "y0": 8},
            "jitter": False,
            "blur": False,
        },
    )
    assert click_on_mask(mask2, x2, y2)


def test_apply_augment_randomized_runs_keep_click_on_mask() -> None:
    img, mask, x, y = _synthetic_sample()
    rng = random.Random(123)
    for _ in range(40):
        _, mask2, x2, y2 = apply_augment(img, mask, x, y, rng=rng)
        assert click_on_mask(mask2, x2, y2)
