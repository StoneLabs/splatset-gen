"""Background compositing tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from background import (
    BackgroundSpec,
    composite,
    load_background_image,
    list_background_images,
    sample_background_layer,
)


def _write_test_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def test_load_background_image_crop_and_stretch(tmp_path: Path) -> None:
    img_path = tmp_path / "bg.png"
    _write_test_image(img_path, (200, 100), (255, 0, 0))

    crop = load_background_image(img_path, 64, 32, "crop")
    assert crop.shape == (32, 64, 3)
    assert crop[16, 32, 0] > 0.9

    stretch = load_background_image(img_path, 64, 32, "stretch")
    assert stretch.shape == (32, 64, 3)


def test_load_background_image_letterbox(tmp_path: Path) -> None:
    img_path = tmp_path / "bg.png"
    _write_test_image(img_path, (200, 100), (0, 255, 0))

    letter = load_background_image(img_path, 64, 64, "letterbox", solid_color=(0.2, 0.2, 0.2))
    assert letter.shape == (64, 64, 3)
    assert letter[0, 0, 0] == 0.2


def test_composite_image_background(tmp_path: Path) -> None:
    img_path = tmp_path / "bg.png"
    _write_test_image(img_path, (32, 32), (0, 0, 255))

    fg_rgb = torch.zeros(32, 32, 3)
    fg_rgb[10:20, 10:20, :] = 1.0
    alpha = torch.zeros(32, 32)
    alpha[10:20, 10:20] = 1.0

    spec = BackgroundSpec(mode="image", image_dir=tmp_path, resize_mode="stretch")
    rng = np.random.default_rng(0)
    rgb, meta = composite(fg_rgb, alpha, spec, 32, 32, rng=rng)

    assert meta["mode"] == "image"
    assert meta["image"] == "bg.png"
    assert rgb[0, 0, 2] > 0.9
    assert rgb[15, 15, 0] > 0.9


def test_list_background_images(tmp_path: Path) -> None:
    _write_test_image(tmp_path / "a.jpg", (8, 8), (1, 1, 1))
    _write_test_image(tmp_path / "b.png", (8, 8), (2, 2, 2))
    (tmp_path / ".gitkeep").write_text("")

    names = [p.name for p in list_background_images(tmp_path)]
    assert names == ["a.jpg", "b.png"]


def test_sample_background_layer_solid() -> None:
    spec = BackgroundSpec(mode="solid", solid_color=(0.25, 0.5, 0.75))
    bg, meta = sample_background_layer(spec, 16, 16, np.random.default_rng(0))
    assert bg.shape == (16, 16, 3)
    assert meta == {"mode": "solid", "color": [0.25, 0.5, 0.75]}
