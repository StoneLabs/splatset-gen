"""Image augmentation tests."""

from __future__ import annotations

import numpy as np
import torch

from augment import (
    apply_augmentation,
    apply_blur_meta,
    apply_lighting,
    apply_random_lines,
    apply_tear,
    apply_tear_plan,
    apply_warp_plan,
    make_augmentation_rng,
    maybe_gaussian_draw_fraction_range,
    merge_post_augmentation,
    init_augmentation_record,
    finalize_augmentation_record,
    sample_tear_plan,
    sample_warp_plan,
)


def _checkerboard(size: int = 32) -> torch.Tensor:
    arr = np.zeros((size, size, 3), dtype=np.float32)
    arr[::2, ::2] = 1.0
    arr[1::2, 1::2] = 1.0
    return torch.from_numpy(arr)


def test_augmentation_disabled_passthrough() -> None:
    rgb = _checkerboard()
    mask = torch.full((32, 32), 255, dtype=torch.uint8)
    out, meta, out_mask = apply_augmentation(rgb, {}, np.random.default_rng(0), mask=mask)
    assert meta == {}
    assert torch.allclose(out, rgb)
    assert torch.equal(out_mask, mask)


def test_lighting_changes_image_deterministically() -> None:
    rgb = torch.full((16, 16, 3), 0.5)
    cfg = {
        "brightness_range": [0.2, 0.2],
        "contrast_range": [1.0, 1.0],
        "gamma_range": [1.0, 1.0],
        "saturation_range": [1.0, 1.0],
        "warmth_range": [0.0, 0.0],
    }
    out_a, meta_a = apply_lighting(rgb.numpy(), cfg, np.random.default_rng(1))
    out_b, meta_b = apply_lighting(rgb.numpy(), cfg, np.random.default_rng(1))
    assert meta_a == meta_b
    assert np.allclose(out_a, out_b)
    assert float(out_a.mean()) > 0.5


def test_tear_shifts_bands() -> None:
    arr = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8, 1)
    arr = np.repeat(arr, 3, axis=2)
    cfg = {
        "num_bands_range": [2, 2],
        "shift_range": [2, 2],
    }
    out, meta = apply_tear(arr, cfg, np.random.default_rng(7))
    assert meta["num_bands"] == 2
    assert meta["shifts"] != [0, 0]
    assert not np.array_equal(out, arr)


def test_full_augmentation_config() -> None:
    rgb = _checkerboard(64)
    config = {
        "augmentation": {
            "enabled": True,
            "lighting": {"enabled": True, "probability": 1.0},
            "blur": {"enabled": True, "probability": 1.0},
            "tear": {"enabled": False},
            "noise": {"enabled": True, "probability": 1.0, "std_range": [0.05, 0.05]},
            "jpeg": {"enabled": False},
            "vignette": {"enabled": True, "probability": 1.0},
            "chromatic_aberration": {"enabled": True, "probability": 1.0},
        }
    }
    out, meta, out_mask = apply_augmentation(rgb, config, np.random.default_rng(42), mask=None)
    assert out_mask is None
    assert meta["enabled"] is True
    effects = [entry["effect"] for entry in meta["applied"]]
    assert effects[0] == "lighting"
    assert "lighting" in effects
    assert "blur" in effects
    assert "noise" in effects
    assert "vignette" in effects
    assert "chromatic_aberration" in effects
    assert meta["applied"][0]["brightness"] is not None
    assert out.shape == rgb.shape
    assert not torch.allclose(out, rgb)


def test_random_lines_avoid_click_and_clear_mask() -> None:
    h, w = 64, 64
    rgb = torch.zeros(h, w, 3)
    rgb[:, 20:44] = 1.0
    mask = torch.zeros(h, w, dtype=torch.uint8)
    mask[:, 20:44] = 255
    click_x, click_y = 32, 32
    config = {
        "augmentation": {
            "enabled": True,
            "lines": {
                "enabled": True,
                "probability": 1.0,
                "count_range": [8, 8],
                "angle_deg_range": [30, 30],
                "thickness_range": [3, 6],
                "max_attempts_per_line": 64,
            },
        }
    }
    out_rgb, out_mask, meta = apply_random_lines(
        rgb,
        mask,
        click_x,
        click_y,
        config,
        np.random.default_rng(0),
    )
    assert meta is not None
    assert meta["count"] > 0
    assert int(out_mask[click_y, click_x]) == 255
    assert torch.allclose(out_rgb[click_y, click_x], torch.tensor([1.0, 1.0, 1.0]))
    assert int(out_mask.sum()) < int(mask.sum())
    assert not torch.allclose(out_rgb, rgb)
    assert all(line["angle_deg"] == 30.0 for line in meta["lines"])
    assert all(3 <= line["thickness"] <= 6 for line in meta["lines"])


def test_random_lines_respect_fixed_angle_and_thickness() -> None:
    rgb = torch.zeros(64, 64, 3)
    mask = torch.zeros(64, 64, dtype=torch.uint8)
    config = {
        "augmentation": {
            "enabled": True,
            "lines": {
                "enabled": True,
                "probability": 1.0,
                "count_range": [1, 1],
                "angle_deg_range": [90, 90],
                "thickness_range": [4, 4],
                "offset_range": [0, 0],
                "max_attempts_per_line": 32,
            },
        }
    }
    _, _, meta = apply_random_lines(
        rgb,
        mask,
        10,
        10,
        config,
        np.random.default_rng(5),
    )
    assert meta is not None
    assert meta["lines"][0]["angle_deg"] == 90.0
    assert meta["lines"][0]["thickness"] == 4


def test_random_lines_disabled_passthrough() -> None:
    rgb = _checkerboard()
    mask = torch.full((32, 32), 255, dtype=torch.uint8)
    out_rgb, out_mask, meta = apply_random_lines(
        rgb,
        mask,
        16,
        16,
        {},
        np.random.default_rng(0),
    )
    assert meta is None
    assert torch.allclose(out_rgb, rgb)
    assert torch.equal(out_mask, mask)


def test_augmentation_with_mask_tensor() -> None:
    rgb = _checkerboard(32)
    mask = torch.full((32, 32), 128, dtype=torch.uint8)
    config = {
        "augmentation": {
            "enabled": True,
            "lighting": {"enabled": True, "probability": 1.0},
            "tear": {
                "enabled": True,
                "probability": 1.0,
                "num_bands_range": [2, 2],
                "shift_range": [4, 4],
                "horizontal_probability": 1.0,
            },
        }
    }
    out_rgb, meta, out_mask = apply_augmentation(rgb, config, np.random.default_rng(0), mask=mask)
    assert out_mask is not None
    assert out_mask.dtype == torch.uint8
    assert out_mask.shape == mask.shape
    effects = [entry["effect"] for entry in meta["applied"]]
    assert "tear" in effects
    assert "lighting" in effects
    assert not torch.allclose(out_rgb, rgb)


def test_tear_keeps_mask_aligned_with_rgb() -> None:
    h, w = 32, 32
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[8:24, 10:18] = 1.0
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[8:24, 10:18] = 255

    cfg = {
        "num_bands_range": [3, 3],
        "shift_range": [-8, 8],
        "horizontal_probability": 1.0,
    }
    plan, _ = sample_tear_plan((h, w), cfg, np.random.default_rng(0))
    torn_rgb = apply_tear_plan(rgb, plan)
    torn_mask = apply_tear_plan(mask, plan)

    rgb_fg = torn_rgb.max(axis=2) > 0.5
    mask_fg = torn_mask > 0
    assert np.array_equal(rgb_fg, mask_fg)


def test_augmentation_rng_is_seed_and_sample_deterministic() -> None:
    rng_a = make_augmentation_rng(43, "000002")
    rng_b = make_augmentation_rng(43, "000002")
    rng_c = make_augmentation_rng(43, "000003")
    assert rng_a.uniform() == rng_b.uniform()
    assert rng_a.uniform() != rng_c.uniform()


def test_augmentation_rng_independent_of_worker() -> None:
    """Same seed+sample_id must match even if scene rng used worker offset."""
    worker_scene_rng = np.random.default_rng(43 + 2 * 10_007 + 2)
    aug_rng = make_augmentation_rng(43, "000002")
    _ = worker_scene_rng.random(20)
    rgb = _checkerboard(32)
    mask = torch.full((32, 32), 200, dtype=torch.uint8)
    config = {
        "augmentation": {
            "enabled": True,
            "lighting": {"enabled": True, "probability": 1.0},
            "blur": {"enabled": False},
            "tear": {"enabled": False},
            "warp": {"enabled": True, "probability": 1.0, "affine": {"enabled": True}},
            "chromatic_aberration": {"enabled": False},
            "noise": {"enabled": False},
            "jpeg": {"enabled": False},
            "vignette": {"enabled": False},
        }
    }
    out_a, meta_a, _ = apply_augmentation(rgb, config, make_augmentation_rng(43, "000002"), mask=mask)
    out_b, meta_b, _ = apply_augmentation(rgb, config, make_augmentation_rng(43, "000002"), mask=mask)
    assert meta_a == meta_b
    assert torch.allclose(out_a, out_b)


def test_warp_keeps_mask_aligned_with_rgb() -> None:
    h, w = 32, 32
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[8:24, 10:18] = 1.0
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[8:24, 10:18] = 255

    plan = {
        "type": "affine",
        "angle_deg": 5.0,
        "scale": 1.02,
        "shear": 0.03,
        "tx": 2.0,
        "ty": -1.0,
    }
    warped_rgb, warped_mask = apply_warp_plan(rgb, mask, plan, mask_mode="soft")
    assert warped_mask is not None
    rgb_fg = warped_rgb.max(axis=2) > 0.5
    mask_fg = (warped_mask.astype(np.float32) / 255.0) > 0.5
    assert np.array_equal(rgb_fg, mask_fg)


def test_blur_meta_applies_same_motion_to_mask() -> None:
    h, w = 32, 32
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[8:24, 10:18] = 1.0
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[8:24, 10:18] = 255
    meta = {"type": "motion", "length": 5, "angle_deg": 30.0}

    blurred_rgb = apply_blur_meta(rgb, meta)
    blurred_mask = apply_blur_meta(mask, meta)
    rgb_fg = blurred_rgb.max(axis=2) > 0.5
    mask_fg = (blurred_mask.astype(np.float32) / 255.0) > 0.5
    assert np.array_equal(rgb_fg, mask_fg)


def test_gaussian_subset_disabled_returns_none() -> None:
    config = {"augmentation": {"enabled": True, "gaussian_subset": {"enabled": False}}}
    assert maybe_gaussian_draw_fraction_range(config, np.random.default_rng(0)) is None


def test_gaussian_subset_probability_gated() -> None:
    config = {
        "augmentation": {
            "enabled": True,
            "gaussian_subset": {
                "enabled": True,
                "probability": 0.0,
                "fraction_range": [0.1, 1.0],
            },
        }
    }
    assert maybe_gaussian_draw_fraction_range(config, np.random.default_rng(0)) is None


def test_gaussian_subset_returns_fraction_range() -> None:
    config = {
        "augmentation": {
            "enabled": True,
            "gaussian_subset": {
                "enabled": True,
                "probability": 1.0,
                "fraction_range": [0.2, 0.8],
            },
        }
    }
    assert maybe_gaussian_draw_fraction_range(config, np.random.default_rng(0)) == [0.2, 0.8]


def test_augmentation_record_ordered_applied_list() -> None:
    record = init_augmentation_record(
        {
            "augmentation": {
                "enabled": True,
                "gaussian_subset": {"enabled": True, "probability": 0.4},
            }
        },
    )
    assert record is not None
    assert record["applied"][0] == {"effect": "gaussian_subset", "enabled": True}

    post = {
        "enabled": True,
        "applied": [
            {"effect": "lighting", "brightness": 0.1, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0, "warmth": 0.0},
            {"effect": "blur", "type": "gaussian", "radius": 1.0},
        ],
    }
    merged = merge_post_augmentation(record, post)
    assert merged is not None
    assert [entry["effect"] for entry in merged["applied"]] == [
        "gaussian_subset",
        "lighting",
        "blur",
    ]
    finalized = finalize_augmentation_record(merged)
    assert finalized is not None
    assert "objects" not in finalized["applied"][0]
