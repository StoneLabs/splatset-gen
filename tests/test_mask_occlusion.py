"""Occlusion-aware mask tests."""

from __future__ import annotations

import numpy as np
import torch

from camera import intrinsics_from_fov, look_at_viewmat
from picker import object_mask, sample_click
from render import render
from synthetic_gaussians import concat_objects, make_object_blob

MASK_WEIGHT_THRESHOLD = 0.05


def test_rear_object_mask_black_where_occluded() -> None:
    """Rear object mask is black under front object (modal / visible-only)."""
    rear = make_object_blob(0, center=(0.0, 0.0, 0.0), sh_dc=(1.0, 0.1, 0.1), seed=1)
    front = make_object_blob(1, center=(0.0, 0.0, 0.35), sh_dc=(0.1, 1.0, 0.1), seed=2)
    scene = concat_objects([rear, front])

    width = height = 128
    viewmat = look_at_viewmat(
        eye=np.array([0.0, 0.0, 2.5], dtype=np.float64),
        target=np.array([0.0, 0.0, 0.15], dtype=np.float64),
    )
    k = intrinsics_from_fov(width, height, 60.0)

    out = render(scene, viewmat, k, width, height)
    assert out.alpha.max() > 0.5

    overlap = (out.object_id_map == 0) | (out.object_id_map == 1)
    assert overlap.sum() > 0

    occluded_by_front = out.object_id_map == 1
    rear_mask = object_mask(
        out.object_weights, clicked_object_id=0, weight_threshold=MASK_WEIGHT_THRESHOLD
    )
    rear_weight = out.object_weights[:, :, 0]

    assert rear_mask[occluded_by_front].max().item() == 0
    visible_rear = (rear_weight > MASK_WEIGHT_THRESHOLD) & (out.object_id_map == 0)
    assert rear_mask[visible_rear].min().item() == 255

    rng = np.random.default_rng(0)
    x, y, clicked_id = sample_click(out.alpha, out.object_id_map, 0.5, rng)
    mask = object_mask(
        out.object_weights, clicked_id, weight_threshold=MASK_WEIGHT_THRESHOLD
    )
    assert mask[y, x].item() == 255


def test_front_object_mask_filled_where_visible() -> None:
    rear = make_object_blob(0, center=(0.0, 0.0, 0.0), seed=3)
    front = make_object_blob(1, center=(0.0, 0.0, 0.35), seed=4)
    scene = concat_objects([rear, front])

    width = height = 128
    viewmat = look_at_viewmat(
        eye=np.array([0.0, 0.0, 2.5], dtype=np.float64),
        target=np.array([0.0, 0.0, 0.15], dtype=np.float64),
    )
    k = intrinsics_from_fov(width, height, 60.0)
    out = render(scene, viewmat, k, width, height)

    front_mask = object_mask(
        out.object_weights, clicked_object_id=1, weight_threshold=MASK_WEIGHT_THRESHOLD
    )
    visible_front = (out.object_weights[:, :, 1] > MASK_WEIGHT_THRESHOLD) & (
        out.object_id_map == 1
    )
    assert front_mask[visible_front].min().item() == 255


def test_transparent_fringe_excluded_from_mask() -> None:
    """Very low per-object weight must not produce full-white mask pixels."""
    obj = make_object_blob(0, center=(0.0, 0.0, 0.0), opacity=0.95, seed=5)
    width = height = 128
    viewmat = look_at_viewmat(
        eye=np.array([0.0, 0.0, 2.5], dtype=np.float64),
        target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    k = intrinsics_from_fov(width, height, 60.0)
    out = render(obj, viewmat, k, width, height)

    fringe = (out.object_weights[:, :, 0] > 0.0) & (
        out.object_weights[:, :, 0] <= MASK_WEIGHT_THRESHOLD
    )
    assert fringe.sum() > 0

    mask = object_mask(
        out.object_weights, clicked_object_id=0, weight_threshold=MASK_WEIGHT_THRESHOLD
    )
    assert mask[fringe].max().item() == 0


def test_dominant_object_id_follows_accumulated_weight() -> None:
    """Dominant object id tracks highest accumulated compositing weight."""
    rear = make_object_blob(
        0,
        center=(0.0, 0.0, 0.0),
        opacity=0.95,
        sh_dc=(1.0, 0.1, 0.1),
        seed=1,
    )
    front = make_object_blob(
        1,
        center=(0.0, 0.0, 0.35),
        opacity=0.25,
        sh_dc=(0.1, 1.0, 0.1),
        seed=2,
    )
    scene = concat_objects([rear, front])

    width = height = 128
    viewmat = look_at_viewmat(
        eye=np.array([0.0, 0.0, 2.5], dtype=np.float64),
        target=np.array([0.0, 0.0, 0.15], dtype=np.float64),
    )
    k = intrinsics_from_fov(width, height, 60.0)
    out = render(scene, viewmat, k, width, height)

    dominant_from_weights = out.object_weights.argmax(dim=-1)
    fg = out.object_id_map >= 0
    assert torch.equal(out.object_id_map[fg], dominant_from_weights[fg])
