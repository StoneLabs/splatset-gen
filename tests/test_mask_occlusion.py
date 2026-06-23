"""Occlusion-aware mask tests."""

from __future__ import annotations

import numpy as np
import torch

from camera import camera_from_orbit, intrinsics_from_fov, look_at_viewmat
from picker import object_mask, sample_click
from render import render
from synthetic_gaussians import concat_objects, make_object_blob


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
    rear_mask = object_mask(out.object_id_map, clicked_object_id=0)

    assert rear_mask[occluded_by_front].max().item() == 0
    assert rear_mask[(out.object_id_map == 0)].min().item() == 255

    rng = np.random.default_rng(0)
    x, y, clicked_id = sample_click(out.alpha, out.object_id_map, 0.5, rng)
    mask = object_mask(out.object_id_map, clicked_id)
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

    front_mask = object_mask(out.object_id_map, clicked_object_id=1)
    visible_front = out.object_id_map == 1
    assert front_mask[visible_front].min().item() == 255
