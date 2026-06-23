"""Single-object mask equals visible object pixels."""

from __future__ import annotations

import numpy as np

from camera import camera_from_orbit
from picker import object_mask, sample_click
from render import render
from synthetic_gaussians import make_object_blob

MASK_WEIGHT_THRESHOLD = 0.05


def test_single_object_mask_matches_visible_region() -> None:
    obj = make_object_blob(0, center=(0.0, 0.0, 0.0), num_gaussians=60)
    width = height = 128
    lo, hi = obj.bounds()
    viewmat, k, w, h, _ = camera_from_orbit((lo, hi), width=width, height=height)

    out = render(obj, viewmat, k, w, h)
    mask = object_mask(
        out.object_weights, clicked_object_id=0, weight_threshold=MASK_WEIGHT_THRESHOLD
    )

    visible_object = (out.object_weights[:, :, 0] > MASK_WEIGHT_THRESHOLD) & (
        out.object_id_map == 0
    )
    assert mask[visible_object].min().item() == 255
    assert mask[~visible_object].max().item() == 0

    rng = np.random.default_rng(7)
    x, y, oid = sample_click(out.alpha, out.object_id_map, 0.5, rng)
    assert oid == 0
    assert mask[y, x].item() == 255
