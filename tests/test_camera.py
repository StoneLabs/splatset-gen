"""Camera intrinsics and view matrix tests."""

from __future__ import annotations

from unittest.mock import patch

import camera as camera_mod
import numpy as np
import pytest
import torch
from camera import (
    intrinsics_from_fov,
    look_at_viewmat,
    sample_random_camera,
    transform_world_to_camera,
)


def test_intrinsics_match_fov() -> None:
    w, h, fov = 512, 512, 60.0
    k = intrinsics_from_fov(w, h, fov)
    fy = k[1, 1].item()
    expected_fy = h / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    assert abs(fy - expected_fy) < 1e-3
    assert k[0, 0].item() == fy
    assert k[0, 2].item() == w / 2.0
    assert k[1, 2].item() == h / 2.0


def test_viewmat_rotation_is_orthonormal() -> None:
    eye = np.array([3.0, 0.0, 1.0])
    target = np.array([0.0, 0.0, 0.0])
    viewmat = look_at_viewmat(eye, target)
    r = viewmat[:3, :3].numpy()
    identity = r @ r.T
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-5)


def test_transform_places_point_in_front_of_camera() -> None:
    eye = np.array([0.0, 0.0, 5.0])
    target = np.array([0.0, 0.0, 0.0])
    viewmat = look_at_viewmat(eye, target)
    p = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    cam = transform_world_to_camera(p, viewmat)
    assert cam[0, 2].item() > 0.0


def test_sample_random_camera() -> None:
    rng = np.random.default_rng(0)
    bbox_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    bbox_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    cfg = {
        "render": {"width": 128, "height": 128},
        "camera": {
            "fov_deg_range": [50.0, 70.0],
            "distance_range": [4.0, 6.0],
            "max_retries": 20,
        },
    }
    viewmat, k, w, h, fov = sample_random_camera((bbox_min, bbox_max), cfg, rng)
    assert w == 128 and h == 128
    assert viewmat.shape == (4, 4)
    assert k.shape == (3, 3)
    assert 50.0 <= fov <= 70.0


def test_sample_camera_raises_after_retries() -> None:
    rng = np.random.default_rng(0)
    cfg = {
        "render": {"width": 8, "height": 8},
        "camera": {
            "fov_deg_range": [10.0, 10.0],
            "distance_range": [1000.0, 1000.0],
            "max_retries": 2,
        },
    }
    bbox_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    bbox_max = np.array([0.001, 0.001, 0.001], dtype=np.float32)
    with patch.object(camera_mod, "camera_sees_scene", return_value=False):
        with pytest.raises(RuntimeError, match="Failed to sample"):
            sample_random_camera((bbox_min, bbox_max), cfg, rng)
