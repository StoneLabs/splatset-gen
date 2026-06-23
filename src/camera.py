"""Pinhole camera sampling (COLMAP / EasyGaussianSplatting convention).

Convention
----------
``viewmat`` is a 4×4 world-to-camera matrix. For a world point ``p_w`` (row vector):

    p_c = p_w @ R^T + t

where ``R = viewmat[:3, :3]`` and ``t = viewmat[:3, 3]``.

Camera coordinates: +X right, +Y down, +Z forward (into scene).
``K`` is the standard pinhole intrinsics matrix (fx, fy, cx, cy).
Pixel ``(x, y)`` uses column ``x``, row ``y``, origin top-left.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def intrinsics_from_fov(
    width: int,
    height: int,
    fov_deg: float,
) -> torch.Tensor:
    """Build [3, 3] intrinsics from vertical FOV (degrees)."""
    fov_rad = np.deg2rad(fov_deg)
    fy = height / (2.0 * np.tan(fov_rad / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return k


def look_at_viewmat(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> torch.Tensor:
    """World-to-camera 4×4 matrix looking from ``eye`` toward ``target``."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("eye and target are too close")
    forward /= forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right_norm = np.linalg.norm(right)
    right /= right_norm

    down = np.cross(forward, right)
    r = np.stack([right, down, forward], axis=0).astype(np.float64)
    t = -r @ eye.astype(np.float64)

    viewmat = np.eye(4, dtype=np.float64)
    viewmat[:3, :3] = r
    viewmat[:3, 3] = t
    return torch.from_numpy(viewmat.astype(np.float32))


def transform_world_to_camera(
    points: torch.Tensor,
    viewmat: torch.Tensor,
) -> torch.Tensor:
    """Transform [N, 3] world points to camera frame."""
    r = viewmat[:3, :3]
    t = viewmat[:3, 3]
    return points @ r.T + t


def project_points(
    points_cam: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-frame points to pixel coords and return (uv, depth)."""
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    u = x * fx / z + cx
    v = y * fy / z + cy
    uv = torch.stack([u, v], dim=-1)
    return uv, z


def _scene_radius(bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    return float(np.linalg.norm(bbox_max - bbox_min) / 2.0)


def camera_sees_scene(
    means: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    margin: float = 0.05,
) -> bool:
    """Return True if projected scene bbox overlaps the image."""
    cam = transform_world_to_camera(means, viewmat)
    if (cam[:, 2] <= 0.01).all():
        return False

    uv, _ = project_points(cam, k)
    u_min, v_min = uv.min(dim=0).values
    u_max, v_max = uv.max(dim=0).values
    w, h = float(width), float(height)
    return not (
        u_max.item() < -margin * w
        or u_min.item() > (1.0 + margin) * w
        or v_max.item() < -margin * h
        or v_min.item() > (1.0 + margin) * h
    )


def sample_random_camera(
    scene_bounds: tuple[np.ndarray, np.ndarray],
    config: dict[str, Any],
    rng: np.random.Generator,
    scene_means: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, float]:
    """Sample viewmat, K, width, height, fov_deg. Respects max_retries."""
    render_cfg = config.get("render", {})
    cam_cfg = config.get("camera", {})

    width = int(render_cfg.get("width", 512))
    height = int(render_cfg.get("height", 512))
    fov_range = cam_cfg.get("fov_deg_range", [45.0, 75.0])
    dist_range = cam_cfg.get("distance_range", [3.0, 8.0])
    max_retries = int(cam_cfg.get("max_retries", 20))
    look_at_jitter = float(cam_cfg.get("look_at_jitter", 0.0))

    bbox_min, bbox_max = scene_bounds
    centroid = (bbox_min + bbox_max) / 2.0
    radius = max(_scene_radius(bbox_min, bbox_max), 0.5)

    for _ in range(max_retries):
        fov_deg = float(rng.uniform(fov_range[0], fov_range[1]))
        distance = float(rng.uniform(dist_range[0], dist_range[1]))
        distance = max(distance, radius * 1.5)

        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        phi = float(rng.uniform(0.25 * np.pi, 0.75 * np.pi))
        offset = np.array(
            [
                distance * np.sin(phi) * np.cos(theta),
                distance * np.sin(phi) * np.sin(theta),
                distance * np.cos(phi),
            ],
            dtype=np.float64,
        )
        eye = centroid + offset
        target = centroid.copy()
        if look_at_jitter > 0:
            target += rng.uniform(-look_at_jitter, look_at_jitter, size=3)

        viewmat = look_at_viewmat(eye, target)
        k = intrinsics_from_fov(width, height, fov_deg)

        check_pts = scene_means if scene_means is not None else torch.from_numpy(
            np.array(
                [
                    bbox_min,
                    bbox_max,
                    [bbox_min[0], bbox_min[1], bbox_max[2]],
                    [bbox_min[0], bbox_max[1], bbox_min[2]],
                    [bbox_max[0], bbox_min[1], bbox_min[2]],
                ],
                dtype=np.float32,
            )
        )
        if camera_sees_scene(check_pts, viewmat, k, width, height):
            return viewmat, k, width, height, fov_deg

    raise RuntimeError(f"Failed to sample valid camera after {max_retries} retries")


def camera_from_orbit(
    scene_bounds: tuple[np.ndarray, np.ndarray],
    width: int = 512,
    height: int = 512,
    fov_deg: float = 60.0,
    distance_scale: float = 2.5,
    azimuth_deg: float = 30.0,
    elevation_deg: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, int, int, float]:
    """Deterministic orbit camera for debug renders."""
    bbox_min, bbox_max = scene_bounds
    centroid = (bbox_min + bbox_max) / 2.0
    radius = max(_scene_radius(bbox_min, bbox_max), 0.5)
    distance = radius * distance_scale

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    offset = np.array(
        [
            distance * np.cos(el) * np.sin(az),
            distance * np.cos(el) * np.cos(az),
            distance * np.sin(el),
        ],
        dtype=np.float64,
    )
    eye = centroid + offset
    viewmat = look_at_viewmat(eye, centroid)
    k = intrinsics_from_fov(width, height, fov_deg)
    return viewmat, k, width, height, fov_deg
