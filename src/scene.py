"""Random multi-object scene assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ply_loader import SceneGaussians, load_ply


def _euler_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_m @ ry_m @ rx_m


def _matrix_to_quat_wxyz(rot: np.ndarray) -> torch.Tensor:
    m = rot.astype(np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = torch.tensor([w, x, y, z], dtype=torch.float32)
    return q / q.norm()


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    out = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _transform_gaussians(
    gaussians: SceneGaussians,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
    object_id: int,
) -> SceneGaussians:
    rot = torch.from_numpy(rotation.astype(np.float32))
    trans = torch.from_numpy(translation.astype(np.float32))
    q_rot = _matrix_to_quat_wxyz(rotation)

    means = gaussians.means @ rot.T * scale + trans
    quats = _quat_multiply(q_rot.expand_as(gaussians.quats), gaussians.quats)
    scales = gaussians.scales * scale
    object_ids = torch.full_like(gaussians.object_ids, object_id)

    return SceneGaussians(
        means=means,
        quats=quats,
        scales=scales,
        opacities=gaussians.opacities,
        sh_dc=gaussians.sh_dc,
        sh_rest=gaussians.sh_rest,
        object_ids=object_ids,
    )


def _concat_gaussians(parts: list[SceneGaussians]) -> SceneGaussians:
    return SceneGaussians(
        means=torch.cat([p.means for p in parts]),
        quats=torch.cat([p.quats for p in parts]),
        scales=torch.cat([p.scales for p in parts]),
        opacities=torch.cat([p.opacities for p in parts]),
        sh_dc=torch.cat([p.sh_dc for p in parts]),
        sh_rest=torch.cat([p.sh_rest for p in parts]),
        object_ids=torch.cat([p.object_ids for p in parts]),
    )


def build_random_scene(
    ply_paths: list[Path],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[SceneGaussians, list[dict[str, Any]]]:
    """Load, transform, and concatenate random PLY objects into one scene."""
    if not ply_paths:
        raise ValueError("ply_paths is empty")

    scene_cfg = config.get("scene", {})
    n_min = int(scene_cfg.get("num_objects_min", 2))
    n_max = int(scene_cfg.get("num_objects_max", 5))
    pos_range = scene_cfg.get("position_range", [-2.0, 2.0])
    rot_max = float(scene_cfg.get("rotation_deg_max", 180))
    scale_jitter = scene_cfg.get("scale_jitter", [0.8, 1.2])

    max_g = config.get("generation", {}).get("max_gaussians_per_object")

    num_objects = int(rng.integers(n_min, n_max + 1))
    chosen = [Path(rng.choice(ply_paths)) for _ in range(num_objects)]

    parts: list[SceneGaussians] = []
    metadata: list[dict[str, Any]] = []

    for object_id, ply_path in enumerate(chosen):
        obj = load_ply(ply_path, max_gaussians=max_g, rng=rng)
        lo, hi = obj.bounds()
        center = torch.from_numpy(((lo + hi) / 2.0).astype(np.float32))
        obj = SceneGaussians(
            means=obj.means - center,
            quats=obj.quats,
            scales=obj.scales,
            opacities=obj.opacities,
            sh_dc=obj.sh_dc,
            sh_rest=obj.sh_rest,
            object_ids=obj.object_ids,
        )

        placement = rng.uniform(pos_range[0], pos_range[1], size=3).astype(np.float32)
        euler = np.deg2rad(rng.uniform(-rot_max, rot_max, size=3))
        rotation = _euler_matrix_xyz(*euler)
        scale = float(rng.uniform(scale_jitter[0], scale_jitter[1]))

        transformed = _transform_gaussians(obj, rotation, placement, scale, object_id)
        parts.append(transformed)
        metadata.append(
            {
                "object_id": object_id,
                "ply": ply_path.name,
                "transform": {
                    "translation": placement.tolist(),
                    "rotation_euler_deg": np.rad2deg(euler).tolist(),
                    "scale": scale,
                },
            }
        )

    return _concat_gaussians(parts), metadata
