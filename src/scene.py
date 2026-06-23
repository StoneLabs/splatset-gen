"""Random multi-object scene assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData

import event_log
from ply_loader import SceneGaussians, format_extent, load_ply, release_acquired


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
        opacities=gaussians.opacities.clone(),
        sh_dc=gaussians.sh_dc.clone(),
        sh_rest=gaussians.sh_rest.clone(),
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
    verbose: bool = False,
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

    if verbose and event_log.is_active():
        from collections import Counter

        counts = Counter(p.name for p in chosen)
        summary = ", ".join(
            f"{name}×{count}" if count > 1 else name for name, count in sorted(counts.items())
        )
        cap_note = f" · cap {max_g:,}/ply" if max_g else ""
        event_log.log(
            f"[dim]scene[/] {num_objects} slots from {len(ply_paths)} PLYs{cap_note}: {summary}"
        )

    parts: list[SceneGaussians] = []
    metadata: list[dict[str, Any]] = []

    try:
        for object_id, ply_path in enumerate(chosen):
            n_total: int | None = None
            if verbose and max_g is not None:
                n_total = len(PlyData.read(str(ply_path))["vertex"])

            obj, load_stats = load_ply(ply_path, max_gaussians=max_g, rng=rng)
            n_loaded = obj.num_gaussians

            if verbose and event_log.is_active():
                if n_total is not None and n_total > n_loaded:
                    count_label = f"{n_loaded:,}/{n_total:,} gaussians (subset)"
                else:
                    count_label = f"{n_loaded:,} gaussians"
                extent_label = format_extent(load_stats)
                event_log.log(
                    f"[dim]ply[/] oid={object_id} {ply_path.name} · {count_label} · "
                    f"extent {extent_label}"
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

            if verbose and event_log.is_active():
                rot_deg = np.rad2deg(euler)
                event_log.log(
                    f"[dim]place[/] oid={object_id} "
                    f"pos=[{placement[0]:+.2f},{placement[1]:+.2f},{placement[2]:+.2f}] "
                    f"rot=[{rot_deg[0]:.0f},{rot_deg[1]:.0f},{rot_deg[2]:.0f}]° "
                    f"scale={scale:.2f}"
                )

        return _concat_gaussians(parts), metadata
    finally:
        # Scene tensors are fully owned after concat; drop cache shm mappings early.
        release_acquired()
