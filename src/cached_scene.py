"""Scene graph referencing shared PLY cache (no worker-side Gaussian copies)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ply_loader import SceneGaussians

import ply_manager


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


@dataclass
class ObjectPlacement:
    """One placed object; Gaussian tensors may live in the shared PLY cache."""

    gaussians: SceneGaussians
    pick: np.ndarray | None
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    object_id: int
    ply_path: Path | None
    from_cache: bool
    local_lo: np.ndarray
    local_hi: np.ndarray
    q_rot: torch.Tensor = field(repr=False)
    rot_t: torch.Tensor = field(repr=False)

    @property
    def num_gaussians(self) -> int:
        if self.pick is not None:
            return int(self.pick.shape[0])
        return self.gaussians.num_gaussians

    def iter_indices(self) -> list[int]:
        if self.pick is not None:
            return self.pick.tolist()
        return list(range(self.gaussians.num_gaussians))

    def transform_mean(self, idx: int) -> torch.Tensor:
        mean = self.gaussians.means[idx]
        return mean @ self.rot_t * self.scale + self.translation

    def transform_quat(self, idx: int) -> torch.Tensor:
        return _quat_multiply(self.q_rot, self.gaussians.quats[idx])

    def transform_scale(self, idx: int) -> torch.Tensor:
        return self.gaussians.scales[idx] * self.scale


@dataclass
class CachedScene:
    """Multi-object scene that reads Gaussian data from cache-backed tensors."""

    objects: list[ObjectPlacement]

    @property
    def num_gaussians(self) -> int:
        return sum(obj.num_gaussians for obj in self.objects)

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        corners_world: list[np.ndarray] = []
        for obj in self.objects:
            for lx in (obj.local_lo[0], obj.local_hi[0]):
                for ly in (obj.local_lo[1], obj.local_hi[1]):
                    for lz in (obj.local_lo[2], obj.local_hi[2]):
                        local = torch.tensor([lx, ly, lz], dtype=torch.float32)
                        world = local @ obj.rot_t * obj.scale + obj.translation
                        corners_world.append(world.detach().cpu().numpy())
        pts = np.stack(corners_world, axis=0)
        return pts.min(axis=0), pts.max(axis=0)

    def release_cache_refs(self) -> None:
        for obj in self.objects:
            if obj.from_cache and obj.ply_path is not None:
                ply_manager.release(obj.ply_path)


def make_placement(
    gaussians: SceneGaussians,
    *,
    pick: np.ndarray | None,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
    object_id: int,
    ply_path: Path | None,
    from_cache: bool,
    local_lo: np.ndarray,
    local_hi: np.ndarray,
) -> ObjectPlacement:
    rot_t = torch.from_numpy(rotation.astype(np.float32))
    trans_t = torch.from_numpy(translation.astype(np.float32))
    q_rot = _matrix_to_quat_wxyz(rotation)
    return ObjectPlacement(
        gaussians=gaussians,
        pick=pick,
        rotation=rotation,
        translation=trans_t,
        scale=scale,
        object_id=object_id,
        ply_path=ply_path,
        from_cache=from_cache,
        local_lo=local_lo,
        local_hi=local_hi,
        q_rot=q_rot,
        rot_t=rot_t,
    )


def euler_placement(
    gaussians: SceneGaussians,
    euler: np.ndarray,
    placement: np.ndarray,
    scale: float,
    object_id: int,
    *,
    pick: np.ndarray | None,
    ply_path: Path | None,
    from_cache: bool,
    local_lo: np.ndarray,
    local_hi: np.ndarray,
) -> ObjectPlacement:
    rotation = _euler_matrix_xyz(*euler)
    return make_placement(
        gaussians,
        pick=pick,
        rotation=rotation,
        translation=placement,
        scale=scale,
        object_id=object_id,
        ply_path=ply_path,
        from_cache=from_cache,
        local_lo=local_lo,
        local_hi=local_hi,
    )
