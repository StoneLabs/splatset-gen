"""Load standard 3D Gaussian Splatting PLY files into SceneGaussians."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

REQUIRED_FIELDS = (
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
)


@dataclass(frozen=True)
class PlyLoadStats:
    """Bounding-box size before/after unit-extent normalization."""

    extent_before: tuple[float, float, float]
    max_extent_before: float
    extent_after: tuple[float, float, float]
    max_extent_after: float


@dataclass
class SceneGaussians:
    """Concatenated scene Gaussians with per-point object assignment."""

    means: torch.Tensor  # [N, 3]
    quats: torch.Tensor  # [N, 4] wxyz
    scales: torch.Tensor  # [N, 3] exp-activated
    opacities: torch.Tensor  # [N] sigmoid-activated
    sh_dc: torch.Tensor  # [N, 3] degree-0 SH coefficients
    sh_rest: torch.Tensor  # [N, K, 3] higher-order SH (K=0 if absent)
    object_ids: torch.Tensor  # [N] int, PLY instance per Gaussian

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @property
    def num_objects(self) -> int:
        if self.object_ids.numel() == 0:
            return 0
        return int(self.object_ids.max().item()) + 1

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned min/max over Gaussian means."""
        pts = self.means.detach().cpu().numpy()
        return pts.min(axis=0), pts.max(axis=0)


def _require_fields(vertex_data, path: Path) -> None:
    names = set(vertex_data.dtype.names or ())
    missing = [field for field in REQUIRED_FIELDS if field not in names]
    if missing:
        raise ValueError(
            f"{path} is not a valid 3DGS PLY: missing fields {missing}. "
            f"Expected standard 3DGS vertex properties."
        )


def _stack_field(vertex_data, prefix: str, count: int) -> np.ndarray:
    return np.stack([vertex_data[f"{prefix}_{i}"] for i in range(count)], axis=-1)


def _normalize_to_unit_extent(
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, PlyLoadStats]:
    """Center at bbox midpoint and scale so the longest axis spans 1 unit."""
    lo = means.min(axis=0)
    hi = means.max(axis=0)
    extent_before = (hi - lo).astype(np.float64)
    max_before = float(extent_before.max())
    center = (lo + hi) / 2.0
    if max_before < 1e-8:
        centered = means - center
        lo_a = centered.min(axis=0)
        hi_a = centered.max(axis=0)
        extent_after = (hi_a - lo_a).astype(np.float64)
        stats = PlyLoadStats(
            extent_before=(float(extent_before[0]), float(extent_before[1]), float(extent_before[2])),
            max_extent_before=max_before,
            extent_after=(float(extent_after[0]), float(extent_after[1]), float(extent_after[2])),
            max_extent_after=float(extent_after.max()),
        )
        return centered, scales, stats
    s = 1.0 / max_before
    means_out = (means - center) * s
    scales_out = scales * s
    lo_a = means_out.min(axis=0)
    hi_a = means_out.max(axis=0)
    extent_after = (hi_a - lo_a).astype(np.float64)
    stats = PlyLoadStats(
        extent_before=(float(extent_before[0]), float(extent_before[1]), float(extent_before[2])),
        max_extent_before=max_before,
        extent_after=(float(extent_after[0]), float(extent_after[1]), float(extent_after[2])),
        max_extent_after=float(extent_after.max()),
    )
    return means_out, scales_out, stats


def format_extent(stats: PlyLoadStats) -> str:
    """Human-readable max bbox extent before→after normalization."""
    return f"{stats.max_extent_before:.3f}→{stats.max_extent_after:.3f}"


def load_ply(
    path: str | Path,
    max_gaussians: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SceneGaussians, PlyLoadStats]:
    """Load one 3DGS PLY file as a single-object SceneGaussians (object_id=0).

    Gaussians are centered at the bounding-box midpoint and scaled so the longest
    axis of the axis-aligned bounds is 1 unit (Gaussian scales are scaled too).

    Returns ``(gaussians, load_stats)`` where ``load_stats`` records bbox extent
    before and after normalization.

    When ``max_gaussians`` is set and the PLY has more points, a random subset is
    taken (without replacement). Pass ``rng`` for reproducible subsampling.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"{path} has no vertex element")

    vertex = ply["vertex"].data
    _require_fields(vertex, path)

    if max_gaussians is not None and len(vertex) > max_gaussians:
        if rng is None:
            rng = np.random.default_rng()
        pick = rng.choice(len(vertex), size=max_gaussians, replace=False)
        vertex = vertex[pick]

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    raw_opacity = vertex["opacity"].astype(np.float32)
    raw_scale = _stack_field(vertex, "scale", 3).astype(np.float32)
    raw_quat = _stack_field(vertex, "rot", 4).astype(np.float32)
    sh_dc = _stack_field(vertex, "f_dc", 3).astype(np.float32)

    rest_names = sorted(
        name for name in (vertex.dtype.names or ()) if name.startswith("f_rest_")
    )
    if rest_names:
        rest_flat = np.stack([vertex[name] for name in rest_names], axis=-1).astype(
            np.float32
        )
        if rest_flat.shape[1] % 3 != 0:
            raise ValueError(
                f"{path}: f_rest_* count {rest_flat.shape[1]} is not divisible by 3"
            )
        sh_rest = rest_flat.reshape(rest_flat.shape[0], rest_flat.shape[1] // 3, 3)
    else:
        sh_rest = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    opacities = 1.0 / (1.0 + np.exp(-raw_opacity))
    scales = np.exp(raw_scale)
    quat_norm = np.linalg.norm(raw_quat, axis=-1, keepdims=True)
    quat_norm = np.maximum(quat_norm, 1e-8)
    quats = raw_quat / quat_norm
    means, scales, load_stats = _normalize_to_unit_extent(means, scales)

    n = means.shape[0]
    object_ids = np.zeros(n, dtype=np.int64)

    return (
        SceneGaussians(
            means=torch.from_numpy(means),
            quats=torch.from_numpy(quats),
            scales=torch.from_numpy(scales),
            opacities=torch.from_numpy(opacities),
            sh_dc=torch.from_numpy(sh_dc),
            sh_rest=torch.from_numpy(sh_rest),
            object_ids=torch.from_numpy(object_ids),
        ),
        load_stats,
    )


def write_synthetic_ply(path: str | Path, num_gaussians: int = 4) -> Path:
    """Write minimal 3DGS PLY for tests (raw log-scale / logit opacity)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(num_gaussians, 3)).astype(np.float32)
    raw_opacity = np.zeros(num_gaussians, dtype=np.float32)
    raw_scale = np.log(np.full((num_gaussians, 3), 0.05, dtype=np.float32))
    quats = np.zeros((num_gaussians, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    sh_dc = rng.normal(size=(num_gaussians, 3)).astype(np.float32) * 0.1

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    rows = np.empty(num_gaussians, dtype=dtype)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rows["opacity"] = raw_opacity
    rows["scale_0"], rows["scale_1"], rows["scale_2"] = (
        raw_scale[:, 0],
        raw_scale[:, 1],
        raw_scale[:, 2],
    )
    rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = (
        quats[:, 0],
        quats[:, 1],
        quats[:, 2],
        quats[:, 3],
    )
    rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]

    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(str(path))
    return path
