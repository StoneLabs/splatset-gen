"""Synthetic Gaussian helpers for render/mask tests."""

from __future__ import annotations

import torch

from ply_loader import SceneGaussians


def make_object_blob(
    object_id: int,
    center: tuple[float, float, float],
    sh_dc: tuple[float, float, float] = (0.2, 0.8, 0.2),
    num_gaussians: int = 40,
    spread: float = 0.06,
    opacity: float = 0.95,
    scale: float = 0.04,
    seed: int = 0,
) -> SceneGaussians:
    """Tight Gaussian blob for unit tests."""
    gen = torch.Generator().manual_seed(seed + object_id * 997)
    center_t = torch.tensor(center, dtype=torch.float32)
    means = center_t + torch.randn(num_gaussians, 3, generator=gen) * spread

    quats = torch.zeros(num_gaussians, 4)
    quats[:, 0] = 1.0
    scales = torch.full((num_gaussians, 3), scale)
    opacities = torch.full((num_gaussians,), opacity)
    sh = torch.tensor(sh_dc, dtype=torch.float32).expand(num_gaussians, 3)
    sh_rest = torch.zeros(num_gaussians, 0, 3)
    object_ids = torch.full((num_gaussians,), object_id, dtype=torch.int64)

    return SceneGaussians(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        sh_dc=sh,
        sh_rest=sh_rest,
        object_ids=object_ids,
    )


def concat_objects(objects: list[SceneGaussians]) -> SceneGaussians:
    return SceneGaussians(
        means=torch.cat([o.means for o in objects]),
        quats=torch.cat([o.quats for o in objects]),
        scales=torch.cat([o.scales for o in objects]),
        opacities=torch.cat([o.opacities for o in objects]),
        sh_dc=torch.cat([o.sh_dc for o in objects]),
        sh_rest=torch.cat([o.sh_rest for o in objects]),
        object_ids=torch.cat([o.object_ids for o in objects]),
    )
