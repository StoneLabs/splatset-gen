# Adapted from EasyGaussianSplatting (https://github.com/scomup/EasyGaussianSplatting)
# SH evaluation for Gaussian splat colors.

from __future__ import annotations

import torch

SH_C0 = 0.28209479177387814


def eval_sh_dc(sh_dc: torch.Tensor) -> torch.Tensor:
    """Evaluate degree-0 SH → RGB in [0, 1] (clamped)."""
    rgb = SH_C0 * sh_dc + 0.5
    return rgb.clamp(0.0, 1.0)


def eval_sh_view(
    sh_dc: torch.Tensor,
    sh_rest: torch.Tensor,
    means: torch.Tensor,
    camera_pos: torch.Tensor,
    sh_degree: int = 0,
) -> torch.Tensor:
    """Evaluate SH color; v1 uses degree 0 only."""
    if sh_degree > 0 and sh_rest.numel() > 0:
        raise NotImplementedError("Higher-order SH is v1.1; use sh_degree=0")
    return eval_sh_dc(sh_dc)


def camera_position_from_viewmat(viewmat: torch.Tensor) -> torch.Tensor:
    """Recover world-space camera position from world-to-camera viewmat."""
    r = viewmat[:3, :3]
    t = viewmat[:3, 3]
    return -(t @ r)
