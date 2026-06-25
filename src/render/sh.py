# Adapted from EasyGaussianSplatting (https://github.com/scomup/EasyGaussianSplatting)
# SH evaluation for Gaussian splat colors.

from __future__ import annotations

import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
SH_C4 = (
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
)

MAX_SH_DEGREE = 3


def validate_sh_degree(sh_degree: int) -> None:
    """Reject config values outside supported 3DGS SH range."""
    if sh_degree < 0 or sh_degree > MAX_SH_DEGREE:
        raise ValueError(f"render.sh_degree must be 0–{MAX_SH_DEGREE}, got {sh_degree}")


def eval_sh_dc(sh_dc: torch.Tensor) -> torch.Tensor:
    """Evaluate degree-0 SH → RGB in [0, 1] (clamped)."""
    rgb = SH_C0 * sh_dc + 0.5
    return rgb.clamp(0.0, 1.0)


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate SH at unit directions.

    Args:
        deg: active SH degree (0–3 supported).
        sh: [..., 3, (deg+1)^2] RGB coefficients, channel-first.
        dirs: [..., 3] unit viewing directions (3DGS: gaussian − camera).
    Returns:
        Linear RGB before +0.5 offset, shape [..., 3].
    """
    if deg < 0 or deg > MAX_SH_DEGREE:
        raise ValueError(f"sh_degree must be 0–{MAX_SH_DEGREE}, got {deg}")

    expected = (deg + 1) ** 2
    if sh.shape[-1] < expected:
        raise ValueError(
            f"sh_degree={deg} needs {expected} coeffs per channel, got {sh.shape[-1]}"
        )

    result = SH_C0 * sh[..., 0]
    if deg == 0:
        return result

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    result = result - SH_C1 * y * sh[..., 1] + SH_C1 * z * sh[..., 2] - SH_C1 * x * sh[..., 3]

    if deg == 1:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + SH_C2[0] * xy * sh[..., 4]
        + SH_C2[1] * yz * sh[..., 5]
        + SH_C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
        + SH_C2[3] * xz * sh[..., 7]
        + SH_C2[4] * (xx - yy) * sh[..., 8]
    )

    if deg == 2:
        return result

    result = (
        result
        + SH_C3[0] * y * (3.0 * xx - yy) * sh[..., 9]
        + SH_C3[1] * xy * z * sh[..., 10]
        + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[..., 11]
        + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[..., 12]
        + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[..., 13]
        + SH_C3[5] * z * (xx - yy) * sh[..., 14]
        + SH_C3[6] * x * (xx - 3.0 * yy) * sh[..., 15]
    )
    return result


def _effective_sh_degree(sh_degree: int, num_rest: int) -> int:
    """Cap requested degree by available ``sh_rest`` rows."""
    if sh_degree <= 0 or num_rest <= 0:
        return 0
    for deg in range(sh_degree, 0, -1):
        if num_rest >= (deg + 1) ** 2 - 1:
            return deg
    return 0


def _pack_sh_coeffs(sh_dc: torch.Tensor, sh_rest: torch.Tensor) -> torch.Tensor:
    """Stack DC + rest into channel-first [N, 3, K] layout."""
    return torch.cat([sh_dc.unsqueeze(-1), sh_rest.permute(0, 2, 1)], dim=-1)


def eval_sh_view(
    sh_dc: torch.Tensor,
    sh_rest: torch.Tensor,
    means: torch.Tensor,
    camera_pos: torch.Tensor,
    sh_degree: int = 0,
) -> torch.Tensor:
    """Evaluate view-dependent SH color for each Gaussian."""
    eff_deg = _effective_sh_degree(sh_degree, sh_rest.shape[1] if sh_rest.ndim == 3 else 0)
    if eff_deg <= 0:
        return eval_sh_dc(sh_dc)

    dirs = means - camera_pos.unsqueeze(0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    num_coeffs = (eff_deg + 1) ** 2 - 1
    sh = _pack_sh_coeffs(sh_dc, sh_rest[:, :num_coeffs])
    rgb = eval_sh(eff_deg, sh, dirs) + 0.5
    return rgb.clamp(0.0, 1.0)


def camera_position_from_viewmat(viewmat: torch.Tensor) -> torch.Tensor:
    """Recover world-space camera position from world-to-camera viewmat."""
    r = viewmat[:3, :3]
    t = viewmat[:3, 3]
    return -(t @ r)
