# Adapted from EasyGaussianSplatting (https://github.com/scomup/EasyGaussianSplatting)
# Front-to-back Gaussian splat rasterizer (PyTorch CPU).

from __future__ import annotations

from dataclasses import dataclass

import torch

import event_log
from ply_loader import SceneGaussians
from render.projection import (
    compute_cov_2d,
    compute_cov_3d,
    inverse_cov2d,
    project_gaussians,
)
from render.sh import camera_position_from_viewmat, eval_sh_view

BACKGROUND_ID = -1


@dataclass
class RenderOutput:
    fg_rgb: torch.Tensor
    alpha: torch.Tensor
    object_id_map: torch.Tensor


def render(
    gaussians: SceneGaussians,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 0,
    near: float = 0.2,
    far: float = 100.0,
    verbose: bool = False,
) -> RenderOutput:
    """Rasterize Gaussians to foreground buffers (no background composite)."""
    device = gaussians.means.device
    dtype = torch.float32

    viewmat = viewmat.to(device=device, dtype=dtype)
    k = k.to(device=device, dtype=dtype)

    uv, points_cam, depth = project_gaussians(gaussians.means, viewmat, k)
    cov3d = compute_cov_3d(gaussians.scales, gaussians.quats)
    cov2d = compute_cov_2d(points_cam, cov3d, viewmat, k, width, height)
    cinv2d, areas = inverse_cov2d(cov2d)

    cam_pos = camera_position_from_viewmat(viewmat)
    colors = eval_sh_view(
        gaussians.sh_dc,
        gaussians.sh_rest,
        gaussians.means,
        cam_pos,
        sh_degree=sh_degree,
    )

    fg_rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
    transmittance = torch.ones(height, width, device=device, dtype=dtype)
    object_id_map = torch.full(
        (height, width),
        BACKGROUND_ID,
        device=device,
        dtype=torch.int32,
    )
    sort_idx = torch.argsort(depth)
    win_size = torch.tensor([width, height], device=device, dtype=dtype)
    n_sorted = sort_idx.numel()
    report_every_ui = max(1, n_sorted // 100)
    report_every_log = max(1, n_sorted // 20)

    yy = torch.arange(height, device=device, dtype=dtype)
    xx = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    for j, idx in enumerate(sort_idx.tolist()):
        pct = 100.0 * (j + 1) / n_sorted
        if j % report_every_ui == 0 or j == n_sorted - 1:
            if event_log.is_active():
                event_log.render_progress(pct)
        if verbose and (j % report_every_log == 0 or j == n_sorted - 1):
            pct_label = f"{pct:5.1f}% ({j + 1}/{n_sorted})"
            if event_log.is_active():
                event_log.log(f"[dim]rasterize[/] {pct_label}")
            else:
                print(f"  rasterize {pct_label}", flush=True)

        d = depth[idx].item()
        if d < near or d > far:
            continue

        u = uv[idx]
        if torch.any(torch.abs(u / win_size) > 1.3):
            continue

        r = areas[idx]
        x0 = int(max(min(u[0] - r[0], width), 0))
        x1 = int(max(min(u[0] + r[0], width), 0))
        y0 = int(max(min(u[1] - r[1], height), 0))
        y1 = int(max(min(u[1] + r[1], height), 0))
        if (x1 - x0) * (y1 - y0) == 0:
            continue

        cinv = cinv2d[idx]
        opa = gaussians.opacities[idx].item()
        patch_color = colors[idx]

        px = grid_x[y0:y1, x0:x1] - u[0]
        py = grid_y[y0:y1, x0:x1] - u[1]
        maha = cinv[0] * px * px + cinv[2] * py * py + 2.0 * cinv[1] * px * py
        patch_alpha = torch.exp(-0.5 * maha) * opa
        patch_alpha = patch_alpha.clamp(max=0.99)

        t_patch = transmittance[y0:y1, x0:x1]
        weight = patch_alpha * t_patch
        oid = int(gaussians.object_ids[idx].item())

        fg_rgb[y0:y1, x0:x1, :] += weight.unsqueeze(-1) * patch_color
        # Front-most hit: first gaussian along the view ray that contributes at this
        # pixel owns the object id (raycast semantics). Do not let deeper splats
        # overwrite — max(alpha*T) wrongly assigned rear objects on semi-transparent
        # foreground pixels.
        id_patch = object_id_map[y0:y1, x0:x1]
        first_hit = (id_patch == BACKGROUND_ID) & (patch_alpha > 1e-4)
        object_id_map[y0:y1, x0:x1] = torch.where(
            first_hit,
            torch.full_like(id_patch, oid),
            id_patch,
        )
        transmittance[y0:y1, x0:x1] = t_patch * (1.0 - patch_alpha)

    alpha = (1.0 - transmittance).clamp(0.0, 1.0)
    return RenderOutput(fg_rgb=fg_rgb, alpha=alpha, object_id_map=object_id_map)
