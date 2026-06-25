# Adapted from EasyGaussianSplatting (https://github.com/scomup/EasyGaussianSplatting)
# Front-to-back Gaussian splat rasterizer (PyTorch CPU).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
_TRANSMITTANCE_MIN = 1e-4


@dataclass
class RenderOutput:
    fg_rgb: torch.Tensor
    alpha: torch.Tensor
    object_id_map: torch.Tensor
    object_weights: torch.Tensor  # [H, W, num_objects] front-to-back weight per object


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

    num_objects = gaussians.num_objects
    sort_idx = torch.argsort(depth)

    depth_np = depth.numpy()
    uv_np = uv.numpy()
    areas_np = areas.numpy()
    cinv_np = cinv2d.numpy()
    opacities_np = gaussians.opacities.numpy()
    colors_np = colors.numpy()
    object_ids_np = gaussians.object_ids.numpy()
    order = sort_idx.numpy()

    wh = np.array([width, height], dtype=np.float32)
    visible = (depth_np >= near) & (depth_np <= far)
    uv_norm = np.abs(uv_np / wh)
    visible &= (uv_norm[:, 0] <= 1.3) & (uv_norm[:, 1] <= 1.3)
    x0_all = np.clip(uv_np[:, 0] - areas_np[:, 0], 0, width).astype(np.int32)
    x1_all = np.clip(uv_np[:, 0] + areas_np[:, 0], 0, width).astype(np.int32)
    y0_all = np.clip(uv_np[:, 1] - areas_np[:, 1], 0, height).astype(np.int32)
    y1_all = np.clip(uv_np[:, 1] + areas_np[:, 1], 0, height).astype(np.int32)
    visible &= (x1_all - x0_all) * (y1_all - y0_all) > 0
    order = order[visible[order]]

    fg_rgb_np = np.zeros((height, width, 3), dtype=np.float32)
    transmittance_np = np.ones((height, width), dtype=np.float32)
    object_weights_np = np.zeros(
        (height, width, max(num_objects, 1)), dtype=np.float32
    )

    n_sorted = order.size
    report_every_ui = max(1, n_sorted // 20)
    report_every_log = max(1, n_sorted // 20)

    for j, idx in enumerate(order):
        pct = 100.0 * (j + 1) / n_sorted
        if j % report_every_ui == 0 or j == n_sorted - 1:
            if event_log.is_active():
                event_log.render_progress(pct)
        if verbose and (j % report_every_log == 0 or j == n_sorted - 1):
            pct_label = f"{pct:5.1f}% ({j + 1}/{n_sorted})"
            if not event_log.is_active():
                print(f"  rasterize {pct_label}", flush=True)

        x0 = int(x0_all[idx])
        x1 = int(x1_all[idx])
        y0 = int(y0_all[idx])
        y1 = int(y1_all[idx])

        t_patch = transmittance_np[y0:y1, x0:x1]
        if t_patch.max() < _TRANSMITTANCE_MIN:
            continue

        u0 = uv_np[idx, 0]
        u1 = uv_np[idx, 1]
        c0, c1, c2 = cinv_np[idx]
        opa = opacities_np[idx]
        oid = int(object_ids_np[idx])
        patch_color = colors_np[idx]

        xs = np.arange(x0, x1, dtype=np.float32) - u0
        ys = np.arange(y0, y1, dtype=np.float32) - u1
        px = xs[None, :]
        py = ys[:, None]
        maha = c0 * px * px + c2 * py * py + 2.0 * c1 * px * py
        patch_alpha = np.exp(-0.5 * maha, dtype=np.float32) * opa
        np.clip(patch_alpha, None, 0.99, out=patch_alpha)

        weight = patch_alpha * t_patch
        fg_rgb_np[y0:y1, x0:x1] += weight[..., None] * patch_color
        object_weights_np[y0:y1, x0:x1, oid] += weight
        transmittance_np[y0:y1, x0:x1] = t_patch * (1.0 - patch_alpha)

    alpha_np = np.clip(1.0 - transmittance_np, 0.0, 1.0)
    if num_objects == 0:
        object_id_map_np = np.full((height, width), BACKGROUND_ID, dtype=np.int32)
        object_weights_np = object_weights_np[:, :, :0]
    else:
        object_weights_np = object_weights_np[:, :, :num_objects]
        total_object_weight = object_weights_np.sum(axis=-1)
        object_id_map_np = object_weights_np.argmax(axis=-1).astype(np.int32)
        object_id_map_np = np.where(
            total_object_weight > 1e-6,
            object_id_map_np,
            BACKGROUND_ID,
        )

    fg_rgb = torch.from_numpy(fg_rgb_np)
    alpha = torch.from_numpy(alpha_np)
    object_id_map = torch.from_numpy(object_id_map_np)
    object_weights = torch.from_numpy(object_weights_np)
    return RenderOutput(
        fg_rgb=fg_rgb,
        alpha=alpha,
        object_id_map=object_id_map,
        object_weights=object_weights,
    )
