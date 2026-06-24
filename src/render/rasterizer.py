# Adapted from EasyGaussianSplatting (https://github.com/scomup/EasyGaussianSplatting)
# Front-to-back Gaussian splat rasterizer (PyTorch CPU).

from __future__ import annotations

from dataclasses import dataclass

import torch

import event_log
from cached_scene import CachedScene, ObjectPlacement
from camera import transform_world_to_camera, project_points
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
    object_weights: torch.Tensor  # [H, W, num_objects] front-to-back weight per object


def render(
    scene: SceneGaussians | CachedScene,
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
    if isinstance(scene, CachedScene):
        return _render_cached(
            scene,
            viewmat,
            k,
            width,
            height,
            sh_degree=sh_degree,
            near=near,
            far=far,
            verbose=verbose,
        )
    return _render_owned(
        scene,
        viewmat,
        k,
        width,
        height,
        sh_degree=sh_degree,
        near=near,
        far=far,
        verbose=verbose,
    )


def _render_owned(
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
    fg_rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
    transmittance = torch.ones(height, width, device=device, dtype=dtype)
    object_weights = torch.zeros(
        height, width, max(num_objects, 1), device=device, dtype=dtype
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
                if int(pct) % 25 == 0:
                    event_log.report_memory()
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
        ow_patch = object_weights[y0:y1, x0:x1, oid]
        object_weights[y0:y1, x0:x1, oid] = ow_patch + weight
        transmittance[y0:y1, x0:x1] = t_patch * (1.0 - patch_alpha)

    return _finalize_output(
        fg_rgb, transmittance, object_weights, num_objects, device, dtype
    )


def _collect_sorted_splats(
    scene: CachedScene,
    viewmat: torch.Tensor,
    k: torch.Tensor,
) -> list[tuple[float, ObjectPlacement, int]]:
    entries: list[tuple[float, ObjectPlacement, int]] = []
    for obj in scene.objects:
        for gi in obj.iter_indices():
            mean_w = obj.transform_mean(gi)
            cam = transform_world_to_camera(mean_w.unsqueeze(0), viewmat)[0]
            if cam[2].item() <= 1e-4:
                continue
            depth = cam[2].item()
            entries.append((depth, obj, gi))
    entries.sort(key=lambda item: item[0])
    return entries


def _render_cached(
    scene: CachedScene,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 0,
    near: float = 0.2,
    far: float = 100.0,
    verbose: bool = False,
) -> RenderOutput:
    device = torch.device("cpu")
    dtype = torch.float32
    viewmat = viewmat.to(device=device, dtype=dtype)
    k = k.to(device=device, dtype=dtype)
    cam_pos = camera_position_from_viewmat(viewmat)

    sorted_splats = _collect_sorted_splats(scene, viewmat, k)
    num_objects = scene.num_objects
    fg_rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
    transmittance = torch.ones(height, width, device=device, dtype=dtype)
    object_weights = torch.zeros(
        height, width, max(num_objects, 1), device=device, dtype=dtype
    )
    n_sorted = len(sorted_splats)
    report_every_ui = max(1, n_sorted // 100)
    report_every_log = max(1, n_sorted // 20)
    win_size = torch.tensor([width, height], device=device, dtype=dtype)

    yy = torch.arange(height, device=device, dtype=dtype)
    xx = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    for j, (depth, obj, gi) in enumerate(sorted_splats):
        pct = 100.0 * (j + 1) / max(n_sorted, 1)
        if j % report_every_ui == 0 or j == n_sorted - 1:
            if event_log.is_active():
                event_log.render_progress(pct)
                if int(pct) % 25 == 0:
                    event_log.report_memory()
        if verbose and (j % report_every_log == 0 or j == n_sorted - 1):
            pct_label = f"{pct:5.1f}% ({j + 1}/{n_sorted})"
            if event_log.is_active():
                event_log.log(f"[dim]rasterize[/] {pct_label}")
            else:
                print(f"  rasterize {pct_label}", flush=True)

        if depth < near or depth > far:
            continue

        mean_w = obj.transform_mean(gi)
        points_cam = transform_world_to_camera(mean_w.unsqueeze(0), viewmat)
        uv, _ = project_points(points_cam, k)
        u = uv[0]

        scale = obj.transform_scale(gi).unsqueeze(0)
        quat = obj.transform_quat(gi).unsqueeze(0)
        cov3d = compute_cov_3d(scale, quat)
        cov2d = compute_cov_2d(points_cam, cov3d, viewmat, k, width, height)
        cinv2d, areas = inverse_cov2d(cov2d)

        if torch.any(torch.abs(u / win_size) > 1.3):
            continue

        r = areas[0]
        x0 = int(max(min(u[0] - r[0], width), 0))
        x1 = int(max(min(u[0] + r[0], width), 0))
        y0 = int(max(min(u[1] - r[1], height), 0))
        y1 = int(max(min(u[1] + r[1], height), 0))
        if (x1 - x0) * (y1 - y0) == 0:
            continue

        cinv = cinv2d[0]
        opa = obj.gaussians.opacities[gi].item()
        color = eval_sh_view(
            obj.gaussians.sh_dc[gi].unsqueeze(0),
            obj.gaussians.sh_rest[gi].unsqueeze(0),
            mean_w.unsqueeze(0),
            cam_pos,
            sh_degree=sh_degree,
        )[0]

        px = grid_x[y0:y1, x0:x1] - u[0]
        py = grid_y[y0:y1, x0:x1] - u[1]
        maha = cinv[0] * px * px + cinv[2] * py * py + 2.0 * cinv[1] * px * py
        patch_alpha = torch.exp(-0.5 * maha) * opa
        patch_alpha = patch_alpha.clamp(max=0.99)

        t_patch = transmittance[y0:y1, x0:x1]
        weight = patch_alpha * t_patch
        oid = obj.object_id

        fg_rgb[y0:y1, x0:x1, :] += weight.unsqueeze(-1) * color
        ow_patch = object_weights[y0:y1, x0:x1, oid]
        object_weights[y0:y1, x0:x1, oid] = ow_patch + weight
        transmittance[y0:y1, x0:x1] = t_patch * (1.0 - patch_alpha)

    return _finalize_output(
        fg_rgb, transmittance, object_weights, num_objects, device, dtype
    )


def _finalize_output(
    fg_rgb: torch.Tensor,
    transmittance: torch.Tensor,
    object_weights: torch.Tensor,
    num_objects: int,
    device: torch.device,
    dtype: torch.dtype,
) -> RenderOutput:
    alpha = (1.0 - transmittance).clamp(0.0, 1.0)
    if num_objects == 0:
        object_id_map = torch.full(
            (fg_rgb.shape[0], fg_rgb.shape[1]), BACKGROUND_ID, device=device, dtype=torch.int32
        )
        object_weights = object_weights[:, :, :0]
    else:
        object_weights = object_weights[:, :, :num_objects]
        total_object_weight = object_weights.sum(dim=-1)
        object_id_map = object_weights.argmax(dim=-1).to(torch.int32)
        object_id_map = torch.where(
            total_object_weight > 1e-6,
            object_id_map,
            torch.full_like(object_id_map, BACKGROUND_ID),
        )
    return RenderOutput(
        fg_rgb=fg_rgb,
        alpha=alpha,
        object_id_map=object_id_map,
        object_weights=object_weights,
    )
