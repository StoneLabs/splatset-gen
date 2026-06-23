"""Generate one training sample: scene → render → click → mask → export."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import event_log
from background import background_from_config, composite
from camera import sample_random_camera
from export import SampleRecord, export_sample
from picker import object_mask, sample_click
from render import render
from render.sh import camera_position_from_viewmat
from scene import build_random_scene


def _vlog(verbose: bool, message: str) -> None:
    if verbose and event_log.is_active():
        event_log.log(message)


def _vstatus(verbose: bool, phase: str, detail: str = "") -> None:
    if verbose and event_log.is_active():
        event_log.worker_status(phase, detail)


def generate_one_sample(
    ply_paths: list[Path],
    config: dict[str, Any],
    rng: np.random.Generator,
    output_dir: Path,
    sample_id: str,
    verbose: bool = False,
    project_root: Path | None = None,
) -> SampleRecord:
    """Build scene, render, pick click, write occlusion-aware mask + RGB."""
    event_log.set_sample(sample_id)
    render_cfg = config.get("render", {})
    sh_degree = int(render_cfg.get("sh_degree", 0))
    alpha_threshold = float(render_cfg.get("alpha_threshold", 0.5))
    max_camera_retries = int(config.get("generation", {}).get("max_camera_retries", 20))

    if verbose:
        pool_workers = int(config.get("_workers", 1))
        torch_threads = max(1, (os.cpu_count() or 1) // max(pool_workers, 1))
        _vlog(verbose, f"[dim]worker[/] {torch_threads} torch threads")

    _vstatus(verbose, "scene", "building…")
    scene, objects_meta = build_random_scene(ply_paths, config, rng, verbose=verbose)
    lo, hi = scene.bounds()
    background = background_from_config(config, base_dir=project_root)
    scene_detail = (
        f"{scene.num_objects} obj · {scene.num_gaussians:,} gauss · "
        f"[{lo[0]:.1f},{lo[1]:.1f},{lo[2]:.1f}]→[{hi[0]:.1f},{hi[1]:.1f},{hi[2]:.1f}]"
    )
    _vstatus(verbose, "scene", scene_detail)
    _vlog(verbose, f"[dim]scene[/] {scene_detail}")

    last_error: Exception | None = None
    for attempt in range(max_camera_retries):
        try:
            _vstatus(verbose, "camera", f"attempt {attempt + 1}/{max_camera_retries}")
            viewmat, k, width, height, fov_deg = sample_random_camera(
                (lo, hi), config, rng, scene_means=scene.means
            )
            cam_detail = f"{width}×{height} · fov {fov_deg:.1f}°"
            _vstatus(verbose, "camera", cam_detail)
            cam_pos = camera_position_from_viewmat(viewmat).detach().cpu().numpy()
            centroid = (lo + hi) / 2.0
            dist = float(np.linalg.norm(cam_pos - centroid))
            _vlog(
                verbose,
                f"[dim]camera[/] {cam_detail} · dist={dist:.2f} · "
                f"eye=[{cam_pos[0]:+.2f},{cam_pos[1]:+.2f},{cam_pos[2]:+.2f}]",
            )

            _vstatus(
                verbose,
                "rasterize",
                f"{scene.num_gaussians:,} gauss · {width}×{height} · sh_degree={sh_degree}",
            )
            out = render(
                scene,
                viewmat,
                k,
                width,
                height,
                sh_degree=sh_degree,
                verbose=verbose,
            )
            alpha_max = float(out.alpha.max().item())
            fg_pixels = int((out.alpha > alpha_threshold).sum().item())
            _vstatus(
                verbose,
                "rasterize",
                f"done · α_max={alpha_max:.3f} · fg={fg_pixels:,}px",
            )
            _vlog(
                verbose,
                f"[dim]rasterize[/] α_max={alpha_max:.3f} fg_pixels={fg_pixels:,}",
            )
            if out.alpha.max() < alpha_threshold:
                raise ValueError("Render has insufficient foreground coverage")

            if background.mode == "image":
                bg_hint = f"random from {background.image_dir.name if background.image_dir else '?'}"
            else:
                bg_hint = f"solid {list(background.solid_color)}"
            _vstatus(verbose, "composite", bg_hint)
            rgb, bg_meta = composite(
                out.fg_rgb, out.alpha, background, width, height, rng=rng
            )
            if bg_meta.get("mode") == "image":
                comp_detail = f"{bg_meta.get('image', '?')} · {bg_meta.get('resize_mode', 'crop')}"
            else:
                comp_detail = f"solid {bg_meta.get('color', [])}"
            _vstatus(verbose, "composite", comp_detail)
            _vlog(verbose, f"[dim]composite[/] {comp_detail}")

            _vstatus(verbose, "pick", f"α>{alpha_threshold}")
            x, y, clicked_object_id = sample_click(
                out.alpha, out.object_id_map, alpha_threshold, rng
            )
            mask = object_mask(out.object_id_map, clicked_object_id)
            mask_pixels = int((mask == 255).sum().item())
            pick_detail = (
                f"oid={clicked_object_id} @ ({x},{y}) · mask={mask_pixels:,}px"
            )
            _vstatus(verbose, "pick", pick_detail)
            _vlog(verbose, f"[dim]pick[/] {pick_detail}")

            if mask[y, x].item() != 255:
                raise ValueError("Click pixel not inside object mask")

            record = SampleRecord(
                id=sample_id,
                image="",
                mask="",
                point=[x, y],
                object_id=clicked_object_id,
                num_objects=scene.num_objects,
                background=bg_meta,
                camera={
                    "width": width,
                    "height": height,
                    "fov_deg": fov_deg,
                    "viewmat": viewmat.detach().cpu().tolist(),
                    "K": k.detach().cpu().tolist(),
                },
                objects=objects_meta,
            )
            _vstatus(verbose, "export", sample_id)
            export_sample(output_dir, sample_id, rgb, mask, record)
            _vstatus(verbose, "done", sample_id)
            _vlog(
                verbose,
                f"[dim]export[/] {record.image} · {record.mask} · oid={clicked_object_id}",
            )
            return record
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            _vstatus(verbose, "retry", str(exc)[:48])
            _vlog(verbose, f"[yellow]retry[/] {exc}")
            continue

    raise RuntimeError(f"Failed to generate sample after retries: {last_error}")
