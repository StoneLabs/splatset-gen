"""Generate one training sample: scene → render → click → mask → export."""

from __future__ import annotations

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
from scene import build_random_scene


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

    scene, objects_meta = build_random_scene(ply_paths, config, rng)
    lo, hi = scene.bounds()
    background = background_from_config(config, base_dir=project_root)

    last_error: Exception | None = None
    for _ in range(max_camera_retries):
        try:
            viewmat, k, width, height, fov_deg = sample_random_camera(
                (lo, hi), config, rng, scene_means=scene.means
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
            if out.alpha.max() < alpha_threshold:
                raise ValueError("Render has insufficient foreground coverage")

            rgb, bg_meta = composite(
                out.fg_rgb, out.alpha, background, width, height, rng=rng
            )
            x, y, clicked_object_id = sample_click(
                out.alpha, out.object_id_map, alpha_threshold, rng
            )
            mask = object_mask(out.object_id_map, clicked_object_id)

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
            export_sample(output_dir, sample_id, rgb, mask, record)
            return record
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Failed to generate sample after retries: {last_error}")
