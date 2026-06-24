"""Random multi-object scene assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import event_log
from cached_scene import CachedScene, euler_placement
from ply_loader import format_extent, load_ply

import ply_manager


def build_random_scene(
    ply_paths: list[Path],
    config: dict[str, Any],
    rng: np.random.Generator,
    verbose: bool = False,
) -> tuple[CachedScene, list[dict[str, Any]]]:
    """Load, place, and reference random PLY objects (Gaussian data stays in cache)."""
    if not ply_paths:
        raise ValueError("ply_paths is empty")

    scene_cfg = config.get("scene", {})
    n_min = int(scene_cfg.get("num_objects_min", 2))
    n_max = int(scene_cfg.get("num_objects_max", 5))
    pos_range = scene_cfg.get("position_range", [-2.0, 2.0])
    rot_max = float(scene_cfg.get("rotation_deg_max", 180))
    scale_jitter = scene_cfg.get("scale_jitter", [0.8, 1.2])

    max_g = config.get("generation", {}).get("max_gaussians_per_object")

    num_objects = int(rng.integers(n_min, n_max + 1))
    chosen = [Path(rng.choice(ply_paths)) for _ in range(num_objects)]

    if verbose and event_log.is_active():
        from collections import Counter

        counts = Counter(p.name for p in chosen)
        summary = ", ".join(
            f"{name}×{count}" if count > 1 else name for name, count in sorted(counts.items())
        )
        cap_note = f" · cap {max_g:,}/ply" if max_g else ""
        event_log.log(
            f"[dim]scene[/] {num_objects} slots from {len(ply_paths)} PLYs{cap_note}: {summary}"
        )

    objects: list = []
    metadata: list[dict[str, Any]] = []

    for object_id, ply_path in enumerate(chosen):
        placement = rng.uniform(pos_range[0], pos_range[1], size=3).astype(np.float32)
        euler = np.deg2rad(rng.uniform(-rot_max, rot_max, size=3))
        scale = float(rng.uniform(scale_jitter[0], scale_jitter[1]))

        if ply_manager.is_active():
            event_log.worker_status("waiting for load", ply_path.name)
            event_log.report_memory()
            gaussians, load_stats, n_total, local_lo, local_hi = ply_manager.acquire(ply_path)
            event_log.report_memory()
            n_source = gaussians.num_gaussians
            pick: np.ndarray | None = None
            if max_g is not None and n_source > max_g:
                pick = rng.choice(n_source, size=max_g, replace=False).astype(np.int64)
            n_loaded = int(pick.shape[0]) if pick is not None else n_source

            if verbose and event_log.is_active():
                if pick is not None and n_total > n_loaded:
                    count_label = f"{n_loaded:,}/{n_total:,} gaussians (subset)"
                else:
                    count_label = f"{n_loaded:,} gaussians"
                extent_label = format_extent(load_stats)
                event_log.log(
                    f"[dim]ply[/] oid={object_id} {ply_path.name} · {count_label} · "
                    f"extent {extent_label} · [red]cache ref[/]"
                )

            objects.append(
                euler_placement(
                    gaussians,
                    euler,
                    placement,
                    scale,
                    object_id,
                    pick=pick,
                    ply_path=ply_path,
                    from_cache=True,
                    local_lo=local_lo,
                    local_hi=local_hi,
                )
            )
        else:
            if verbose and event_log.is_active():
                event_log.worker_status("waiting for load", ply_path.name)
            event_log.report_memory()
            gaussians, load_stats, n_total = load_ply(ply_path, max_gaussians=max_g, rng=rng)
            n_loaded = gaussians.num_gaussians
            event_log.report_memory()
            lo, hi = gaussians.bounds()

            if verbose and event_log.is_active():
                if max_g is not None and n_total > n_loaded:
                    count_label = f"{n_loaded:,}/{n_total:,} gaussians (subset)"
                else:
                    count_label = f"{n_loaded:,} gaussians"
                extent_label = format_extent(load_stats)
                event_log.log(
                    f"[dim]ply[/] oid={object_id} {ply_path.name} · {count_label} · "
                    f"extent {extent_label}"
                )

            objects.append(
                euler_placement(
                    gaussians,
                    euler,
                    placement,
                    scale,
                    object_id,
                    pick=None,
                    ply_path=None,
                    from_cache=False,
                    local_lo=lo,
                    local_hi=hi,
                )
            )

        metadata.append(
            {
                "object_id": object_id,
                "ply": ply_path.name,
                "transform": {
                    "translation": placement.tolist(),
                    "rotation_euler_deg": np.rad2deg(euler).tolist(),
                    "scale": scale,
                },
            }
        )

        if verbose and event_log.is_active():
            rot_deg = np.rad2deg(euler)
            event_log.log(
                f"[dim]place[/] oid={object_id} "
                f"pos=[{placement[0]:+.2f},{placement[1]:+.2f},{placement[2]:+.2f}] "
                f"rot=[{rot_deg[0]:.0f},{rot_deg[1]:.0f},{rot_deg[2]:.0f}]° "
                f"scale={scale:.2f}"
            )

    return CachedScene(objects=objects), metadata
