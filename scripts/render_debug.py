#!/usr/bin/env python3
"""Debug render: one PLY → PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from background import background_from_config, composite  # noqa: E402
from camera import camera_from_orbit  # noqa: E402
from ply_loader import load_ply, format_extent  # noqa: E402
from render import render  # noqa: E402


def _save_rgb(path: Path, rgb: torch.Tensor) -> None:
    arr = (rgb.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one 3DGS PLY to PNG")
    parser.add_argument(
        "--ply",
        type=Path,
        default=Path("assets/ply/Grape.ply"),
        help="Input 3DGS PLY file",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=None,
        help="Cap Gaussians loaded from PLY (random subset; default: all)",
    )
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/debug/render.png"),
        help="Output composited PNG path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML config for background + SH settings",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print rasterizer progress",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    print(f"Loading {args.ply} ...")
    scene, load_stats, _total = load_ply(args.ply, max_gaussians=args.max_gaussians)
    print(f"  {scene.num_gaussians} Gaussians · extent {format_extent(load_stats)}")

    lo, hi = scene.bounds()
    viewmat, k, w, h, _ = camera_from_orbit((lo, hi), width=args.width, height=args.height)

    print(f"Rendering {w}x{h} ...")
    out = render(
        scene,
        viewmat,
        k,
        w,
        h,
        sh_degree=int(cfg.get("render", {}).get("sh_degree", 0)),
        verbose=args.verbose,
    )

    fg_path = args.output.with_name(args.output.stem + "_fg" + args.output.suffix)
    _save_rgb(fg_path, out.fg_rgb)
    print(f"  fg_rgb → {fg_path}  (alpha max={out.alpha.max():.3f})")

    bg = background_from_config(cfg, base_dir=ROOT)
    rng = np.random.default_rng(0)
    rgb, bg_meta = composite(out.fg_rgb, out.alpha, bg, w, h, rng=rng)
    _save_rgb(args.output, rgb)
    print(f"  rgb    → {args.output}  (background={bg_meta})")

    if out.alpha.max() < 0.01:
        print("Warning: render is nearly blank")


if __name__ == "__main__":
    main()
