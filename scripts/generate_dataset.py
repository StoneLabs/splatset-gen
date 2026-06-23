#!/usr/bin/env python3
"""Generate click-to-segment training dataset from 3DGS PLY objects.

Each sample: composited RGB, occlusion-aware object mask, click (x, y) in JSONL.

Examples:
  PYTHONPATH=src python scripts/generate_dataset.py
  PYTHONPATH=src python scripts/generate_dataset.py -h
  PYTHONPATH=src python scripts/generate_dataset.py -n 10 -j 1 -c configs/dev_fast.yaml -y
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from console import confirm_or_exit, print_plan  # noqa: E402
from parallel import generate_dataset_parallel, load_config  # noqa: E402

DEFAULT_PLY_DIR = ROOT / "assets" / "ply"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT = ROOT / "outputs" / "run_001"

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=False,
)


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


@app.command()
def main(
    ply_dir: Path = typer.Option(
        DEFAULT_PLY_DIR,
        "--ply-dir",
        help=f"Directory containing input .ply splat files [default: {DEFAULT_PLY_DIR}]",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help=f"Output run directory [default: {DEFAULT_OUTPUT}]",
    ),
    num_samples: int = typer.Option(
        100,
        "--num-samples",
        "-n",
        min=1,
        help="Number of training samples to generate",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        "-j",
        help="Parallel worker processes (default: CPU count - 1)",
    ),
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help=f"YAML config [default: {DEFAULT_CONFIG}]",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Master RNG seed; each worker/sample derives a deterministic sub-seed",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print per-Gaussian rasterizer progress (very noisy)",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip pre-flight confirmation prompt",
    ),
) -> None:
    """Generate a click-to-segment dataset from 3DGS PLY objects."""
    ply_dir = _resolve_path(ply_dir)
    config = _resolve_path(config)
    output = _resolve_path(output)

    if not config.is_file():
        raise typer.BadParameter(f"Config not found: {config}")
    if not ply_dir.is_dir():
        raise typer.BadParameter(f"PLY directory not found: {ply_dir}")

    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        raise typer.BadParameter(f"No .ply files in {ply_dir}")

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    cfg = load_config(config)

    print_plan(
        ply_dir=ply_dir,
        ply_files=ply_files,
        config_path=config,
        cfg=cfg,
        output=output,
        num_samples=num_samples,
        workers=workers,
        seed=seed,
        verbose=verbose,
    )
    confirm_or_exit(skip=yes)

    generate_dataset_parallel(
        ply_dir=ply_dir,
        output_dir=output,
        config=cfg,
        num_samples=num_samples,
        workers=workers,
        seed=seed,
        verbose=verbose,
        show_progress=True,
    )


if __name__ == "__main__":
    app()
