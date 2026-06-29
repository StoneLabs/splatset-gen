#!/usr/bin/env python3
"""Generate click-to-segment training dataset from 3DGS PLY objects.

Each sample: composited RGB, occlusion-aware object mask, click (x, y) in JSONL.

Examples:
  PYTHONPATH=src python scripts/generate_dataset.py
  PYTHONPATH=src python scripts/generate_dataset.py -h
  PYTHONPATH=src python scripts/generate_dataset.py -n 10 -j 1 -c configs/dev_fast.yaml -y
  PYTHONPATH=src python scripts/generate_dataset.py -o outputs/run_001 --continue -n 50 -y
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from console import (  # noqa: E402
    confirm_or_exit,
    output_dir_in_use,
    prepare_output_dir,
    print_plan,
    resolve_continue_config,
)
from export import get_last_sample_index  # noqa: E402
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
        help=(
            f"Output run directory [default: {DEFAULT_OUTPUT}]. "
            "Must be empty unless --continue is set; if it already contains a run "
            "you will be asked y/N to delete it (or pass --yes to delete without asking)."
        ),
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
        help=(
            "Non-interactive mode: delete existing output without y/N prompt, "
            "then start generation without the Enter prompt"
        ),
    ),
    continue_: bool = typer.Option(
        False,
        "--continue",
        help=(
            "Append to an existing output run: detect the last sample and generate "
            "the next --num-samples IDs without deleting prior files"
        ),
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
    config_path = config

    start_index = 1
    write_config_snapshot = True
    continue_from: int | None = None
    if continue_:
        if not output.is_dir():
            raise typer.BadParameter(f"Cannot continue: output directory not found: {output}")
        if not output_dir_in_use(output):
            raise typer.BadParameter(f"Cannot continue: output directory is empty: {output}")
        last_index = get_last_sample_index(output)
        if last_index <= 0:
            raise typer.BadParameter(f"Cannot continue: no existing samples in {output}")
        start_index = last_index + 1
        continue_from = start_index
        write_config_snapshot = False
        cfg, config_path = resolve_continue_config(
            output_dir=output,
            cli_config_path=config,
            cli_cfg=cfg,
            auto_confirm=yes,
        )
    else:
        prepare_output_dir(output, auto_confirm=yes)

    print_plan(
        ply_dir=ply_dir,
        ply_files=ply_files,
        config_path=config_path,
        cfg=cfg,
        output=output,
        num_samples=num_samples,
        workers=workers,
        seed=seed,
        verbose=verbose,
        project_root=ROOT,
        continue_from=continue_from,
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
        project_root=ROOT,
        start_index=start_index,
        write_config_snapshot=write_config_snapshot,
    )


if __name__ == "__main__":
    app()
