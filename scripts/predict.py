#!/usr/bin/env python3
"""Run click-to-segment inference and save a PNG mask.

Examples:
  python scripts/predict.py image.png 120 340 -o out/mask.png -c train/checkpoints/best.pth
  python scripts/predict.py image.png 64 64 -o compare.png -c ckpt.pth --visualization compare --gt gt.png
  python scripts/predict.py -h
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from PIL import Image
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
sys.path.insert(0, str(TRAIN_DIR))

from config import cfg  # noqa: E402
from inference import ModelRunner  # noqa: E402

console = Console()

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    no_args_is_help=True,
)

OutputFormat = Literal["alpha", "binary"]
Visualization = Literal["raw", "compare"]
Background = Literal["transparent", "black"]


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _resolve_checkpoint(path: Path | None) -> Path:
    if path is not None:
        return _resolve_path(path)
    if cfg.INFERENCE_CHECKPOINT:
        return Path(cfg.INFERENCE_CHECKPOINT)
    raise typer.BadParameter(
        "Pass --checkpoint or set inference.checkpoint in train/training_config.yaml",
    )


def _print_plan(
    *,
    image: Path,
    x: int,
    y: int,
    output: Path,
    checkpoint: Path,
    gt: Path | None,
    output_format: OutputFormat,
    visualization: Visualization,
    background: Background,
    threshold: float | None,
    device: str | None,
) -> None:
    table = Table(box=box.ROUNDED, show_header=False, pad_edge=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Image", str(image))
    table.add_row("Click", f"[bold]({x}, {y})[/]")
    table.add_row("Output", str(output))
    table.add_row("Checkpoint", str(checkpoint))
    table.add_row("Format", output_format)
    table.add_row("Visualization", visualization)
    table.add_row("Background", background)
    table.add_row("Threshold", str(threshold if threshold is not None else cfg.MASK_THRESHOLD))
    table.add_row("Device", device or cfg.DEVICE)
    if gt is not None:
        table.add_row("Ground truth", str(gt))

    console.print(
        Panel(
            table,
            title="[bold]Click-to-segment prediction[/]",
            border_style="blue",
        ),
    )


@app.command()
def main(
    image: Annotated[Path, typer.Argument(help="Input RGB image")],
    x: Annotated[int, typer.Argument(min=0, help="Click x coordinate (pixels)")],
    y: Annotated[int, typer.Argument(min=0, help="Click y coordinate (pixels)")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output PNG path"),
    ],
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            "-c",
            help=(
                "Model checkpoint (.pth). "
                "Default: [cyan]inference.checkpoint[/] in [cyan]train/training_config.yaml[/]"
            ),
        ),
    ] = None,
    gt: Annotated[
        Path | None,
        typer.Option(
            "--gt",
            help="Ground-truth mask PNG ([bold]required[/] when [cyan]--visualization compare[/])",
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help="[cyan]alpha[/] = soft sigmoid mask · [cyan]binary[/] = thresholded detect mask",
        ),
    ] = "alpha",
    visualization: Annotated[
        Visualization,
        typer.Option(
            "--visualization",
            "-V",
            help="[cyan]raw[/] = mask output · [cyan]compare[/] = TP green / FP red / FN white error map",
        ),
    ] = "raw",
    background: Annotated[
        Background,
        typer.Option(
            "--background",
            "-b",
            help="[cyan]transparent[/] = clear RGBA background · [cyan]black[/] = black background",
        ),
    ] = "transparent",
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            "-t",
            min=0.0,
            max=1.0,
            help="Mask binarization threshold (default: training config)",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            "-d",
            help="Torch device ([cyan]auto[/], [cyan]cpu[/], [cyan]cuda[/], [cyan]mps[/])",
        ),
    ] = None,
) -> None:
    """Predict a segmentation mask from an RGB image and click point."""
    image = _resolve_path(image)
    output = _resolve_path(output)
    checkpoint = _resolve_checkpoint(checkpoint)
    gt_path = _resolve_path(gt) if gt is not None else None

    if not image.is_file():
        raise typer.BadParameter(f"Image not found: {image}")
    if not checkpoint.is_file():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint}")
    if visualization == "compare" and gt_path is None:
        raise typer.BadParameter("--gt is required when --visualization compare")
    if gt_path is not None and not gt_path.is_file():
        raise typer.BadParameter(f"Ground-truth mask not found: {gt_path}")

    _print_plan(
        image=image,
        x=x,
        y=y,
        output=output,
        checkpoint=checkpoint,
        gt=gt_path,
        output_format=output_format,
        visualization=visualization,
        background=background,
        threshold=threshold,
        device=device,
    )

    with console.status("[bold cyan]Running inference…[/]", spinner="dots"):
        runner = ModelRunner(
            checkpoint,
            device=device,
            mask_threshold=threshold,
        )
        encoded, mode = runner.predict_png(
            image,
            [x, y],
            gt_path=gt_path,
            output_format=output_format,
            visualization=visualization,
            background=background,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode=mode).save(output)

    summary = Table(box=None, show_header=False, pad_edge=False)
    summary.add_column(style="cyan", no_wrap=True)
    summary.add_column()
    summary.add_row("Path", f"[bold]{output}[/]")
    summary.add_row("PIL mode", mode)
    summary.add_row("Format", output_format)
    summary.add_row("Visualization", visualization)
    summary.add_row("Background", background)
    summary.add_row("Device", runner.meta["device"])
    summary.add_row("Checkpoint epoch", str(runner.meta["epoch"]))
    summary.add_row("Threshold", f"{runner.mask_threshold:.3f}")
    summary.add_row("Size", f"{output.stat().st_size:,} bytes")

    console.print(
        Panel(
            summary,
            title="[bold green]Prediction saved[/]",
            border_style="green",
        ),
    )


if __name__ == "__main__":
    app()
