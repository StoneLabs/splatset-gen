"""Rich terminal UI for dataset generation."""

from __future__ import annotations

import os
import shutil
import sys
import time

import click
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()

RECENT_LINES = 15
BRAILLE_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class WorkerState:
    status: Literal["idle", "working", "failed"] = "idle"
    sample_id: str = "—"
    elapsed: float = 0.0
    started_at: float | None = None


@dataclass
class ProgressTracker:
    num_samples: int
    workers: int
    worker_states: dict[int, WorkerState] = field(default_factory=dict)
    completed: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=RECENT_LINES))
    _progress: Progress = field(init=False, repr=False)
    _task_id: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for i in range(self.workers):
            self.worker_states[i] = WorkerState()
        self._progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]Overall"),
            BarColumn(bar_width=40, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[active]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        )
        self._task_id = self._progress.add_task(
            "samples", total=self.num_samples, active="0 active"
        )

    @property
    def in_flight(self) -> int:
        return sum(1 for st in self.worker_states.values() if st.status == "working")

    def on_start(self, worker_id: int, sample_id: str) -> None:
        st = self.worker_states[worker_id]
        st.status = "working"
        st.sample_id = sample_id
        st.started_at = time.perf_counter()
        st.elapsed = 0.0

    def on_done(self, worker_id: int, sample_id: str, elapsed: float, error: str | None) -> None:
        st = self.worker_states[worker_id]
        if error:
            st.status = "failed"
            self.failed += 1
            self.recent.append(f"[red]✗[/] {sample_id}  {escape(error[:72])}")
        else:
            self.completed += 1
            self.recent.append(f"[green]✓[/] {sample_id}  [dim]{elapsed:.1f}s[/]")
        st.status = "idle"
        st.sample_id = "—"
        st.elapsed = 0.0
        st.started_at = None

    def on_log(self, message: str) -> None:
        self.recent.append(message)

    def tick(self) -> None:
        now = time.perf_counter()
        active = 0
        for st in self.worker_states.values():
            if st.status == "working" and st.started_at is not None:
                st.elapsed = now - st.started_at
                active += 1
        self._progress.update(
            self._task_id,
            completed=self.completed,
            active=f"{active} active",
        )


def _status_style(status: str) -> str:
    return {
        "idle": "dim",
        "working": "cyan bold",
        "failed": "red bold",
    }.get(status, "white")


def _worker_label(worker_id: int, status: str) -> Text:
    """Worker column with braille spinner (animates while working)."""
    if status == "working":
        tick = int(time.perf_counter() * 10)
        frame = BRAILLE_SPINNER[(tick + worker_id) % len(BRAILLE_SPINNER)]
        return Text.assemble((frame + " ", "cyan bold"), (f"#{worker_id}", "cyan bold"))
    if status == "failed":
        return Text.assemble(("⠿ ", "red"), (f"#{worker_id}", "red bold"))
    return Text.assemble(("⠀ ", "dim"), (f"#{worker_id}", "dim"))


def _format_recent_panel(tracker: ProgressTracker) -> Text:
    """Fixed-height log window (always RECENT_LINES rows)."""
    lines = list(tracker.recent)
    if len(lines) > RECENT_LINES:
        lines = lines[-RECENT_LINES:]
    pad = RECENT_LINES - len(lines)
    padded = ["[dim] [/]"] * pad + lines
    return Text.from_markup("\n".join(padded))


def build_live_render(tracker: ProgressTracker) -> Table:
    root = Table.grid(expand=True)
    root.add_column(ratio=1)

    tracker.tick()

    worker_table = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False)
    worker_table.add_column("Worker", width=10)
    worker_table.add_column("Status", width=10)
    worker_table.add_column("Sample", width=10)
    worker_table.add_column("Elapsed", justify="right", width=10)

    for wid in sorted(tracker.worker_states):
        st = tracker.worker_states[wid]
        elapsed = f"{st.elapsed:.1f}s" if st.status == "working" else "—"
        worker_table.add_row(
            _worker_label(wid, st.status),
            Text(st.status, style=_status_style(st.status)),
            st.sample_id,
            elapsed,
        )

    root.add_row(Panel(tracker._progress, border_style="cyan", padding=(0, 1)))
    root.add_row(worker_table)
    root.add_row(
        Panel(
            _format_recent_panel(tracker),
            title="Recent",
            border_style="dim",
            padding=(0, 1),
            height=RECENT_LINES + 2,
        )
    )

    return root


def _fmt_range(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]}–{value[1]}"
    return str(value)


def print_plan(
    *,
    ply_dir: Path,
    ply_files: list[Path],
    config_path: Path,
    cfg: dict[str, Any],
    output: Path,
    num_samples: int,
    workers: int,
    seed: int,
    verbose: bool,
) -> None:
    render = cfg.get("render", {})
    scene = cfg.get("scene", {})
    bg = cfg.get("background", {})
    camera = cfg.get("camera", {})
    generation = cfg.get("generation", {})

    cpu = os.cpu_count() or 1
    threads_per_worker = max(1, cpu // max(workers, 1))
    cap = generation.get("max_gaussians_per_object")

    if output_dir_in_use(output):
        summary = _describe_output_dir(output)
        output_line = (
            f"  Output           [cyan]{output}[/]  "
            f"[yellow](exists — {summary}; delete y/N prompt next)[/]"
        )
    else:
        output_line = f"  Output           [cyan]{output}[/]  [dim](new directory)[/]"

    lines = [
        "[bold underline]Run[/]",
        f"  Samples          {num_samples}",
        f"  Workers          {workers}  [dim]({cpu} CPUs, ~{threads_per_worker} torch threads/worker)[/]",
        f"  Seed             {seed}",
        f"  Verbose raster   {'[yellow]on[/]' if verbose else '[dim]off[/]'}",
        output_line,
        "",
        "[bold underline]Input[/]",
        f"  PLY dir          [cyan]{ply_dir}[/]  ({len(ply_files)} files)",
        f"  Config           [cyan]{config_path}[/]",
    ]

    names = ", ".join(p.name for p in ply_files[:8])
    if len(ply_files) > 8:
        names += f", … +{len(ply_files) - 8} more"
    lines.append(f"  PLY files        {names}")

    lines.extend(
        [
            "",
            "[bold underline]Render[/]",
            f"  Resolution       {render.get('width', '?')}×{render.get('height', '?')}",
            f"  SH degree        {render.get('sh_degree', 0)}",
            f"  Alpha threshold  {render.get('alpha_threshold', 0.5)}",
            "",
            "[bold underline]Scene[/]",
            f"  Objects / sample {_fmt_range([scene.get('num_objects_min'), scene.get('num_objects_max')])}",
            f"  Position range   {_fmt_range(scene.get('position_range', []))} per axis",
            f"  Rotation max     {scene.get('rotation_deg_max', '?')}°",
            f"  Scale jitter     {_fmt_range(scene.get('scale_jitter', []))}",
            "",
            "[bold underline]Camera[/]",
            f"  FOV range        {_fmt_range(camera.get('fov_deg_range', []))}°",
            f"  Distance range   {_fmt_range(camera.get('distance_range', []))}",
            f"  Max retries      {camera.get('max_retries', '?')}",
            "",
            "[bold underline]Background[/]",
            f"  Mode             {bg.get('mode', 'solid')}",
            f"  Solid color      {bg.get('solid_color', '')}",
            "",
            "[bold underline]Generation[/]",
            f"  Camera retries   {generation.get('max_camera_retries', '?')}",
        ]
    )

    if cap:
        lines.append(
            f"  Gaussian cap     [yellow]{cap:,}[/] / PLY "
            f"[dim](random subset, generation.max_gaussians_per_object)[/]"
        )
    else:
        lines.append("  Gaussian cap     [dim]none (full PLY)[/]")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold cyan]splat-dataset[/]  generation plan",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def output_dir_in_use(path: Path) -> bool:
    """Return True if ``path`` exists and contains any prior run artifacts."""
    if not path.exists():
        return False
    return any(path.iterdir())


def _describe_output_dir(path: Path) -> str:
    jsonl = path / "annotations.jsonl"
    if jsonl.is_file():
        try:
            count = sum(1 for _ in jsonl.open())
            return f"{count} annotation(s) in annotations.jsonl"
        except OSError:
            pass
    entries = sum(1 for _ in path.iterdir())
    return f"{entries} item(s)"


def prepare_output_dir(output: Path, *, auto_confirm: bool = False) -> None:
    """Refuse to append to an existing run; delete after [y/N] confirmation or exit."""
    if not output_dir_in_use(output):
        return

    summary = _describe_output_dir(output)
    console.print(
        f"[yellow]Output directory already in use:[/] [cyan]{output}[/] [dim]({summary})[/]\n"
        "Existing runs cannot be appended to."
    )

    if auto_confirm:
        console.print("[dim]Deleting existing output without prompt (--yes)[/]")
    elif not click.confirm("Delete this directory and continue?", default=False, show_default=True):
        console.print("[yellow]Cancelled.[/]")
        raise SystemExit(0)

    shutil.rmtree(output)
    console.print(f"[dim]Deleted[/] [cyan]{output}[/]\n")


def confirm_or_exit(skip: bool) -> None:
    if skip:
        console.print("[dim]Skipping confirmation (--yes)[/]\n")
        return
    if not sys.stdin.isatty():
        console.print(
            "[yellow]Non-interactive terminal — use --yes to run without confirmation.[/]"
        )
        raise SystemExit(1)
    try:
        console.print("[dim]Press[/] [bold]Enter[/][dim] to start · Ctrl+C to cancel[/]")
        input()
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/]")
        raise SystemExit(0) from None
    console.print()


def print_summary(output: Path, completed: int, failed: int, elapsed: float) -> None:
    console.print()
    if failed:
        console.print(
            Panel(
                f"[green]✓ {completed}[/] samples written\n[red]✗ {failed}[/] failed\n"
                f"[dim]{elapsed:.1f}s total[/] → [cyan]{output}[/]",
                title="[bold]Done[/]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"[green]✓ {completed}[/] samples in [bold]{elapsed:.1f}s[/]\n[cyan]{output}[/]",
                title="[bold green]Done[/]",
                border_style="green",
            )
        )
