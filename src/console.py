"""Rich terminal UI for dataset generation."""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

import click

from background import background_from_config, list_background_images
from ply_cache import CACHE_WORKER_ID

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

RECENT_LINES = 20
BRAILLE_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_RICH_TAG = re.compile(r"\[[^\]]*\]")


def _worker_prefix_rich(worker_id: int) -> str:
    if worker_id == CACHE_WORKER_ID:
        return "[red]mem[/] "
    return f"[purple]#{worker_id}[/] "


def _worker_prefix_plain(worker_id: int) -> str:
    if worker_id == CACHE_WORKER_ID:
        return "mem "
    return f"#{worker_id} "


def _strip_rich_markup(text: str) -> str:
    return _RICH_TAG.sub("", text)


@dataclass
class WorkerState:
    status: Literal["idle", "working", "failed"] = "idle"
    sample_id: str = "—"
    elapsed: float = 0.0
    started_at: float | None = None
    render_pct: float | None = None
    phase: str = "—"
    detail: str = "—"


@dataclass
class CacheEntryRow:
    name: str
    refs: int
    memory_bytes: int
    status: str
    timeout_s: float | None = None


@dataclass
class ProgressTracker:
    num_samples: int
    workers: int
    verbose: bool = False
    log_path: Path | None = None
    worker_states: dict[int, WorkerState] = field(default_factory=dict)
    cache_entries: list[CacheEntryRow] = field(default_factory=list)
    completed: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=RECENT_LINES))
    _progress: Progress = field(init=False, repr=False)
    _task_id: int = field(init=False, repr=False)
    _log_file: TextIO | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        for i in range(self.workers):
            self.worker_states[i] = WorkerState()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("a", encoding="utf-8")
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

    def close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _record(self, worker_id: int, rich_body: str) -> None:
        self.recent.append(_worker_prefix_rich(worker_id) + rich_body)
        if self._log_file is not None:
            plain = _worker_prefix_plain(worker_id) + _strip_rich_markup(rich_body)
            self._log_file.write(plain + "\n")
            self._log_file.flush()

    def on_start(self, worker_id: int, sample_id: str) -> None:
        st = self.worker_states[worker_id]
        st.status = "working"
        st.sample_id = sample_id
        st.started_at = time.perf_counter()
        st.elapsed = 0.0
        st.render_pct = None
        st.phase = "starting"
        st.detail = "—"

    def on_done(self, worker_id: int, sample_id: str, elapsed: float, error: str | None) -> None:
        st = self.worker_states[worker_id]
        if error:
            st.status = "failed"
            self.failed += 1
            self._record(worker_id, f"[red]✗[/] {sample_id}  {escape(error[:72])}")
        else:
            self.completed += 1
            self._record(worker_id, f"[green]✓[/] {sample_id}  [dim]{elapsed:.1f}s[/]")
        st.status = "idle"
        st.sample_id = "—"
        st.elapsed = 0.0
        st.started_at = None
        st.render_pct = None
        st.phase = "—"
        st.detail = "—"

    def on_render(self, worker_id: int, pct: float) -> None:
        st = self.worker_states[worker_id]
        if st.status == "working":
            st.render_pct = max(0.0, min(100.0, pct))

    def on_status(self, worker_id: int, phase: str, detail: str) -> None:
        st = self.worker_states[worker_id]
        if st.status == "working":
            st.phase = phase
            st.detail = detail or "—"

    def on_log(self, worker_id: int, message: str) -> None:
        self._record(worker_id, message)

    def _record_mem(self, rich_body: str) -> None:
        if self.verbose:
            self._record(CACHE_WORKER_ID, rich_body)

    def on_cache_claim(self, worker_id: int, ply_name: str, ref_count: int) -> None:
        self._record_mem(
            f"[red]worker {worker_id}[/] claimed usage of [cyan]{ply_name}[/] "
            f"[dim]· {ref_count} handle{'s' if ref_count != 1 else ''}[/]"
        )

    def on_cache_release(self, worker_id: int, ply_name: str, ref_count: int) -> None:
        self._record_mem(
            f"[red]worker {worker_id}[/] released [cyan]{ply_name}[/] "
            f"[dim]· {ref_count} handle{'s' if ref_count != 1 else ''} open[/]"
        )

    def on_cache_evict(self, ply_name: str) -> None:
        self._record_mem(f"[dim]evicted[/] [cyan]{ply_name}[/] from cache")

    def refresh_cache(self, cache_proxy: Any) -> None:
        try:
            rows = cache_proxy.snapshot()
        except (BrokenPipeError, ConnectionError, EOFError, OSError):
            return
        self.cache_entries = [
            CacheEntryRow(
                name=row["name"],
                refs=int(row["refs"]),
                memory_bytes=int(row["memory_bytes"]),
                status=str(row["status"]),
                timeout_s=row["timeout_s"],
            )
            for row in rows
        ]

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


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def _cache_status_style(status: str) -> str:
    return {
        "loading": "yellow",
        "loaded": "green",
        "unloading": "red dim",
    }.get(status, "red")


def _format_recent_panel(tracker: ProgressTracker) -> Text:
    """Fixed-height log window (always RECENT_LINES rows)."""
    lines = list(tracker.recent)
    if len(lines) > RECENT_LINES:
        lines = lines[-RECENT_LINES:]
    pad = RECENT_LINES - len(lines)
    padded = ["[dim] [/]"] * pad + lines
    return Text.from_markup("\n".join(padded))


def _build_memory_panel(tracker: ProgressTracker) -> Table:
    mem_table = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False)
    mem_table.add_column("Splat", min_width=10, no_wrap=False)
    mem_table.add_column("Status", width=10)
    mem_table.add_column("Refs", justify="right", width=5)
    mem_table.add_column("Memory", justify="right", width=9)
    mem_table.add_column("Timeout", justify="right", width=8)

    if tracker.cache_entries:
        for row in tracker.cache_entries:
            if row.refs > 0 or row.timeout_s is None:
                timeout = "—"
            else:
                timeout = f"{row.timeout_s:.1f}s"
            mem_table.add_row(
                row.name,
                Text(row.status, style=_cache_status_style(row.status)),
                str(row.refs),
                _format_bytes(row.memory_bytes),
                timeout,
            )
    else:
        mem_table.add_row("[dim]—[/]", "—", "—", "—", "—")

    return mem_table


def build_live_render(tracker: ProgressTracker) -> Table:
    root = Table.grid(expand=True)
    root.add_column(ratio=1)

    tracker.tick()

    worker_table = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False)
    worker_table.add_column("Worker", width=10)
    worker_table.add_column("Status", width=10)
    worker_table.add_column("Sample", width=10)
    worker_table.add_column("Render", justify="right", width=7)
    if tracker.verbose:
        worker_table.add_column("Phase", width=10)
        worker_table.add_column("Detail", min_width=24, no_wrap=False)
    worker_table.add_column("Elapsed", justify="right", width=10)

    for wid in sorted(tracker.worker_states):
        st = tracker.worker_states[wid]
        elapsed = f"{st.elapsed:.1f}s" if st.status == "working" else "—"
        if st.status == "working" and st.render_pct is not None:
            render = f"{int(st.render_pct)}%"
        else:
            render = "—"
        if st.status == "working" and st.detail.startswith("waiting for load"):
            status_label = st.detail
            status_style = "yellow"
        else:
            status_label = st.status
            status_style = _status_style(st.status)
        row: list[Any] = [
            _worker_label(wid, st.status),
            Text(status_label, style=status_style),
            st.sample_id,
            render,
        ]
        if tracker.verbose:
            row.extend([st.phase, st.detail])
        row.append(elapsed)
        worker_table.add_row(*row)

    bottom = Table.grid(expand=True)
    bottom.add_column(ratio=3)
    bottom.add_column(ratio=2)
    bottom.add_row(
        Panel(
            _format_recent_panel(tracker),
            title="Recent",
            border_style="dim",
            padding=(0, 1),
            height=RECENT_LINES + 2,
        ),
        Panel(
            _build_memory_panel(tracker),
            title="[red]Memory[/]",
            border_style="red",
            padding=(0, 1),
            height=RECENT_LINES + 2,
        ),
    )

    root.add_row(Panel(tracker._progress, border_style="cyan", padding=(0, 1)))
    root.add_row(worker_table)
    root.add_row(bottom)

    return root


def _fmt_range(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]}–{value[1]}"
    return str(value)


def _background_plan_lines(cfg: dict[str, Any], project_root: Path | None) -> list[str]:
    """Background section lines for the pre-run plan panel."""
    bg_spec = background_from_config(cfg, base_dir=project_root)
    lines = [f"  Mode             {bg_spec.mode}"]

    if bg_spec.mode == "image":
        image_dir = bg_spec.image_dir
        if image_dir is None:
            lines.append("  Image dir        [red](not set)[/]")
        else:
            try:
                images = list_background_images(image_dir)
                names = ", ".join(p.name for p in images[:6])
                if len(images) > 6:
                    names += f", … +{len(images) - 6} more"
                lines.append(f"  Image dir        [cyan]{image_dir}[/]  ({len(images)} files)")
                lines.append(f"  Images           {names}")
            except OSError as exc:
                lines.append(f"  Image dir        [cyan]{image_dir}[/]  [red]({exc})[/]")
        lines.append(f"  Resize mode      {bg_spec.resize_mode}")
        lines.append(f"  Letterbox color  {list(bg_spec.solid_color)}")
    else:
        lines.append(f"  Solid color      {list(bg_spec.solid_color)}")

    return lines


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
    project_root: Path | None = None,
) -> None:
    render = cfg.get("render", {})
    scene = cfg.get("scene", {})
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
            f"  Mask mode        {render.get('mask_mode', 'binary')}",
            f"  Mask weight τ    {render.get('mask_weight_threshold', 0.05)}",
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
        ]
    )
    lines.extend(_background_plan_lines(cfg, project_root))
    lines.extend(
        [
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
