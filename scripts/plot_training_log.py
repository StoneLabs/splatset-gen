#!/usr/bin/env python3
"""Plot training metrics from a CSV log (e.g. train/logs/training_log.csv).

Examples:
  uv run scripts/plot_training_log.py train/logs/training_log.csv
  uv run scripts/plot_training_log.py train/logs/training_log.csv -o out/plots
  uv run scripts/plot_training_log.py train/logs/training_log.csv --no-show
  uv run scripts/plot_training_log.py train/logs/training_log.csv -f

Opens a local Flask HTML gallery when --show is used. Also writes index.html beside the PNGs.
With -f/--follow, watches the log and regenerates plots; the browser auto-refreshes on new epochs.
"""

from __future__ import annotations

import csv
import html
import math
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from flask import Flask, jsonify, send_from_directory

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "outputs"

CORE_COLUMNS = {"epoch", "lr", "train_loss", "val_loss"}
BINARY_METRIC_KEYS = [
    "bin_iou",
    "bin_dice",
    "bin_f1",
    "bin_precision",
    "bin_recall",
]
SOFT_METRIC_KEYS = ["soft_iou", "soft_f1", "soft_dice", "alpha_mae"]
METRIC_KEYS = BINARY_METRIC_KEYS + SOFT_METRIC_KEYS
RUN_LOSS_KEYS = ["soft_loss", "bin_loss"]
RUN_FIELD_KEYS = METRIC_KEYS + RUN_LOSS_KEYS

SECTION_LAYOUT: dict[str, dict[str, int]] = {
    "Overall": {"cols": 1},
    "Per-dataset loss": {"cols": 2},
    "Per-dataset metrics": {"cols": 2},
}

_REGEN_LOCK = threading.Lock()


@dataclass(frozen=True)
class PanelSpec:
    filename: str
    title: str
    x_label: str
    y_label: str
    x_values: list[float]
    series: dict[str, list[float]]


@dataclass(frozen=True)
class PanelSection:
    title: str
    panels: list[PanelSpec]


def _use_backend(name: str) -> None:
    import matplotlib

    matplotlib.use(name, force=True)


def _import_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _discover_dataset_runs() -> list[str]:
    if not DEFAULT_DATASET_ROOT.is_dir():
        return []
    return sorted(
        path.name
        for path in DEFAULT_DATASET_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("run_")
    )


def _runs_from_columns(columns: list[str]) -> list[str]:
    runs: set[str] = set()
    for column in columns:
        if column in CORE_COLUMNS or column.startswith("val_"):
            continue
        for suffix in sorted(RUN_FIELD_KEYS, key=len, reverse=True):
            token = f"_{suffix}"
            if column.endswith(token):
                runs.add(column[: -len(token)])
                break
    return sorted(runs)


def _extend_header_for_trailing_fields(
    header: list[str],
    row_width: int,
    *,
    known_runs: list[str],
) -> list[str]:
    if row_width <= len(header):
        return header

    extra = row_width - len(header)
    if extra % len(METRIC_KEYS) != 0:
        raise typer.BadParameter(
            f"Row has {extra} trailing field(s) that do not match metric block size "
            f"({len(METRIC_KEYS)}); cannot infer missing dataset columns."
        )

    header_runs = set(_runs_from_columns(header))
    missing_runs = [run for run in known_runs if run not in header_runs]
    new_run_count = extra // len(METRIC_KEYS)
    if len(missing_runs) < new_run_count:
        for index in range(new_run_count - len(missing_runs)):
            missing_runs.append(f"run_unknown_{index + 1}")

    extended = list(header)
    for run in missing_runs[:new_run_count]:
        extended.extend(f"{run}_{key}" for key in METRIC_KEYS)
    return extended


def _load_log(
    path: Path,
    *,
    quiet: bool = False,
) -> tuple[list[str], list[dict[str, str]]]:
    known_runs = _discover_dataset_runs()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise typer.BadParameter(f"Log file is empty: {path}") from exc

        if not header:
            raise typer.BadParameter(f"No columns found in {path}")

        rows: list[dict[str, str]] = []
        max_width = len(header)
        raw_rows: list[list[str]] = []
        for raw in reader:
            if not raw:
                continue
            raw_rows.append(raw)
            max_width = max(max_width, len(raw))

        columns = _extend_header_for_trailing_fields(
            header,
            max_width,
            known_runs=known_runs,
        )
        if len(columns) > len(header) and not quiet:
            added = [column for column in columns if column not in header]
            typer.echo(
                f"Recovered {len(added)} column(s) for dataset(s) added mid-run: "
                + ", ".join(_runs_from_columns(added)),
            )

        for raw in raw_rows:
            padded = raw + [""] * (len(columns) - len(raw))
            rows.append(dict(zip(columns, padded, strict=False)))

    if not rows:
        raise typer.BadParameter(f"No data rows in {path}")
    return columns, rows


def _parse_series(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "")
        if raw is None or raw == "":
            values.append(math.nan)
            continue
        try:
            values.append(float(raw))
        except ValueError as exc:
            raise typer.BadParameter(
                f"Column {column!r} has non-numeric value {raw!r} at row {len(values) + 1}"
            ) from exc
    return values


def _x_axis(columns: list[str], rows: list[dict[str, str]]) -> tuple[str, list[float]]:
    if "epoch" in columns:
        return "epoch", _parse_series(rows, "epoch")
    return "step", [float(i) for i in range(len(rows))]


def _val_metric_suffixes(columns: list[str]) -> list[str]:
    suffixes = [
        column.removeprefix("val_")
        for column in columns
        if column.startswith("val_") and column not in {"val_loss"}
    ]
    return sorted(set(suffixes), key=len, reverse=True)


def _split_run_column(column: str, metric_suffixes: list[str]) -> tuple[str, str] | None:
    for suffix in metric_suffixes:
        token = f"_{suffix}"
        if column.endswith(token):
            return column[: -len(token)], suffix
    return None


def _group_per_run(columns: list[str], metric_suffixes: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for column in columns:
        if column in CORE_COLUMNS or column.startswith("val_"):
            continue
        parsed = _split_run_column(column, metric_suffixes)
        if parsed is None:
            continue
        run, suffix = parsed
        if suffix in METRIC_KEYS:
            grouped.setdefault(run, []).append(column)
    for run in grouped:
        grouped[run].sort()
    return dict(sorted(grouped.items()))


def _series_or_proxy(
    rows: list[dict[str, str]],
    columns: list[str],
    run: str,
    loss_key: str,
    *,
    proxy_metric: str,
) -> tuple[list[float], bool]:
    loss_column = f"{run}_{loss_key}"
    if loss_column in columns:
        return _parse_series(rows, loss_column), False

    metric_column = f"{run}_{proxy_metric}"
    if metric_column not in columns:
        return [math.nan] * len(rows), False

    proxy = [1.0 - value for value in _parse_series(rows, metric_column)]
    return proxy, True


def _build_panel_sections(columns: list[str], rows: list[dict[str, str]]) -> list[PanelSection]:
    x_label, x_values = _x_axis(columns, rows)
    metric_suffixes = _val_metric_suffixes(columns)
    per_run = _group_per_run(columns, metric_suffixes)
    runs = sorted(per_run)

    overall: list[PanelSpec] = []
    loss_panels: list[PanelSpec] = []
    per_dataset: list[PanelSpec] = []

    loss_cols = [c for c in ("train_loss", "val_loss") if c in columns]
    if loss_cols:
        overall.append(
            PanelSpec(
                filename="losses.png",
                title="Losses (overall)",
                x_label=x_label,
                y_label="loss",
                x_values=x_values,
                series={col: _parse_series(rows, col) for col in loss_cols},
            )
        )

    if "lr" in columns:
        overall.append(
            PanelSpec(
                filename="learning_rate.png",
                title="Learning rate",
                x_label=x_label,
                y_label="lr",
                x_values=x_values,
                series={"lr": _parse_series(rows, "lr")},
            )
        )

    val_metric_cols = sorted(
        c for c in columns if c.startswith("val_") and c not in {"val_loss"}
    )
    val_bin_cols = [c for c in val_metric_cols if "_bin_" in c]
    val_soft_cols = [c for c in val_metric_cols if c not in val_bin_cols]
    if val_bin_cols:
        overall.append(
            PanelSpec(
                filename="val_bin_metrics.png",
                title="Validation binary metrics (aggregate)",
                x_label=x_label,
                y_label="value",
                x_values=x_values,
                series={col: _parse_series(rows, col) for col in val_bin_cols},
            )
        )
    if val_soft_cols:
        overall.append(
            PanelSpec(
                filename="val_soft_metrics.png",
                title="Validation soft metrics (aggregate)",
                x_label=x_label,
                y_label="value",
                x_values=x_values,
                series={col: _parse_series(rows, col) for col in val_soft_cols},
            )
        )

    soft_series: dict[str, list[float]] = {}
    bin_series: dict[str, list[float]] = {}
    soft_proxy = False
    bin_proxy = False
    for run in runs:
        soft_values, used_soft_proxy = _series_or_proxy(
            rows,
            columns,
            run,
            "soft_loss",
            proxy_metric="soft_f1",
        )
        bin_values, used_bin_proxy = _series_or_proxy(
            rows,
            columns,
            run,
            "bin_loss",
            proxy_metric="bin_f1",
        )
        soft_series[run] = soft_values
        bin_series[run] = bin_values
        soft_proxy = soft_proxy or used_soft_proxy
        bin_proxy = bin_proxy or used_bin_proxy

    if soft_series:
        title = "Soft loss per dataset"
        if soft_proxy:
            title += " (proxy: 1 - soft_f1)"
        loss_panels.append(
            PanelSpec(
                filename="soft_loss_per_dataset.png",
                title=title,
                x_label=x_label,
                y_label="soft loss",
                x_values=x_values,
                series=soft_series,
            )
        )

    if bin_series:
        title = "Bin loss per dataset"
        if bin_proxy:
            title += " (proxy: 1 - bin_f1)"
        loss_panels.append(
            PanelSpec(
                filename="bin_loss_per_dataset.png",
                title=title,
                x_label=x_label,
                y_label="bin loss",
                x_values=x_values,
                series=bin_series,
            )
        )

    for run, run_cols in per_run.items():
        safe_run = run.replace("/", "_")
        per_dataset.append(
            PanelSpec(
                filename=f"{safe_run}_metrics.png",
                title=f"Metrics — {run}",
                x_label=x_label,
                y_label="value",
                x_values=x_values,
                series={col: _parse_series(rows, col) for col in run_cols},
            )
        )

    grouped_cols = {col for cols in per_run.values() for col in cols}
    extra_cols = [
        c
        for c in columns
        if c not in CORE_COLUMNS
        and not c.startswith("val_")
        and c not in grouped_cols
    ]
    if extra_cols:
        per_dataset.append(
            PanelSpec(
                filename="other_metrics.png",
                title="Other metrics",
                x_label=x_label,
                y_label="value",
                x_values=x_values,
                series={col: _parse_series(rows, col) for col in sorted(extra_cols)},
            )
        )

    sections: list[PanelSection] = []
    if overall:
        sections.append(PanelSection(title="Overall", panels=overall))
    if loss_panels:
        sections.append(PanelSection(title="Per-dataset loss", panels=loss_panels))
    if per_dataset:
        sections.append(PanelSection(title="Per-dataset metrics", panels=per_dataset))
    return sections


def _all_panels(sections: list[PanelSection]) -> list[PanelSpec]:
    panels: list[PanelSpec] = []
    for section in sections:
        panels.extend(section.panels)
    return panels


_PLOT_COLORS = [
    "#4f8cff",
    "#ff6b6b",
    "#51cf66",
    "#fcc419",
    "#cc5de8",
    "#20c997",
    "#ff922b",
    "#748ffc",
]


def _draw_panel(ax, spec: PanelSpec) -> None:
    for index, (label, y_values) in enumerate(spec.series.items()):
        color = _PLOT_COLORS[index % len(_PLOT_COLORS)]
        ax.plot(
            spec.x_values,
            y_values,
            marker="o",
            markersize=4,
            linewidth=2,
            label=label,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            alpha=0.92,
        )
    ax.set_title(spec.title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(spec.x_label, fontsize=9)
    ax.set_ylabel(spec.y_label, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.7)
    ax.set_facecolor("#fafbfc")
    for spine in ax.spines.values():
        spine.set_alpha(0.35)
    if len(spec.series) > 1:
        ax.legend(loc="best", fontsize=7.5, framealpha=0.9, edgecolor="#e0e0e0")


def _render_panel_figure(spec: PanelSpec):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    _draw_panel(ax, spec)
    fig.tight_layout(pad=1.2)
    return fig


def _save_panels(panels: list[PanelSpec], out_dir: Path) -> list[Path]:
    _use_backend("Agg")
    plt = _import_pyplot()
    saved: list[Path] = []
    for spec in panels:
        fig = _render_panel_figure(spec)
        out_path = out_dir / spec.filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
    return saved


_GALLERY_CSS = """
:root {
  --bg: #0f1117;
  --surface: #181b24;
  --surface-raised: #1f2430;
  --border: #2a3142;
  --text: #e8ecf4;
  --muted: #8b95a8;
  --accent: #5b8def;
  --accent-soft: rgb(91 141 239 / 14%);
  --success: #3dd68c;
  --success-soft: rgb(61 214 140 / 12%);
  --shadow: 0 8px 32px rgb(0 0 0 / 28%);
  --radius: 14px;
  --font: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  --mono: "JetBrains Mono", "SF Mono", ui-monospace, monospace;
}
* { box-sizing: border-box; }
body {
  font-family: var(--font);
  margin: 0;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  min-height: 100vh;
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgb(91 141 239 / 10%), transparent),
    radial-gradient(ellipse 60% 40% at 100% 0%, rgb(61 214 140 / 5%), transparent);
  pointer-events: none;
  z-index: 0;
}
header {
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgb(15 17 23 / 82%);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 1.1rem 1.75rem;
}
.header-inner {
  max-width: 1280px;
  margin: 0 auto;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.25rem;
  flex-wrap: wrap;
}
.header-text h1 {
  margin: 0;
  font-size: 1.45rem;
  font-weight: 600;
  letter-spacing: -0.02em;
}
.header-text p {
  margin: 0.3rem 0 0;
  color: var(--muted);
  font-size: 0.82rem;
  font-family: var(--mono);
  word-break: break-all;
}
.header-meta {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 550;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
  white-space: nowrap;
}
.badge.epochs { color: var(--text); }
.badge.live {
  border-color: rgb(61 214 140 / 35%);
  background: var(--success-soft);
  color: var(--success);
}
.badge.live .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--success);
  animation: pulse 2s ease-in-out infinite;
}
.badge.refreshing {
  border-color: rgb(91 141 239 / 35%);
  background: var(--accent-soft);
  color: var(--accent);
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.45; transform: scale(0.85); }
}
main {
  position: relative;
  z-index: 1;
  max-width: 1280px;
  margin: 0 auto;
  padding: 1.5rem 1.75rem 3rem;
}
section { margin: 2.25rem 0; }
section h2 {
  margin: 0 0 1.1rem;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
}
.grid { display: grid; gap: 1.1rem; }
.cols-1 { grid-template-columns: 1fr; }
.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
figure {
  margin: 0;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.85rem;
  box-shadow: var(--shadow);
  transition: border-color 0.2s ease, transform 0.2s ease;
}
figure:hover {
  border-color: rgb(91 141 239 / 40%);
  transform: translateY(-1px);
}
figure img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 8px;
  background: #fff;
}
figcaption {
  margin-top: 0.65rem;
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--muted);
  text-align: center;
}
.empty-state {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--muted);
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}
footer {
  position: relative;
  z-index: 1;
  text-align: center;
  padding: 0 1.75rem 2rem;
  font-size: 0.75rem;
  color: var(--muted);
}
@media (max-width: 900px) {
  .cols-2 { grid-template-columns: 1fr; }
  header { padding: 1rem; }
  main { padding: 1rem 1rem 2.5rem; }
}
"""

_GALLERY_JS = """
(function () {
  const POLL_MS = 10000;
  const initialVersion = Number(document.body.dataset.version || "0");
  const initialEpochs = Number(document.body.dataset.epochs || "0");
  const follow = document.body.dataset.follow === "true";
  let knownVersion = initialVersion;
  let knownEpochs = initialEpochs;
  let refreshing = false;

  const liveBadge = document.getElementById("live-badge");
  const epochBadge = document.getElementById("epoch-badge");
  const footerUpdated = document.getElementById("footer-updated");

  function setRefreshing(on) {
    refreshing = on;
    if (!liveBadge) return;
    if (on) {
      liveBadge.className = "badge refreshing";
      liveBadge.innerHTML = "Updating&hellip;";
    } else if (follow) {
      liveBadge.className = "badge live";
      liveBadge.innerHTML = '<span class="dot"></span> Live';
    }
  }

  function updateEpochLabel(epochs, lastEpoch) {
    if (!epochBadge) return;
    const label = lastEpoch != null
      ? `${epochs} epoch${epochs === 1 ? "" : "s"} &middot; latest ${lastEpoch}`
      : `${epochs} epoch${epochs === 1 ? "" : "s"}`;
    epochBadge.innerHTML = label;
  }

  async function poll() {
    try {
      const res = await fetch("/api/status", { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      updateEpochLabel(data.epochs, data.last_epoch);

      const versionChanged = data.version !== knownVersion;
      const epochsChanged = data.epochs > knownEpochs;

      if (versionChanged || epochsChanged) {
        if (epochsChanged && data.version === knownVersion) {
          return;
        }
        knownVersion = data.version;
        knownEpochs = data.epochs;
        setRefreshing(true);
        window.location.reload();
        return;
      }

      if (footerUpdated && data.updated_at) {
        footerUpdated.textContent = "Last updated " + data.updated_at;
      }
    } catch (_) {
      /* server may be restarting */
    }
  }

  poll();
  window.setInterval(poll, POLL_MS);
})();
"""


def _epoch_summary(rows: list[dict[str, str]], columns: list[str]) -> tuple[int, float | None]:
    count = len(rows)
    if count == 0 or "epoch" not in columns:
        return count, None
    try:
        last = float(rows[-1].get("epoch", ""))
        if math.isfinite(last):
            return count, last
    except ValueError:
        pass
    return count, None


def _format_updated_at() -> str:
    return time.strftime("%H:%M:%S")


def _section_grid_class(section_title: str) -> str:
    cols = SECTION_LAYOUT.get(section_title, {"cols": 2})["cols"]
    return "cols-1" if cols == 1 else "cols-2"


def _build_gallery_html(
    sections: list[PanelSection],
    log_path: Path,
    *,
    image_prefix: str = "images",
    version: int | None = None,
    epochs: int = 0,
    last_epoch: float | None = None,
    follow: bool = False,
    updated_at: str | None = None,
) -> str:
    cache_suffix = f"?v={version}" if version is not None else ""
    page_title = f"Training log — {log_path.name}"
    version_attr = str(version if version is not None else 0)
    follow_attr = "true" if follow else "false"

    if last_epoch is not None:
        epoch_label = (
            f"{epochs} epoch{'s' if epochs != 1 else ''} · latest {last_epoch:g}"
        )
    else:
        epoch_label = f"{epochs} epoch{'s' if epochs != 1 else ''}"

    live_badge = ""
    if follow:
        live_badge = '<span class="badge live" id="live-badge"><span class="dot"></span> Live</span>'

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(page_title)}</title>",
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600'
        '&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">',
        f"<style>{_GALLERY_CSS}</style>",
        "</head>",
        f'<body data-version="{html.escape(version_attr)}" '
        f'data-epochs="{epochs}" data-follow="{follow_attr}">',
        "<header>",
        '<div class="header-inner">',
        '<div class="header-text">',
        f"<h1>{html.escape(page_title)}</h1>",
        f"<p>{html.escape(str(log_path))}</p>",
        "</div>",
        '<div class="header-meta">',
        f'<span class="badge epochs" id="epoch-badge">{html.escape(epoch_label)}</span>',
        live_badge,
        "</div>",
        "</div>",
        "</header>",
        "<main>",
    ]

    if not sections:
        parts.append(
            '<div class="empty-state">'
            "No plots yet — waiting for training data&hellip;"
            "</div>"
        )
    else:
        for section in sections:
            grid_class = _section_grid_class(section.title)
            parts.append("<section>")
            parts.append(f"<h2>{html.escape(section.title)}</h2>")
            parts.append(f'<div class="grid {grid_class}">')
            for panel in section.panels:
                src = f"{image_prefix}/{panel.filename}{cache_suffix}"
                parts.append("<figure>")
                parts.append(
                    f'<img src="{html.escape(src, quote=True)}" '
                    f'alt="{html.escape(panel.title)}" loading="lazy">'
                )
                parts.append(f"<figcaption>{html.escape(panel.title)}</figcaption>")
                parts.append("</figure>")
            parts.append("</div>")
            parts.append("</section>")

    parts.extend(
        [
            "</main>",
            "<footer>",
            f'<span id="footer-updated">Last updated {html.escape(updated_at or "—")}</span>',
            "</footer>",
            f"<script>{_GALLERY_JS}</script>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(parts)


def _write_gallery_html(
    sections: list[PanelSection],
    log_path: Path,
    out_dir: Path,
    *,
    version: int | None = None,
    epochs: int = 0,
    last_epoch: float | None = None,
    follow: bool = False,
    updated_at: str | None = None,
) -> Path:
    html_path = out_dir / "index.html"
    html_path.write_text(
        _build_gallery_html(
            sections,
            log_path,
            image_prefix=".",
            version=version,
            epochs=epochs,
            last_epoch=last_epoch,
            follow=follow,
            updated_at=updated_at,
        ),
        encoding="utf-8",
    )
    return html_path


def _gallery_version(log_path: Path) -> int:
    return log_path.stat().st_mtime_ns


def _regenerate_gallery(
    log_path: Path,
    out_dir: Path,
    *,
    quiet: bool = False,
    follow: bool = False,
) -> tuple[list[PanelSection], int, int, float | None] | None:
    with _REGEN_LOCK:
        try:
            columns, rows = _load_log(log_path, quiet=quiet)
            sections = _build_panel_sections(columns, rows)
            panels = _all_panels(sections)
            if not panels:
                return None
            version = _gallery_version(log_path)
            epochs, last_epoch = _epoch_summary(rows, columns)
            updated_at = _format_updated_at()
            saved = _save_panels(panels, out_dir)
            html_path = _write_gallery_html(
                sections,
                log_path,
                out_dir,
                version=version,
                epochs=epochs,
                last_epoch=last_epoch,
                follow=follow,
                updated_at=updated_at,
            )
            if quiet:
                typer.echo(
                    f"Updated gallery — {epochs} epoch(s); browser will refresh automatically"
                )
            else:
                typer.echo(f"Saved {len(saved)} plot(s) to {out_dir}:")
                for path in saved:
                    typer.echo(f"  {path}")
                typer.echo(f"  {html_path}")
            return sections, version, epochs, last_epoch
        except typer.BadParameter as exc:
            if quiet:
                typer.echo(f"Skip update — {exc}", err=True)
                return None
            raise


def _regenerate_into_state(
    log_path: Path,
    out_dir: Path,
    state: dict[str, object],
    *,
    quiet: bool,
    follow: bool,
) -> bool:
    result = _regenerate_gallery(log_path, out_dir, quiet=quiet, follow=follow)
    if result is None:
        return False
    sections, version, epochs, last_epoch = result
    state["sections"] = sections
    state["version"] = version
    state["epochs"] = epochs
    state["last_epoch"] = last_epoch
    state["updated_at"] = _format_updated_at()
    return True


def _follow_log(
    log_path: Path,
    out_dir: Path,
    stop_event: threading.Event,
    state: dict[str, object] | None = None,
    *,
    follow: bool = True,
) -> None:
    last_key: tuple[int, int] | None = None
    if log_path.is_file():
        stat = log_path.stat()
        last_key = (stat.st_mtime_ns, stat.st_size)

    typer.echo(f"Following {log_path} (Ctrl+C to stop)")

    while not stop_event.is_set():
        if not log_path.is_file():
            stop_event.wait(1.0)
            continue

        stat = log_path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        if key != last_key:
            if state is None:
                ok = (
                    _regenerate_gallery(log_path, out_dir, quiet=True, follow=follow)
                    is not None
                )
            else:
                ok = _regenerate_into_state(
                    log_path, out_dir, state, quiet=True, follow=follow
                )
            if ok:
                last_key = key

        stop_event.wait(1.0)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _open_html_gallery(
    log_path: Path,
    out_dir: Path,
    *,
    follow: bool = False,
    initial_sections: list[PanelSection] | None = None,
    initial_version: int | None = None,
    initial_epochs: int = 0,
    initial_last_epoch: float | None = None,
) -> None:
    port = _pick_free_port()
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"Gallery server: {url} (Ctrl+C to stop)")
    if follow:
        typer.echo("Follow mode — plots regenerate when the log changes; page auto-refreshes")

    state: dict[str, object] = {
        "sections": initial_sections or [],
        "version": initial_version,
        "epochs": initial_epochs,
        "last_epoch": initial_last_epoch,
        "updated_at": _format_updated_at(),
        "follow": follow,
    }
    gallery = Flask(__name__)
    stop_event = threading.Event()

    @gallery.get("/")
    def index() -> str:
        sections = state["sections"]
        version = state.get("version")
        epochs = state.get("epochs", 0)
        last_epoch = state.get("last_epoch")
        updated_at = state.get("updated_at")
        if not sections:
            return _build_gallery_html(
                [],
                log_path,
                image_prefix="/images",
                version=version if isinstance(version, int) else None,
                epochs=epochs if isinstance(epochs, int) else 0,
                last_epoch=last_epoch if isinstance(last_epoch, (int, float)) else None,
                follow=follow,
                updated_at=updated_at if isinstance(updated_at, str) else None,
            )
        return _build_gallery_html(
            sections,
            log_path,
            image_prefix="/images",
            version=version if isinstance(version, int) else None,
            epochs=epochs if isinstance(epochs, int) else 0,
            last_epoch=last_epoch if isinstance(last_epoch, (int, float)) else None,
            follow=follow,
            updated_at=updated_at if isinstance(updated_at, str) else None,
        )

    @gallery.get("/api/status")
    def status():
        version = state.get("version")
        epochs = state.get("epochs", 0)
        last_epoch = state.get("last_epoch")
        updated_at = state.get("updated_at")
        return jsonify(
            {
                "version": version if isinstance(version, int) else 0,
                "epochs": epochs if isinstance(epochs, int) else 0,
                "last_epoch": last_epoch
                if isinstance(last_epoch, (int, float))
                else None,
                "updated_at": updated_at if isinstance(updated_at, str) else None,
                "follow": follow,
            }
        )

    @gallery.get("/images/<path:filename>")
    def image(filename: str):
        return send_from_directory(out_dir, filename)

    if follow:
        watcher = threading.Thread(
            target=_follow_log,
            args=(log_path, out_dir, stop_event, state),
            kwargs={"follow": follow},
            daemon=True,
        )
        watcher.start()

    webbrowser.open(url)
    try:
        gallery.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    finally:
        stop_event.set()


@app.command()
def main(
    log_file: Annotated[
        Path,
        typer.Argument(help="Training CSV log (e.g. train/logs/training_log.csv)"),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output-dir",
            help="Directory for PNG plots (default: <log_dir>/plots/<log_stem>)",
        ),
    ] = None,
    show: Annotated[
        bool,
        typer.Option("--show/--no-show", help="Open HTML gallery in browser"),
    ] = True,
    follow: Annotated[
        bool,
        typer.Option(
            "-f",
            "--follow",
            help="Watch log file and regenerate plots (browser auto-refreshes on new epochs)",
        ),
    ] = False,
) -> None:
    """Plot losses, learning rate, and metrics from a training log CSV."""
    log_path = _resolve_path(log_file)

    if output_dir is None:
        out_dir = log_path.parent / "plots" / log_path.stem
    else:
        out_dir = _resolve_path(output_dir)

    if follow and not log_path.is_file():
        typer.echo(f"Waiting for log file: {log_path}")
        while not log_path.is_file():
            time.sleep(0.5)
    elif not log_path.is_file():
        raise typer.BadParameter(f"Log file not found: {log_path}")

    result = _regenerate_gallery(log_path, out_dir, quiet=False, follow=follow)
    if result is None:
        raise typer.BadParameter(f"No plottable numeric columns found in {log_path}")
    sections, version, epochs, last_epoch = result

    if follow and not show:
        stop_event = threading.Event()
        try:
            _follow_log(log_path, out_dir, stop_event, follow=follow)
        except KeyboardInterrupt:
            stop_event.set()
        return

    if show:
        _open_html_gallery(
            log_path,
            out_dir,
            follow=follow,
            initial_sections=sections,
            initial_version=version,
            initial_epochs=epochs,
            initial_last_epoch=last_epoch,
        )


if __name__ == "__main__":
    app()
