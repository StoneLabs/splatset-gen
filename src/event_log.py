"""Cross-process event sink for live UI (no imports from parallel/sample/render)."""

from __future__ import annotations

from typing import Any

_queue: Any = None
_worker_slot: int = 0
_sample_id: str = ""


def bind(queue: Any, worker_slot: int = 0) -> None:
    global _queue, _worker_slot
    _queue = queue
    _worker_slot = worker_slot


def set_sample(sample_id: str) -> None:
    global _sample_id
    _sample_id = sample_id


def clear() -> None:
    global _queue, _sample_id
    _queue = None
    _sample_id = ""


def is_active() -> bool:
    return _queue is not None


def emit(kind: str, *payload: Any) -> None:
    if _queue is not None:
        _queue.put((kind, *payload))


def log(message: str) -> None:
    if _sample_id:
        emit("log", _worker_slot, f"[cyan]{_sample_id}[/]  {message}")
    else:
        emit("log", _worker_slot, message)


def render_progress(pct: float) -> None:
    """Report rasterizer completion percentage for the live worker table."""
    emit("render", _worker_slot, float(pct))


def render_gaussians(count: int) -> None:
    """Report Gaussian count for gaussians/sec in the live worker table."""
    emit("gaussians", _worker_slot, int(count))


def worker_status(phase: str, detail: str = "") -> None:
    """Update worker table phase/detail (verbose UI)."""
    emit("status", _worker_slot, phase, detail)
