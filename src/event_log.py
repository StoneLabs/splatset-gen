"""Cross-process event sink for live UI (no imports from parallel/sample/render)."""

from __future__ import annotations

import time
from queue import Full
from typing import Any

_queue: Any = None
_worker_slot: int = 0
_sample_id: str = ""
_last_render_pct: float = -1.0
_last_render_emit_at: float = 0.0

RENDER_EMIT_MIN_INTERVAL_S = 0.2
RENDER_EMIT_MIN_STEP_PCT = 5.0


def bind(queue: Any, worker_slot: int = 0) -> None:
    global _queue, _worker_slot, _last_render_pct, _last_render_emit_at
    _queue = queue
    _worker_slot = worker_slot
    _last_render_pct = -1.0
    _last_render_emit_at = 0.0


def set_sample(sample_id: str) -> None:
    global _sample_id, _last_render_pct, _last_render_emit_at
    _sample_id = sample_id
    _last_render_pct = -1.0
    _last_render_emit_at = 0.0


def clear() -> None:
    global _queue, _sample_id, _last_render_pct, _last_render_emit_at
    _queue = None
    _sample_id = ""
    _last_render_pct = -1.0
    _last_render_emit_at = 0.0


def is_active() -> bool:
    return _queue is not None


def emit(kind: str, *payload: Any, drop_if_full: bool = False) -> None:
    if _queue is None:
        return
    msg = (kind, *payload)
    if drop_if_full:
        try:
            _queue.put_nowait(msg)
        except Full:
            return
    else:
        _queue.put(msg)


def log(message: str) -> None:
    if _sample_id:
        emit("log", _worker_slot, f"[cyan]{_sample_id}[/]  {message}")
    else:
        emit("log", _worker_slot, message)


def render_progress(pct: float) -> None:
    """Report rasterizer completion percentage for the live worker table."""
    global _last_render_pct, _last_render_emit_at
    pct = float(pct)
    now = time.perf_counter()
    if pct < 100.0:
        if (
            now - _last_render_emit_at < RENDER_EMIT_MIN_INTERVAL_S
            and pct - _last_render_pct < RENDER_EMIT_MIN_STEP_PCT
        ):
            return
    _last_render_emit_at = now
    _last_render_pct = pct
    emit("render", _worker_slot, pct, drop_if_full=True)


def render_gaussians(count: int) -> None:
    """Report Gaussian count for gaussians/sec in the live worker table."""
    emit("gaussians", _worker_slot, int(count))


def worker_status(phase: str, detail: str = "") -> None:
    """Update worker table phase/detail."""
    emit("status", _worker_slot, phase, detail, drop_if_full=True)
