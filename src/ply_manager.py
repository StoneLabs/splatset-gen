"""Shared PLY cache process for parallel dataset generation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch.multiprocessing as mp

from memutil import self_rss_kb
from ply_loader import (
    PlyLoadStats,
    SceneGaussians,
    load_ply_full,
    share_gaussians,
)

ACQUIRE = "acquire"
RELEASE = "release"
STOP = "stop"
SNAPSHOT = "snapshot"


@dataclass
class CacheEntryView:
    name: str
    size_mb: float
    ref_count: int
    status: str
    evict_in_sec: float | None


@dataclass
class CacheSnapshot:
    entries: list[CacheEntryView] = field(default_factory=list)
    cache_rss_mb: float = 0.0
    loading: list[str] = field(default_factory=list)


@dataclass
class _CacheEntry:
    gaussians: SceneGaussians
    stats: PlyLoadStats
    total_gaussians: int
    local_lo: np.ndarray
    local_hi: np.ndarray
    ref_count: int = 0
    last_used: float = 0.0


def _entry_size_mb(entry: _CacheEntry) -> float:
    total = 0
    for tensor in (
        entry.gaussians.means,
        entry.gaussians.quats,
        entry.gaussians.scales,
        entry.gaussians.opacities,
        entry.gaussians.sh_dc,
        entry.gaussians.sh_rest,
        entry.gaussians.object_ids,
    ):
        total += tensor.untyped_storage().nbytes()
    return total / (1024 * 1024)


class PlyManagerClient:
    """Worker-side RPC client for the shared PLY cache."""

    def __init__(self, request_queue: Any, response_queue: Any, worker_id: int) -> None:
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._worker_id = worker_id
        self._next_req_id = 0

    def acquire(self, path: str | Path) -> tuple[SceneGaussians, PlyLoadStats, int, np.ndarray, np.ndarray]:
        path_key = str(Path(path).resolve())
        req_id = self._next_req_id
        self._next_req_id += 1
        self._request_queue.put((ACQUIRE, self._worker_id, req_id, path_key))
        kind, resp_req_id, payload = self._response_queue.get()
        if kind != ACQUIRE or resp_req_id != req_id:
            raise RuntimeError(f"Unexpected PLY manager response: {kind!r} {resp_req_id!r}")
        if payload is None:
            raise RuntimeError(f"PLY manager failed to load {path_key}")
        gaussians, stats, total_gaussians, local_lo, local_hi = payload
        return gaussians, stats, total_gaussians, local_lo, local_hi

    def release(self, path: str | Path) -> None:
        path_key = str(Path(path).resolve())
        self._request_queue.put((RELEASE, path_key))


_client: PlyManagerClient | None = None


def bind(client: PlyManagerClient | None) -> None:
    global _client
    _client = client


def clear() -> None:
    bind(None)


def is_active() -> bool:
    return _client is not None


def acquire(path: str | Path) -> tuple[SceneGaussians, PlyLoadStats, int, np.ndarray, np.ndarray]:
    if _client is None:
        raise RuntimeError("PLY manager client is not bound")
    return _client.acquire(path)


def release(path: str | Path) -> None:
    if _client is not None:
        _client.release(path)


def _log_event(
    log_file: TextIO | None,
    event: str,
    path_key: str,
    ref_count: int,
    cache_size: int,
    *,
    worker_id: int | None = None,
    cached: str | None = None,
    idle_sec: float | None = None,
) -> None:
    if log_file is None:
        return
    name = Path(path_key).name
    parts = [
        f"t={time.monotonic():.3f}",
        event,
        f"path={name}",
        f"ref={ref_count}",
        f"cache_size={cache_size}",
    ]
    if worker_id is not None:
        parts.append(f"worker={worker_id}")
    if cached is not None:
        parts.append(f"cached={cached}")
    if idle_sec is not None:
        parts.append(f"idle_sec={idle_sec:.2f}")
    log_file.write(" ".join(parts) + "\n")
    log_file.flush()


def _ui_cache_log(ui_queue: Any | None, message: str) -> None:
    if ui_queue is not None:
        ui_queue.put(("cache_log", message))


def _build_snapshot(
    cache: dict[str, _CacheEntry],
    loading: set[str],
    cache_ttl_sec: float,
    now: float,
) -> CacheSnapshot:
    entries: list[CacheEntryView] = []
    for key in sorted(loading):
        entries.append(
            CacheEntryView(
                name=Path(key).name,
                size_mb=0.0,
                ref_count=0,
                status="loading",
                evict_in_sec=None,
            )
        )
    for key, entry in sorted(cache.items(), key=lambda kv: kv[0]):
        if key in loading:
            continue
        idle = now - entry.last_used if entry.ref_count == 0 else 0.0
        if entry.ref_count > 0:
            status = "loaded"
            evict_in: float | None = None
        else:
            status = "wait TTL"
            evict_in = max(0.0, cache_ttl_sec - idle)
        entries.append(
            CacheEntryView(
                name=Path(key).name,
                size_mb=_entry_size_mb(entry),
                ref_count=entry.ref_count,
                status=status,
                evict_in_sec=evict_in,
            )
        )
    return CacheSnapshot(
        entries=entries,
        cache_rss_mb=self_rss_kb() / 1024,
        loading=[Path(k).name for k in sorted(loading)],
    )


def _manager_loop(
    request_queue: Any,
    response_queues: list[Any],
    snapshot_queue: Any,
    ui_queue: Any | None,
    cache_ttl_sec: float,
    eviction_interval_sec: float,
    stats_path: str | None,
    verbose: bool,
) -> None:
    import sys

    sys.argv[0] = "ply-manager"
    cache: dict[str, _CacheEntry] = {}
    loading: set[str] = set()
    lock = threading.Lock()
    stop_event = threading.Event()
    log_file: TextIO | None = None
    if stats_path:
        stats_file = Path(stats_path)
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        log_file = stats_file.open("a", encoding="utf-8")
        log_file.write(
            f"# ply cache ttl={cache_ttl_sec}s eviction_interval={eviction_interval_sec}s\n"
        )
        log_file.flush()

    def debug(msg: str) -> None:
        if verbose:
            _ui_cache_log(ui_queue, msg)

    def evict_idle() -> None:
        while not stop_event.wait(eviction_interval_sec):
            now = time.monotonic()
            with lock:
                stale: list[tuple[str, float]] = []
                for key, entry in cache.items():
                    if entry.ref_count == 0:
                        idle = now - entry.last_used
                        if idle > cache_ttl_sec:
                            stale.append((key, idle))
                for key, idle in stale:
                    entry = cache.pop(key)
                    _log_event(
                        log_file,
                        "evict",
                        key,
                        entry.ref_count,
                        len(cache),
                        idle_sec=idle,
                    )
                    debug(
                        f"[red]cache[/] evict {Path(key).name} "
                        f"idle={idle:.1f}s refs={entry.ref_count}"
                    )

    eviction_thread = threading.Thread(target=evict_idle, daemon=True)
    eviction_thread.start()

    while True:
        msg = request_queue.get()
        kind = msg[0]

        if kind == STOP:
            stop_event.set()
            if log_file is not None:
                with lock:
                    _log_event(log_file, "stop", "-", 0, len(cache))
                log_file.close()
            break

        if kind == SNAPSHOT:
            _, req_id = msg
            with lock:
                snap = _build_snapshot(cache, loading, cache_ttl_sec, time.monotonic())
            snapshot_queue.put((SNAPSHOT, req_id, snap))
            continue

        if kind == RELEASE:
            _, path_key = msg
            now = time.monotonic()
            with lock:
                entry = cache.get(path_key)
                if entry is not None:
                    entry.ref_count = max(0, entry.ref_count - 1)
                    entry.last_used = now
                    _log_event(
                        log_file,
                        "release",
                        path_key,
                        entry.ref_count,
                        len(cache),
                    )
                    debug(
                        f"[red]cache[/] release {Path(path_key).name} "
                        f"ref={entry.ref_count}"
                    )
                    if cache_ttl_sec <= 0 and entry.ref_count == 0:
                        del cache[path_key]
                        _log_event(
                            log_file,
                            "evict_immediate",
                            path_key,
                            0,
                            len(cache),
                        )
                        debug(f"[red]cache[/] evict_immediate {Path(path_key).name}")
            continue

        if kind == ACQUIRE:
            _, worker_id, req_id, path_key = msg
            resp_q = response_queues[worker_id]
            try:
                with lock:
                    entry = cache.get(path_key)
                    if entry is None:
                        loading.add(path_key)
                        debug(
                            f"[red]cache[/] loading {Path(path_key).name} "
                            f"worker=#{worker_id}"
                        )
                        lock.release()
                        try:
                            gaussians, stats, total = load_ply_full(path_key)
                        finally:
                            lock.acquire()
                            loading.discard(path_key)
                        share_gaussians(gaussians)
                        lo = gaussians.means.min(dim=0).values.cpu().numpy()
                        hi = gaussians.means.max(dim=0).values.cpu().numpy()
                        entry = _CacheEntry(
                            gaussians=gaussians,
                            stats=stats,
                            total_gaussians=total,
                            local_lo=lo,
                            local_hi=hi,
                        )
                        cache[path_key] = entry
                        cached = "miss"
                    else:
                        cached = "hit"
                    entry.ref_count += 1
                    entry.last_used = time.monotonic()
                    _log_event(
                        log_file,
                        "acquire",
                        path_key,
                        entry.ref_count,
                        len(cache),
                        worker_id=worker_id,
                        cached=cached,
                    )
                    debug(
                        f"[red]cache[/] acquire {Path(path_key).name} "
                        f"ref={entry.ref_count} {cached} worker=#{worker_id}"
                    )
                    payload = (
                        entry.gaussians,
                        entry.stats,
                        entry.total_gaussians,
                        entry.local_lo,
                        entry.local_hi,
                    )

                resp_q.put((ACQUIRE, req_id, payload))
            except Exception:
                with lock:
                    loading.discard(path_key)
                resp_q.put((ACQUIRE, req_id, None))
            continue


class PlyManagerServer:
    """Owns the cache process and per-worker response queues."""

    def __init__(
        self,
        workers: int,
        cache_ttl_sec: float = 30.0,
        eviction_interval_sec: float = 5.0,
        stats_path: Path | None = None,
        ui_queue: Any | None = None,
        verbose: bool = False,
    ) -> None:
        self._workers = workers
        self._cache_ttl_sec = cache_ttl_sec
        self._eviction_interval_sec = eviction_interval_sec
        self._stats_path = str(stats_path) if stats_path is not None else None
        self._ui_queue = ui_queue
        self._verbose = verbose
        self.request_queue: Any = mp.Queue()
        self.response_queues: list[Any] = [mp.Queue() for _ in range(workers)]
        self.snapshot_queue: Any = mp.Queue()
        self._process: Any = None
        self._snapshot_id = 0

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = mp.Process(
            target=_manager_loop,
            args=(
                self.request_queue,
                self.response_queues,
                self.snapshot_queue,
                self._ui_queue,
                self._cache_ttl_sec,
                self._eviction_interval_sec,
                self._stats_path,
                self._verbose,
            ),
            daemon=True,
        )
        self._process.start()

    def client_for(self, worker_id: int) -> PlyManagerClient:
        return PlyManagerClient(
            self.request_queue,
            self.response_queues[worker_id],
            worker_id,
        )

    def fetch_snapshot(self) -> CacheSnapshot | None:
        if self._process is None or not self._process.is_alive():
            return None
        req_id = self._snapshot_id
        self._snapshot_id += 1
        self.request_queue.put((SNAPSHOT, req_id))
        try:
            while True:
                kind, resp_id, snap = self.snapshot_queue.get(timeout=0.05)
                if kind == SNAPSHOT and resp_id == req_id:
                    return snap
        except Exception:
            return None

    def stop(self) -> None:
        if self._process is None:
            return
        self.request_queue.put((STOP,))
        self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._process = None
