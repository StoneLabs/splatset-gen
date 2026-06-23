"""Shared PLY cache with reference counting across worker processes."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from multiprocessing.managers import SyncManager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ply_loader import PlyLoadStats, SceneGaussians, _load_ply_from_disk

CACHE_WORKER_ID = -1
EVICT_DELAY_S = 5.0


def _stats_to_dict(stats: PlyLoadStats) -> dict[str, Any]:
    return {
        "extent_before": stats.extent_before,
        "max_extent_before": stats.max_extent_before,
        "extent_after": stats.extent_after,
        "max_extent_after": stats.max_extent_after,
    }


def _stats_from_dict(data: dict[str, Any]) -> PlyLoadStats:
    return PlyLoadStats(
        extent_before=tuple(data["extent_before"]),
        max_extent_before=float(data["max_extent_before"]),
        extent_after=tuple(data["extent_after"]),
        max_extent_after=float(data["max_extent_after"]),
    )


def _gaussians_to_arrays(gaussians: SceneGaussians) -> dict[str, np.ndarray]:
    return {
        "means": gaussians.means.detach().cpu().numpy(),
        "quats": gaussians.quats.detach().cpu().numpy(),
        "scales": gaussians.scales.detach().cpu().numpy(),
        "opacities": gaussians.opacities.detach().cpu().numpy(),
        "sh_dc": gaussians.sh_dc.detach().cpu().numpy(),
        "sh_rest": gaussians.sh_rest.detach().cpu().numpy(),
        "object_ids": gaussians.object_ids.detach().cpu().numpy(),
    }


def _arrays_to_gaussians(arrays: dict[str, np.ndarray], *, copy: bool = False) -> SceneGaussians:
    def _tensor(name: str) -> torch.Tensor:
        arr = arrays[name].copy() if copy else arrays[name]
        return torch.from_numpy(arr)

    return SceneGaussians(
        means=_tensor("means"),
        quats=_tensor("quats"),
        scales=_tensor("scales"),
        opacities=_tensor("opacities"),
        sh_dc=_tensor("sh_dc"),
        sh_rest=_tensor("sh_rest"),
        object_ids=_tensor("object_ids"),
    )


def _field_memory_bytes(meta: dict[str, dict[str, Any]]) -> int:
    total = 0
    for field in meta.values():
        if field.get("empty"):
            continue
        shape = field["shape"]
        total += int(np.prod(shape)) * np.dtype(field["dtype"]).itemsize
    return total


def _store_arrays(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for name, arr in arrays.items():
        if arr.nbytes == 0:
            meta[name] = {"empty": True, "shape": arr.shape, "dtype": str(arr.dtype)}
            continue
        # Let Python assign the segment name — custom names like
        # ``splat_ply_<uuid>_opacities`` exceed macOS POSIX shm limits (~31 chars).
        shm = SharedMemory(create=True, size=arr.nbytes)
        buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        buf[:] = arr
        meta[name] = {
            "name": shm.name,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }
        shm.close()
    return meta


def _load_arrays(
    meta: dict[str, dict[str, Any]],
    *,
    copy: bool = False,
) -> tuple[dict[str, np.ndarray], list[SharedMemory]]:
    """Attach to cached shared memory.

    When ``copy`` is false, returned arrays are views into shared segments; keep
    the returned ``SharedMemory`` handles alive until the worker releases them.
    """
    arrays: dict[str, np.ndarray] = {}
    shms: list[SharedMemory] = []
    for name, field in meta.items():
        if field.get("empty"):
            arrays[name] = np.empty(field["shape"], dtype=np.dtype(field["dtype"]))
            continue
        shm = SharedMemory(name=field["name"])
        shms.append(shm)
        view = np.ndarray(field["shape"], dtype=np.dtype(field["dtype"]), buffer=shm.buf)
        arrays[name] = view.copy() if copy else view
    if copy:
        for shm in shms:
            shm.close()
        shms = []
    return arrays, shms


def _unlink_fields(meta: dict[str, dict[str, Any]]) -> None:
    for field in meta.values():
        if field.get("empty"):
            continue
        try:
            shm = SharedMemory(name=field["name"])
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


@dataclass
class _CacheEntry:
    ref_count: int
    fields: dict[str, dict[str, Any]]
    stats: dict[str, Any]
    display_name: str
    memory_bytes: int
    status: str = "loaded"
    evict_at: float | None = None


class _PlyCacheCore:
    """Runs in the manager server process; owns shared memory for cached PLYs."""

    def __init__(self, progress_queue: Any = None) -> None:
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._progress_queue = progress_queue
        self._stop = threading.Event()
        self._evict_thread = threading.Thread(target=self._eviction_loop, daemon=True)
        self._evict_thread.start()

    def set_progress_queue(self, progress_queue: Any) -> None:
        self._progress_queue = progress_queue

    def shutdown(self) -> None:
        self._stop.set()
        self._evict_thread.join(timeout=1.0)

    def _emit(self, kind: str, *payload: Any) -> None:
        if self._progress_queue is not None:
            self._progress_queue.put((kind, *payload))

    def is_resident(self, path: str) -> bool:
        resolved = str(Path(path).resolve())
        with self._lock:
            return resolved in self._entries

    def acquire(self, path: str, worker_id: int) -> dict[str, Any]:
        resolved = str(Path(path).resolve())
        display_name = Path(resolved).name
        with self._lock:
            entry = self._entries.get(resolved)
            if entry is not None:
                entry.evict_at = None
                entry.ref_count += 1
                entry.status = "loaded"
                self._emit("cache_claim", worker_id, display_name, entry.ref_count)
                return {"fields": entry.fields, "stats": entry.stats}

            self._entries[resolved] = _CacheEntry(
                ref_count=1,
                fields={},
                stats={},
                display_name=display_name,
                memory_bytes=0,
                status="loading",
            )
            self._emit("cache_claim", worker_id, display_name, 1)

            gaussians, stats = _load_ply_from_disk(Path(resolved))
            fields = _store_arrays(_gaussians_to_arrays(gaussians))
            entry = self._entries[resolved]
            entry.fields = fields
            entry.stats = _stats_to_dict(stats)
            entry.memory_bytes = _field_memory_bytes(fields)
            entry.status = "loaded"
            return {"fields": fields, "stats": entry.stats}

    def release_paths(self, paths: list[str], worker_id: int) -> None:
        if not paths:
            return
        with self._lock:
            for path in paths:
                resolved = str(Path(path).resolve())
                entry = self._entries.get(resolved)
                if entry is None:
                    continue
                entry.ref_count = max(0, entry.ref_count - 1)
                self._emit("cache_release", worker_id, entry.display_name, entry.ref_count)
                if entry.ref_count == 0:
                    entry.evict_at = time.monotonic() + EVICT_DELAY_S
                    entry.status = "unloading"
                else:
                    entry.status = "loaded"

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            rows: list[dict[str, Any]] = []
            for entry in self._entries.values():
                timeout_s: float | None = None
                if entry.ref_count == 0 and entry.evict_at is not None:
                    timeout_s = max(0.0, entry.evict_at - now)
                rows.append(
                    {
                        "name": entry.display_name,
                        "refs": entry.ref_count,
                        "memory_bytes": entry.memory_bytes,
                        "timeout_s": timeout_s,
                        "status": entry.status,
                    }
                )
            rows.sort(key=lambda row: row["name"].lower())
            return rows

    def _eviction_loop(self) -> None:
        while not self._stop.wait(0.25):
            self._process_evictions()

    def _process_evictions(self) -> None:
        now = time.monotonic()
        to_evict: list[str] = []
        with self._lock:
            for resolved, entry in self._entries.items():
                if (
                    entry.ref_count == 0
                    and entry.evict_at is not None
                    and now >= entry.evict_at
                ):
                    to_evict.append(resolved)
            for resolved in to_evict:
                entry = self._entries.pop(resolved)
                _unlink_fields(entry.fields)
                self._emit("cache_evict", entry.display_name)

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                _unlink_fields(entry.fields)
            self._entries.clear()


_progress_queue_ref: Any = None


def _make_ply_cache() -> _PlyCacheCore:
    return _PlyCacheCore(_progress_queue_ref)


class PlyCacheManager(SyncManager):
    pass


def create_ply_cache_manager(progress_queue: Any) -> tuple[PlyCacheManager, Any]:
    """Start manager process with shared PLY cache service."""
    global _progress_queue_ref
    _progress_queue_ref = progress_queue
    PlyCacheManager.register("PlyCache", callable=_make_ply_cache)
    manager = PlyCacheManager()
    manager.start()
    cache = manager.PlyCache()
    cache.set_progress_queue(progress_queue)
    return manager, cache


def gaussians_from_handle(
    handle: dict[str, Any],
    *,
    copy: bool = False,
) -> tuple[SceneGaussians, PlyLoadStats, list[SharedMemory]]:
    """Build SceneGaussians from a cache handle.

    Default attaches to shared memory without copying tensor payload into the
    worker process. The returned ``SharedMemory`` handles must stay open until
    ``release_acquired()`` runs in the worker.
    """
    arrays, shms = _load_arrays(handle["fields"], copy=copy)
    return _arrays_to_gaussians(arrays, copy=False), _stats_from_dict(handle["stats"]), shms
