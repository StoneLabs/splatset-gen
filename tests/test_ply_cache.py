"""Tests for shared PLY cache."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ply_cache import EVICT_DELAY_S, _PlyCacheCore, gaussians_from_handle
from ply_loader import bind_cache, clear_cache_binding, load_ply, release_acquired, write_synthetic_ply


def test_cache_acquire_release_evicts(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=8)
    cache = _PlyCacheCore()

    handle = cache.acquire(str(ply_path), worker_id=0)
    assert cache.count() == 1

    gaussians, _stats, shms = gaussians_from_handle(handle)
    try:
        assert gaussians.num_gaussians == 8
    finally:
        for shm in shms:
            shm.close()

    cache.release_paths([str(ply_path)], worker_id=0)
    assert cache.count() == 1
    row = cache.snapshot()[0]
    assert row["refs"] == 0
    assert row["status"] == "unloading"
    assert row["timeout_s"] is not None

    time.sleep(EVICT_DELAY_S + 0.5)
    assert cache.count() == 0
    cache.shutdown()


def test_cache_ref_count_keeps_entry(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=4)
    cache = _PlyCacheCore()

    cache.acquire(str(ply_path), worker_id=0)
    cache.acquire(str(ply_path), worker_id=1)
    assert cache.count() == 1
    assert cache.snapshot()[0]["refs"] == 2

    cache.release_paths([str(ply_path)], worker_id=0)
    assert cache.count() == 1
    assert cache.snapshot()[0]["refs"] == 1

    cache.release_paths([str(ply_path)], worker_id=1)
    assert cache.count() == 1
    assert cache.snapshot()[0]["refs"] == 0
    cache.shutdown()


def test_load_ply_uses_cache_proxy(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=6)
    cache = _PlyCacheCore()

    class _Proxy:
        def is_resident(self, path: str) -> bool:
            return cache.is_resident(path)

        def acquire(self, path: str, worker_id: int) -> dict:
            return cache.acquire(path, worker_id)

        def release_paths(self, paths: list[str], worker_id: int) -> None:
            cache.release_paths(paths, worker_id)

    bind_cache(_Proxy())
    try:
        scene_a, _ = load_ply(ply_path)
        scene_b, _ = load_ply(ply_path)
        assert cache.count() == 1
        assert scene_a.num_gaussians == scene_b.num_gaussians == 6
        np.testing.assert_allclose(scene_a.means.numpy(), scene_b.means.numpy())
    finally:
        release_acquired()
        clear_cache_binding()
    assert cache.snapshot()[0]["refs"] == 0
    cache.shutdown()


def test_zero_copy_views_share_storage(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=4)
    cache = _PlyCacheCore()
    handle = cache.acquire(str(ply_path), worker_id=0)

    gauss_a, _stats, shms_a = gaussians_from_handle(handle)
    gauss_b, _, shms_b = gaussians_from_handle(handle)
    try:
        gauss_a.means[0, 0] = 123.0
        assert float(gauss_b.means[0, 0].item()) == 123.0
    finally:
        for shm in shms_a + shms_b:
            shm.close()
    cache.release_paths([str(ply_path)], worker_id=0)
    cache.shutdown()


def test_reacquire_cancels_pending_eviction(tmp_path: Path) -> None:
    ply_path = write_synthetic_ply(tmp_path / "object.ply", num_gaussians=4)
    cache = _PlyCacheCore()
    path = str(ply_path)

    cache.acquire(path, worker_id=0)
    cache.release_paths([path], worker_id=0)
    assert cache.snapshot()[0]["timeout_s"] is not None

    cache.acquire(path, worker_id=1)
    row = cache.snapshot()[0]
    assert row["refs"] == 1
    assert row["status"] == "loaded"
    assert row["timeout_s"] is None

    cache.shutdown()
