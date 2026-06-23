"""Multiprocessing worker pool for dataset generation."""

from __future__ import annotations

import os
import threading
import time
from multiprocessing import Manager, Pool
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch
import yaml

import event_log
from console import ProgressTracker, build_live_render, print_summary
from export import save_config_snapshot
from rich.live import Live
from sample import generate_one_sample

_PROGRESS_QUEUE: Any = None
_WORKER_SLOT: int = 0


def _pool_init(progress_queue: Any, slot_counter: Any, slot_lock: Any) -> None:
    global _PROGRESS_QUEUE, _WORKER_SLOT
    _PROGRESS_QUEUE = progress_queue
    with slot_lock:
        _WORKER_SLOT = slot_counter.value
        slot_counter.value += 1
    event_log.bind(progress_queue, _WORKER_SLOT)


def _emit(kind: str, *payload: Any) -> None:
    if _PROGRESS_QUEUE is not None:
        _PROGRESS_QUEUE.put((kind, *payload))


def _worker(
    args: tuple[int, int, dict[str, Any], list[str], str, bool],
) -> str:
    sample_index, seed, config, ply_paths_str, output_dir_str, verbose = args
    worker_id = _WORKER_SLOT
    rng = np.random.default_rng(seed + worker_id * 10_007 + sample_index)

    workers = int(config.get("_workers", 1))
    threads = max(1, (os.cpu_count() or 1) // max(workers, 1))
    torch.set_num_threads(threads)

    sample_id = f"{sample_index:06d}"
    _emit("start", worker_id, sample_id)

    t0 = time.perf_counter()
    try:
        generate_one_sample(
            [Path(p) for p in ply_paths_str],
            config,
            rng,
            Path(output_dir_str),
            sample_id,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        _emit("done", worker_id, sample_id, elapsed, None)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _emit("done", worker_id, sample_id, elapsed, str(exc))
        raise

    return sample_id


def _drain_queue(queue: Any, tracker: ProgressTracker) -> None:
    while True:
        try:
            msg = queue.get_nowait()
        except Empty:
            break
        kind = msg[0]
        if kind == "start":
            _, worker_id, sample_id = msg
            tracker.on_start(worker_id, sample_id)
        elif kind == "done":
            _, worker_id, sample_id, elapsed, error = msg
            tracker.on_done(worker_id, sample_id, elapsed, error)
        elif kind == "log":
            _, _worker_id, message = msg
            tracker.on_log(message)


def _run_with_live(
    tracker: ProgressTracker,
    progress_queue: Any,
    run_fn: Any,
) -> list[str]:
    """Run ``run_fn`` while a background thread pumps progress events to the UI."""
    with Live(build_live_render(tracker), refresh_per_second=10, transient=False) as live:
        stop = threading.Event()

        def pump() -> None:
            while not stop.is_set():
                _drain_queue(progress_queue, tracker)
                tracker.tick()
                live.update(build_live_render(tracker))
                stop.wait(0.08)

        pump_thread = threading.Thread(target=pump, daemon=True)
        pump_thread.start()
        try:
            results = run_fn()
        finally:
            stop.set()
            pump_thread.join(timeout=2.0)
            _drain_queue(progress_queue, tracker)
            tracker.tick()
            live.update(build_live_render(tracker))

    return results


def generate_dataset_parallel(
    ply_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    num_samples: int,
    workers: int,
    seed: int,
    verbose: bool = False,
    show_progress: bool = True,
) -> list[str]:
    """Generate ``num_samples`` in parallel; write config snapshot once."""
    ply_paths = sorted(ply_dir.glob("*.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No .ply files in {ply_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(output_dir, config)

    config = dict(config)
    config["_workers"] = workers
    ply_paths_str = [str(p) for p in ply_paths]

    tasks = [
        (i + 1, seed, config, ply_paths_str, str(output_dir), verbose)
        for i in range(num_samples)
    ]

    tracker = ProgressTracker(num_samples=num_samples, workers=workers)
    t0 = time.perf_counter()

    if not show_progress:
        if workers <= 1:
            return [_worker(t) for t in tasks]
        with Pool(processes=workers) as pool:
            return pool.map(_worker, tasks)

    manager = Manager()
    progress_queue = manager.Queue()
    slot_counter = manager.Value("i", 0)
    slot_lock = manager.Lock()

    def run_sequential() -> list[str]:
        global _PROGRESS_QUEUE, _WORKER_SLOT
        _PROGRESS_QUEUE = progress_queue
        _WORKER_SLOT = 0
        event_log.bind(progress_queue, 0)
        out: list[str] = []
        for task in tasks:
            out.append(_worker(task))
        _PROGRESS_QUEUE = None
        event_log.clear()
        return out

    def run_pool() -> list[str]:
        with Pool(
            processes=workers,
            initializer=_pool_init,
            initargs=(progress_queue, slot_counter, slot_lock),
        ) as pool:
            return list(pool.imap_unordered(_worker, tasks))

    if workers <= 1:
        results = _run_with_live(tracker, progress_queue, run_sequential)
    else:
        results = _run_with_live(tracker, progress_queue, run_pool)

    elapsed = time.perf_counter() - t0
    print_summary(output_dir, tracker.completed, tracker.failed, elapsed)
    return sorted(results)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)
