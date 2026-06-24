#!/usr/bin/env python3
"""Run dataset generation and report peak RSS by process role."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_MATCH = (
    "generate_dataset.py",
    "ply-manager",
    "ForkPoolWorker",
    "PoolWorker",
    "spawn_main",
    "ply_manager.py",
    "_manager_loop",
)


def _rss_kb(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def _cmdline(pid: int) -> str:
    raw = Path(f"/proc/{pid}/cmdline")
    if not raw.is_file():
        return ""
    return raw.read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore").strip()


def _related_processes() -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = _cmdline(pid)
        if not cmd:
            continue
        if any(token in cmd for token in _MATCH):
            hits.append((pid, cmd))
    return hits


def _role(cmd: str) -> str:
    if "ForkPoolWorker" in cmd or "PoolWorker" in cmd or "spawn_main" in cmd:
        return "worker"
    if "_manager_loop" in cmd or "ply_manager" in cmd or "ply-manager" in cmd:
        return "manager"
    if "generate_dataset.py" in cmd:
        return "main"
    return "other"


def main(argv: list[str]) -> int:
    cmd = argv[1:]
    if not cmd:
        print("usage: measure_generation_memory.py <generate_dataset args...>", file=sys.stderr)
        return 2

    proc = subprocess.Popen(cmd, cwd=ROOT)
    peak_by_pid: dict[int, tuple[int, str]] = {}
    peak_instant_total_mb = 0.0
    samples = 0

    while proc.poll() is None:
        related = _related_processes()
        instant_total = 0
        for pid, cmdline in related:
            rss = _rss_kb(pid)
            instant_total += rss
            prev = peak_by_pid.get(pid)
            if prev is None or rss > prev[0]:
                peak_by_pid[pid] = (rss, cmdline)
        peak_instant_total_mb = max(peak_instant_total_mb, instant_total / 1024)
        samples += 1
        time.sleep(0.25)

    for pid, cmdline in _related_processes():
        rss = _rss_kb(pid)
        prev = peak_by_pid.get(pid)
        if prev is None or rss > prev[0]:
            peak_by_pid[pid] = (rss, cmdline)

    by_role: dict[str, list[float]] = {"main": [], "manager": [], "worker": [], "other": []}
    for rss_kb, cmdline in peak_by_pid.values():
        by_role[_role(cmdline)].append(rss_kb / 1024)

    print(f"samples={samples} exit={proc.returncode}")
    print(f"peak_instant_total_rss_mb={peak_instant_total_mb:.1f}")
    print(f"peak_sum_of_process_peaks_mb={sum(r for r, _ in peak_by_pid.values()) / 1024:.1f}")
    if by_role["main"]:
        print(f"peak_main_mb={max(by_role['main']):.1f}")
    if by_role["manager"]:
        print(f"peak_manager_mb={max(by_role['manager']):.1f}")
    if by_role["worker"]:
        worker_rss = by_role["worker"]
        print(f"peak_workers_mb={sum(worker_rss):.1f} count={len(worker_rss)}")
        print(f"peak_worker_avg_mb={sum(worker_rss) / len(worker_rss):.1f}")
        print(f"peak_worker_max_mb={max(worker_rss):.1f}")
    else:
        print("peak_workers_mb=0 count=0")

    for pid, (rss_kb, cmdline) in sorted(peak_by_pid.items(), key=lambda x: -x[1][0]):
        print(f"  pid={pid} peak_mb={rss_kb / 1024:.1f} role={_role(cmdline)} cmd={cmdline[:100]}")

    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
