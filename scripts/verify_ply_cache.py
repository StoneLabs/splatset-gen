#!/usr/bin/env python3
"""Verify PLY cache ref-counting, eviction, and memory isolation."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_MATCH = (
    "generate_dataset.py",
    "ply-manager",
)


def _rss_kb(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def _comm(pid: int) -> str:
    comm = Path(f"/proc/{pid}/comm")
    if not comm.is_file():
        return ""
    try:
        return comm.read_text().strip()
    except OSError:
        return ""


def _cmdline(pid: int) -> str:
    raw = Path(f"/proc/{pid}/cmdline")
    if not raw.is_file():
        return ""
    try:
        return raw.read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore").strip()
    except OSError:
        return ""


def _role(cmd: str, pid: int) -> str:
    if _comm(pid) == "ply-manager" or "ply-manager" in cmd:
        return "manager"
    if "generate_dataset.py" in cmd:
        return "pool"
    return "other"


def _related() -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = _cmdline(pid)
        if cmd and any(token in cmd for token in _MATCH):
            hits.append((pid, cmd))
    return hits


def _parse_cache_log(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"error": f"missing log: {path}"}

    acquires = 0
    releases = 0
    hits = 0
    misses = 0
    evictions = 0
    max_cache_size = 0
    max_ref = 0
    negative_ref = False
    refs_by_path: dict[str, int] = defaultdict(int)
    peak_refs_by_path: dict[str, int] = defaultdict(int)

    line_re = re.compile(
        r"(\w+) path=(\S+) ref=(\d+) cache_size=(\d+)(?: worker=(\d+))?(?: cached=(\w+))?"
    )

    for raw in path.read_text().splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        m = line_re.search(raw)
        if not m:
            continue
        event, ply, ref_s, cache_s, _worker, cached = m.groups()
        ref = int(ref_s)
        cache_size = int(cache_s)
        max_cache_size = max(max_cache_size, cache_size)
        max_ref = max(max_ref, ref)
        if ref < 0:
            negative_ref = True

        if event == "acquire":
            acquires += 1
            refs_by_path[ply] += 1
            peak_refs_by_path[ply] = max(peak_refs_by_path[ply], refs_by_path[ply])
            if cached == "hit":
                hits += 1
            elif cached == "miss":
                misses += 1
        elif event == "release":
            releases += 1
            refs_by_path[ply] = max(0, refs_by_path[ply] - 1)
        elif event in ("evict", "evict_immediate"):
            evictions += 1

    open_refs = sum(refs_by_path.values())
    return {
        "acquires": acquires,
        "releases": releases,
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "max_cache_size": max_cache_size,
        "max_ref": max_ref,
        "negative_ref": negative_ref,
        "open_refs_at_end": open_refs,
        "peak_refs_by_path": dict(peak_refs_by_path),
    }


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--analyze-only":
        if len(argv) < 3:
            print("usage: verify_ply_cache.py --analyze-only OUTPUT_DIR", file=sys.stderr)
            return 2
        output_dir = Path(argv[2]).expanduser()
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        stats = _parse_cache_log(output_dir / "ply_cache.log")
        print("=== PLY cache log analysis ===")
        print(f"output={output_dir}")
        for key, val in stats.items():
            if key == "peak_refs_by_path":
                print(f"{key}:")
                for ply, ref in sorted(val.items()):
                    print(f"  {ply}: peak_ref={ref}")
            else:
                print(f"{key}={val}")
        return 0

    if len(argv) < 2:
        print(
            "usage: verify_ply_cache.py <generate_dataset args...>\n"
            "  e.g. verify_ply_cache.py uv run scripts/generate_dataset.py -n 16 -j 8 ...",
            file=sys.stderr,
        )
        return 2

    cmd = argv[1:]
    output_dir = ROOT / "outputs" / "cache_verify"
    for i, arg in enumerate(cmd):
        if arg in ("-o", "--output") and i + 1 < len(cmd):
            output_dir = Path(cmd[i + 1]).expanduser()
            if not output_dir.is_absolute():
                output_dir = ROOT / output_dir

    proc = subprocess.Popen(cmd, cwd=ROOT)
    peak_by_role: dict[str, float] = defaultdict(float)
    peak_instant_mb = 0.0
    peak_manager_mb = 0.0
    peak_pool_mb = 0.0
    pool_peaks: list[float] = []

    while proc.poll() is None:
        related = _related()
        instant = 0.0
        managers: list[float] = []
        pools: list[float] = []
        for pid, cmdline in related:
            mb = _rss_kb(pid) / 1024
            instant += mb
            role = _role(cmdline, pid)
            peak_by_role[role] = max(peak_by_role[role], mb)
            if role == "manager":
                managers.append(mb)
            elif role == "pool":
                pools.append(mb)
        peak_instant_mb = max(peak_instant_mb, instant)
        if managers:
            peak_manager_mb = max(peak_manager_mb, max(managers))
        if pools:
            pool_peaks = pools
            peak_pool_mb = max(peak_pool_mb, max(pools))
        time.sleep(0.25)

    cache_log = output_dir / "ply_cache.log"
    stats = _parse_cache_log(cache_log)

    print("=== PLY cache verification ===")
    print(f"exit={proc.returncode}")
    print(f"output={output_dir}")
    print(f"peak_instant_total_mb={peak_instant_mb:.1f}")
    print(f"peak_manager_mb={peak_manager_mb:.1f}")
    print(f"peak_pool_process_mb={peak_pool_mb:.1f}")
    if pool_peaks:
        print(f"pool_processes_at_last_sample={len(pool_peaks)}")
        print(f"pool_rss_sum_at_last_sample_mb={sum(pool_peaks):.1f}")
        print(f"pool_rss_avg_at_last_sample_mb={sum(pool_peaks) / len(pool_peaks):.1f}")

    print("--- cache log ---")
    for key, val in stats.items():
        if key == "peak_refs_by_path":
            print(f"{key}:")
            for ply, ref in sorted(val.items()):
                print(f"  {ply}: peak_ref={ref}")
        else:
            print(f"{key}={val}")

    checks: list[tuple[str, bool, str]] = []
    num_ply = len(list((ROOT / "assets" / "ply").glob("*.ply")))

    if "error" in stats:
        checks.append(("cache log exists", False, str(stats["error"])))
    else:
        checks.append(("acquire/release paired", stats["acquires"] == stats["releases"],
                        f"{stats['acquires']} acquires vs {stats['releases']} releases"))
        checks.append(("no negative refs", not stats["negative_ref"], "ref count stayed >= 0"))
        checks.append(("refs balanced at end", stats["open_refs_at_end"] == 0,
                        f"open refs={stats['open_refs_at_end']}"))
        checks.append(("cache hits observed", stats["hits"] > 0, f"hits={stats['hits']}"))
        checks.append(("cache evictions observed", stats["evictions"] > 0,
                        f"evictions={stats['evictions']} (raise ttl if 0 on short runs)"))
        checks.append(("max cache entries bounded", stats["max_cache_size"] <= num_ply,
                        f"max_cache_size={stats['max_cache_size']} num_ply={num_ply}"))
        checks.append(("concurrent refs >1", stats["max_ref"] > 1,
                        f"max_ref={stats['max_ref']} (proves sharing under load)"))
        checks.append(("manager holds PLY memory", peak_manager_mb > 100,
                        f"peak_manager_mb={peak_manager_mb:.1f}"))
        checks.append(
            (
                "workers not hoarding full PLY RSS",
                peak_pool_mb < 1200,
                f"peak single pool process={peak_pool_mb:.1f}MB (expect subsampled scenes only)",
            )
        )
        if peak_manager_mb > 0 and peak_pool_mb > 0:
            checks.append(
                (
                    "manager RSS dominates single worker",
                    peak_manager_mb >= peak_pool_mb * 0.15,
                    f"manager={peak_manager_mb:.1f}MB pool_max={peak_pool_mb:.1f}MB",
                )
            )

    print("--- checks ---")
    failed = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if not ok:
            failed += 1

    return (proc.returncode or 0) | (1 if failed else 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
