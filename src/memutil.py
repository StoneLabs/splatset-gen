"""Process memory helpers."""

from __future__ import annotations

from pathlib import Path


def self_rss_kb() -> int:
    status = Path("/proc/self/status")
    if not status.is_file():
        return 0
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return 0
    return 0


def pid_rss_kb(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return 0
    return 0
