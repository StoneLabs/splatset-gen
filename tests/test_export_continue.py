"""Tests for dataset continuation helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from export import get_last_sample_index  # noqa: E402


def _write_annotation(dataset_dir: Path, sample_id: str) -> None:
    record = {"id": sample_id, "image": f"images/{sample_id}.png"}
    with (dataset_dir / "annotations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def test_get_last_sample_index_from_jsonl(tmp_path: Path) -> None:
    ds = tmp_path / "run"
    ds.mkdir()
    _write_annotation(ds, "000001")
    _write_annotation(ds, "000042")

    assert get_last_sample_index(ds) == 42


def test_get_last_sample_index_from_images_fallback(tmp_path: Path) -> None:
    ds = tmp_path / "run"
    images = ds / "images"
    images.mkdir(parents=True)
    (images / "000007.png").write_bytes(b"png")

    assert get_last_sample_index(ds) == 7


def test_get_last_sample_index_empty(tmp_path: Path) -> None:
    ds = tmp_path / "run"
    ds.mkdir()

    assert get_last_sample_index(ds) == 0
