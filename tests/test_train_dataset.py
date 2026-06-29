"""Training dataset loading tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
sys.path.insert(0, str(_TRAIN_DIR))

from dataset import load_all_samples, load_samples_from_dir, stratified_split


def _write_sample(
    dataset_dir: Path,
    sample_id: str,
    point: list[int],
    *,
    mask_value: int = 255,
    size: tuple[int, int] = (64, 64),
) -> None:
    images = dataset_dir / "images"
    masks = dataset_dir / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    width, height = size
    Image.new("RGB", (width, height), (128, 128, 128)).save(images / f"{sample_id}.png")

    mask = Image.new("L", (width, height), 0)
    px = max(0, min(width - 1, point[0]))
    py = max(0, min(height - 1, point[1]))
    mask.putpixel((px, py), mask_value)
    mask.save(masks / f"{sample_id}.png")

    record = {
        "id": sample_id,
        "image": f"images/{sample_id}.png",
        "mask": f"masks/{sample_id}.png",
        "point": point,
    }
    with (dataset_dir / "annotations.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def test_load_samples_from_dir(tmp_path: Path) -> None:
    ds = tmp_path / "run_a"
    ds.mkdir()
    _write_sample(ds, "000001", [10, 20])
    _write_sample(ds, "000002", [30, 40])

    samples = load_samples_from_dir(ds)
    assert len(samples) == 2
    assert samples[0]["run"] == "run_a"
    assert samples[0]["image"].endswith("images/000001.png")
    assert samples[0]["point"] == [10, 20]


def test_load_all_samples_multiple_dirs(tmp_path: Path) -> None:
    ds_a = tmp_path / "run_a"
    ds_b = tmp_path / "run_b"
    ds_a.mkdir()
    ds_b.mkdir()
    _write_sample(ds_a, "000001", [1, 2])
    _write_sample(ds_b, "000001", [3, 4])

    samples = load_all_samples(dataset_dirs=[ds_a, ds_b])
    assert len(samples) == 2
    assert {s["run"] for s in samples} == {"run_a", "run_b"}


def test_load_all_samples_legacy_run_layout(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    ds = root / "run_legacy"
    ds.mkdir(parents=True)
    _write_sample(ds, "000001", [5, 6])

    samples = load_all_samples(data_dir=root)
    assert len(samples) == 1
    assert samples[0]["run"] == "run_legacy"


def test_load_all_samples_missing_dir() -> None:
    with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
        load_all_samples(dataset_dirs=["/no/such/dataset"])


def test_load_samples_skips_click_on_black_mask(tmp_path: Path) -> None:
    ds = tmp_path / "run_a"
    ds.mkdir()
    _write_sample(ds, "000001", [10, 20], mask_value=255)
    _write_sample(ds, "000002", [30, 40], mask_value=0)

    samples = load_samples_from_dir(ds)
    assert len(samples) == 1
    assert samples[0]["point"] == [10, 20]


def test_load_samples_raises_when_all_clicks_on_black(tmp_path: Path) -> None:
    ds = tmp_path / "run_a"
    ds.mkdir()
    _write_sample(ds, "000001", [5, 5], mask_value=0)

    with pytest.raises(ValueError, match="click on black mask"):
        load_samples_from_dir(ds)


def test_stratified_split_keeps_runs() -> None:
    samples = [
        {"run": "a", "image": "1", "mask": "1", "point": [0, 0]},
        {"run": "a", "image": "2", "mask": "2", "point": [1, 1]},
        {"run": "b", "image": "3", "mask": "3", "point": [2, 2]},
        {"run": "b", "image": "4", "mask": "4", "point": [3, 3]},
    ]
    train, val, test = stratified_split(samples, 0.5, 0.25, seed=0)
    assert train or val or test
    assert len(train) + len(val) + len(test) == 4
