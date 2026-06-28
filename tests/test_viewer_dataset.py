"""Tests for viewer datasets root discovery and selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"
sys.path.insert(0, str(VIEWER_DIR))

from app import create_app, discover_datasets, resolve_datasets_root  # noqa: E402


def _write_dataset(path: Path, sample_id: str = "000001") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "annotations.jsonl").write_text(
        json.dumps(
            {
                "id": sample_id,
                "image": "images/000001.png",
                "mask": "masks/000001.png",
                "point": [1, 2],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_resolve_datasets_root_relative(tmp_path, monkeypatch) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    assert resolve_datasets_root("outputs") == root.resolve()


def test_discover_datasets_finds_runs(tmp_path) -> None:
    root = tmp_path / "outputs"
    _write_dataset(root / "run_a")
    _write_dataset(root / "run_b", sample_id="000002")
    (root / "not_a_dataset").mkdir()

    found = discover_datasets(root)
    assert [item["name"] for item in found] == ["run_a", "run_b"]
    assert found[0]["count"] == 1
    assert found[1]["count"] == 1


def test_create_app_requires_datasets(tmp_path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(SystemExit, match="No datasets found"):
        create_app(root)


def test_api_model_reload(tmp_path) -> None:
    root = tmp_path / "outputs"
    _write_dataset(root / "run_a")

    app = create_app(root, initial="run_a", checkpoint=tmp_path / "missing.pth")
    client = app.test_client()

    meta = client.get("/api/meta").get_json()
    assert meta["model"]["loaded"] is False

    response = client.post("/api/model/reload")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["model"]["loaded"] is False
    assert payload["selected"] == "run_a"


def test_api_dataset_select(tmp_path) -> None:
    root = tmp_path / "outputs"
    _write_dataset(root / "run_a")
    _write_dataset(root / "run_b")

    app = create_app(root, initial="run_a")
    client = app.test_client()

    meta = client.get("/api/meta").get_json()
    assert meta["datasets_root"] == str(root.resolve())
    assert [item["name"] for item in meta["datasets"]] == ["run_a", "run_b"]
    assert meta["selected"] == "run_a"
    assert meta["training_config"]["found"] is False
    assert meta["training_config"]["yaml"] == "training / inference config data not found"

    response = client.post("/api/dataset/select", json={"name": "run_b"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["selected"] == "run_b"
    assert payload["count"] == 1

    reloaded = client.post("/api/dataset/reload", json={"name": "run_b"})
    assert reloaded.status_code == 200
    assert reloaded.get_json()["selected"] == "run_b"

    bad = client.post("/api/dataset/select", json={"name": "missing"})
    assert bad.status_code == 400


def test_training_config_sidecar_from_checkpoint(tmp_path, monkeypatch) -> None:
    root = tmp_path / "outputs"
    _write_dataset(root / "run_a")

    train_dir = Path(__file__).resolve().parent.parent / "train"
    sys.path.insert(0, str(train_dir))
    from config import load_training_config  # noqa: E402
    import config as train_config_mod  # noqa: E402
    import train as train_mod  # noqa: E402

    cfg = load_training_config(train_dir / "training_config.yaml")
    monkeypatch.setattr(train_mod, "cfg", cfg)
    monkeypatch.setattr(train_config_mod, "cfg", cfg)

    ckpt_path = tmp_path / "model.pth"
    model = train_mod.PointConditionedUNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_mod.save_checkpoint(
        model,
        optimizer,
        12,
        ckpt_path,
        training_state={"best_val_loss": 0.11, "patience_counter": 0, "divergence_streak": 0, "prev_train_loss": 0.2},
    )

    sidecar = train_mod.checkpoint_config_path(ckpt_path)
    sidecar.write_text("device: sidecar\n", encoding="utf-8")

    app = create_app(root, initial="run_a", checkpoint=ckpt_path)
    client = app.test_client()

    meta = client.get("/api/meta").get_json()
    assert meta["training_config"]["found"] is True
    assert meta["training_config"]["yaml"] == "device: sidecar\n"
    assert meta["training_config"]["path"] == str(sidecar.resolve())
    assert meta["model"]["loaded"] is True
    assert meta["model"]["epoch"] == 12
    assert meta["model"]["metadata"]["training_state"]["best_val_loss"] == 0.11
