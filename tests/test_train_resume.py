"""Tests for train checkpoint resume helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
sys.path.insert(0, str(_TRAIN_DIR))

from config import load_training_config  # noqa: E402
import train as train_mod  # noqa: E402


def test_find_resume_checkpoint_priority(tmp_path, monkeypatch) -> None:
    cfg = load_training_config(_TRAIN_DIR / "training_config.yaml")
    monkeypatch.setattr(train_mod, "cfg", cfg)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    best = ckpt_dir / cfg.BEST_MODEL_NAME
    backup = ckpt_dir / cfg.BACKUP_NAME
    interrupt = ckpt_dir / cfg.INTERRUPT_NAME
    for path in (best, backup, interrupt):
        path.write_bytes(b"x")

    path, kind = train_mod.find_resume_checkpoint(str(ckpt_dir))
    assert path == str(interrupt)
    assert kind == "interrupted"

    interrupt.unlink()
    path, kind = train_mod.find_resume_checkpoint(str(ckpt_dir))
    assert path == str(backup)
    assert kind == "backup"

    backup.unlink()
    path, kind = train_mod.find_resume_checkpoint(str(ckpt_dir))
    assert path == str(best)
    assert kind == "best"


def test_save_and_load_resume_checkpoint(tmp_path, monkeypatch) -> None:
    cfg = load_training_config(_TRAIN_DIR / "training_config.yaml")
    monkeypatch.setattr(train_mod, "cfg", cfg)

    model = train_mod.PointConditionedUNet(base_ch=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    path = tmp_path / "resume.pth"
    training_state = {
        "best_val_loss": 0.42,
        "patience_counter": 3,
        "divergence_streak": 1,
        "prev_train_loss": 0.55,
    }

    train_mod.save_checkpoint(
        model,
        optimizer,
        7,
        path,
        scheduler=scheduler,
        training_state=training_state,
    )

    sidecar = train_mod.checkpoint_config_path(path)
    assert sidecar.is_file()
    assert sidecar.read_text(encoding="utf-8") == Path(cfg.CONFIG_PATH).read_text(encoding="utf-8")

    model2 = train_mod.PointConditionedUNet(base_ch=8)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    epoch, loaded_state, scheduler_state = train_mod.load_resume_checkpoint(
        path, model2, optimizer2, torch.device("cpu"),
    )

    assert epoch == 7
    assert loaded_state == training_state
    assert scheduler_state is not None

    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.allclose(p1, p2)
