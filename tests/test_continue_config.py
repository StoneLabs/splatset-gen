"""Tests for continue-run config resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from console import resolve_continue_config  # noqa: E402


def _write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def test_resolve_continue_config_uses_cli_when_matching(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    cli_cfg = {"render": {"width": 128, "height": 128}}
    cli_path = tmp_path / "cli.yaml"
    _write_config(cli_path, cli_cfg)
    _write_config(output / "config.yaml", dict(cli_cfg))

    cfg, chosen = resolve_continue_config(
        output_dir=output,
        cli_config_path=cli_path,
        cli_cfg=cli_cfg,
        auto_confirm=False,
    )

    assert cfg == cli_cfg
    assert chosen == cli_path


def test_resolve_continue_config_prompts_for_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    cli_cfg = {"render": {"width": 256}}
    dataset_cfg = {"render": {"width": 128}}
    cli_path = tmp_path / "cli.yaml"
    _write_config(cli_path, cli_cfg)
    _write_config(output / "config.yaml", dataset_cfg)

    monkeypatch.setattr("console.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("console.click.prompt", lambda *a, **k: "dataset")

    cfg, chosen = resolve_continue_config(
        output_dir=output,
        cli_config_path=cli_path,
        cli_cfg=cli_cfg,
        auto_confirm=False,
    )

    assert cfg == dataset_cfg
    assert chosen == output / "config.yaml"


def test_resolve_continue_config_prompts_for_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    cli_cfg = {"render": {"width": 256}}
    dataset_cfg = {"render": {"width": 128}}
    cli_path = tmp_path / "cli.yaml"
    _write_config(cli_path, cli_cfg)
    _write_config(output / "config.yaml", dataset_cfg)

    monkeypatch.setattr("console.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("console.click.prompt", lambda *a, **k: "cli")

    cfg, chosen = resolve_continue_config(
        output_dir=output,
        cli_config_path=cli_path,
        cli_cfg=cli_cfg,
        auto_confirm=False,
    )

    assert cfg == cli_cfg
    assert chosen == cli_path


def test_resolve_continue_config_exits_on_mismatch_with_yes(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    cli_cfg = {"render": {"width": 256}}
    cli_path = tmp_path / "cli.yaml"
    _write_config(cli_path, cli_cfg)
    _write_config(output / "config.yaml", {"render": {"width": 128}})

    with pytest.raises(SystemExit) as exc:
        resolve_continue_config(
            output_dir=output,
            cli_config_path=cli_path,
            cli_cfg=cli_cfg,
            auto_confirm=True,
        )
    assert exc.value.code == 1


def test_resolve_continue_config_without_snapshot_uses_cli(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    cli_cfg = {"render": {"width": 128}}
    cli_path = tmp_path / "cli.yaml"
    _write_config(cli_path, cli_cfg)

    cfg, chosen = resolve_continue_config(
        output_dir=output,
        cli_config_path=cli_path,
        cli_cfg=cli_cfg,
        auto_confirm=False,
    )

    assert cfg == cli_cfg
    assert chosen == cli_path
