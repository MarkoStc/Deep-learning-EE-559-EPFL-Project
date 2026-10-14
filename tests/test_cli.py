from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from red_queen.cli import DEFAULT_CONFIG, build_parser, main, resolve_config
from red_queen.config import AppConfig
from red_queen.data import load_splits
from red_queen.guardian import Guardian
from red_queen.pipeline import ALL_STEPS, Step


def test_parser_steps_and_overrides() -> None:
    args = build_parser().parse_args(
        ["pipeline", "--steps", "guardian,evader", "--set", "seed=3", "--set", "guardian.epochs=1"]
    )
    assert args.steps == [Step.GUARDIAN, Step.EVADER]
    assert args.overrides == ["seed=3", "guardian.epochs=1"]
    assert build_parser().parse_args(["pipeline"]).steps == list(ALL_STEPS)


def test_unknown_step_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["pipeline", "--steps", "guardian,nonsense"])


def test_resolve_config_prefers_explicit_then_run_then_default(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    AppConfig(seed=11).to_yaml(run / "config.yaml")
    args = build_parser().parse_args(["evader", "--run-dir", str(run)])
    assert resolve_config(args).seed == 11
    args = build_parser().parse_args(
        ["evader", "--run-dir", str(run), "--config", str(DEFAULT_CONFIG)]
    )
    assert resolve_config(args).seed == 42
    args = build_parser().parse_args(["train", "--set", "seed=5"])
    assert resolve_config(args).seed == 5


def test_missing_run_dir_exits_with_expected_error(tmp_path: Path) -> None:
    assert main(["evader", "--run-dir", str(tmp_path / "missing")]) == 2


def test_bad_override_exits_with_expected_error() -> None:
    assert main(["train", "--set", "guardian.epochs=-1"]) == 2


def test_predict_prints_json_and_masks(
    tiny_config: AppConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    splits = load_splits(tiny_config)
    path = Guardian.create(tiny_config, torch.device("cpu"), splits.train["text"].tolist()).save(
        tmp_path / "ckpt", tiny_config
    )
    config_path = tmp_path / "cfg.yaml"
    tiny_config.to_yaml(config_path)
    code = main(
        ["predict", "--checkpoint", str(path), "--config", str(config_path), "--json", "you b1tch"]
    )
    assert code == 0
    row = json.loads(capsys.readouterr().out.strip())
    assert row["label"] in {"safe", "hate"}
    assert 0.0 <= row["p_hate"] <= 1.0
    assert "b1tch" not in row["text"]
