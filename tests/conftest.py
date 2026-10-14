"""Shared fixtures. Nothing here touches the network or a GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from red_queen.config import AppConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig()


TINY_OVERRIDES: dict[str, object] = {
    "seed": 0,
    "device": "cpu",
    "dataset": {"name": "synthetic", "synthetic_size": 300, "max_samples": None},
    "guardian": {
        "tiny": {"dim": 16, "n_layers": 1, "n_heads": 2, "hidden_dim": 32, "vocab_size": 120},
        "max_length": 24,
        "batch_size": 8,
        "eval_batch_size": 64,
        "learning_rate": 3.0e-3,
        "epochs": 6,
        "warmup_ratio": 0.0,
    },
    "robust": {"epochs": 1, "learning_rate": 1.0e-3},
    "evader": {"episodes": 20, "batch_size": 16, "min_batch_size": 4, "log_every": 5},
    "tracking": {"csv": True},
}


@pytest.fixture
def tiny_config(tmp_path: Path) -> AppConfig:
    from red_queen.config import apply_overrides, parse_config

    raw = apply_overrides(TINY_OVERRIDES, [f"paths.runs_dir={tmp_path / 'runs'}"])
    raw = apply_overrides(raw, [f"paths.data_dir={tmp_path / 'data'}"])
    return parse_config(raw)
