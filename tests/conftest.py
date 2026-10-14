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
