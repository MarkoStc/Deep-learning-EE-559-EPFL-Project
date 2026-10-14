from __future__ import annotations

from pathlib import Path

import pytest

from red_queen.config import (
    AppConfig,
    DatasetName,
    apply_overrides,
    load_config,
    parse_config,
)
from red_queen.errors import ConfigError


def test_default_yaml_matches_model_defaults(default_config_path: Path) -> None:
    assert load_config(default_config_path) == AppConfig()


def test_default_yaml_keeps_original_course_hyperparameters(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    assert cfg.seed == 42
    assert cfg.dataset.name is DatasetName.CIVIL_COMMENTS
    assert cfg.dataset.max_samples == 50_000
    assert cfg.guardian.model_name == "distilbert-base-uncased"
    assert cfg.guardian.learning_rate == pytest.approx(2e-5)
    assert cfg.evader.episodes == 2000
    assert cfg.robust.augmentation_ratio == pytest.approx(0.5)


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="guardain"):
        parse_config({"guardain": {"epochs": 1}})


def test_invalid_value_is_rejected() -> None:
    with pytest.raises(ConfigError, match="learning_rate"):
        parse_config({"guardian": {"learning_rate": -1}})


def test_split_sizes_must_leave_training_data() -> None:
    with pytest.raises(ConfigError, match="test_size"):
        parse_config({"dataset": {"test_size": 0.5, "val_size": 0.5}})


def test_tiny_model_heads_must_divide_dim() -> None:
    with pytest.raises(ConfigError, match="divisible"):
        parse_config({"guardian": {"tiny": {"dim": 30, "n_heads": 4}}})


def test_overrides_parse_yaml_values() -> None:
    raw = apply_overrides(
        {"guardian": {"epochs": 3}},
        ["guardian.epochs=1", "guardian.tiny={dim: 16, n_heads: 2}", "dataset.max_samples=null"],
    )
    cfg = parse_config(raw)
    assert cfg.guardian.epochs == 1
    assert cfg.guardian.tiny is not None
    assert cfg.guardian.tiny.dim == 16
    assert cfg.dataset.max_samples is None


def test_overrides_do_not_mutate_input() -> None:
    raw = {"guardian": {"epochs": 3}}
    apply_overrides(raw, ["guardian.epochs=1"])
    assert raw == {"guardian": {"epochs": 3}}


@pytest.mark.parametrize("bad", ["no_equals_sign", "=1", "seed.inner=1"])
def test_malformed_overrides_raise(bad: str) -> None:
    with pytest.raises(ConfigError):
        apply_overrides({"seed": 1}, [bad])


def test_roundtrip_through_yaml(tmp_path: Path) -> None:
    cfg = parse_config({"seed": 7, "guardian": {"tiny": {}}, "tracking": {"tensorboard": True}})
    out = tmp_path / "nested" / "config.yaml"
    cfg.to_yaml(out)
    assert load_config(out) == cfg


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_robust_schedule_inherits_guardian_batching() -> None:
    cfg = parse_config({"guardian": {"batch_size": 8, "weight_decay": 0.2}})
    schedule = cfg.robust.schedule(cfg.guardian)
    assert schedule.batch_size == 8
    assert schedule.weight_decay == pytest.approx(0.2)
    assert schedule.epochs == cfg.robust.epochs
