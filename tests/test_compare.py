from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from red_queen.cli import main
from red_queen.compare import Attack, CheckpointComparison
from red_queen.config import AppConfig
from red_queen.data import load_splits
from red_queen.errors import ConfigError
from red_queen.guardian import Guardian
from red_queen.pipeline import Pipeline
from red_queen.runs import RunDirectory


class NoSynonyms:
    def synonyms(self, word: str) -> list[str]:
        return []


def test_attack_needs_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        Attack()
    with pytest.raises(ValueError):
        Attack(Path("p.pt"), "leetspeak")
    assert Attack(action="typo").describe() == "fixed action typo"


@pytest.fixture
def two_checkpoints(tiny_config: AppConfig, tmp_path: Path) -> dict[str, Path]:
    corpus = load_splits(tiny_config).train["text"].tolist()
    paths = {}
    for i, name in enumerate(("first", "second")):
        torch.manual_seed(i)
        guardian = Guardian.create(
            tiny_config.model_copy(update={"seed": i}), torch.device("cpu"), corpus
        )
        paths[name] = guardian.save(tmp_path / name)
    return paths


def test_comparison_scores_every_checkpoint_on_one_stress_set(
    tiny_config: AppConfig, two_checkpoints: dict[str, Path]
) -> None:
    run = RunDirectory.create(tiny_config.paths.runs_dir, "compare")
    pipeline = Pipeline(tiny_config, run, torch.device("cpu"), synonyms=NoSynonyms())
    out = CheckpointComparison(pipeline).run(two_checkpoints, Attack(action="leetspeak"))
    payload = json.loads((out / "all_metrics.json").read_text())
    n_hate = len(pipeline.splits.hateful_texts("test"))
    assert set(payload["models"]) == {"first", "second"}
    assert all(m["attack"]["n"] == n_hate for m in payload["models"].values())
    assert set(payload["brute_force"]["first"]["per_action"]) == set(pipeline.perturbations.names)
    assert "against 'first'" in payload["protocol"]
    assert (out / "stress_eval" / "stress_cases.parquet").exists()

    with pytest.raises(ConfigError, match="unknown action"):
        CheckpointComparison(pipeline).run(two_checkpoints, Attack(action="rot13"))


def test_compare_cli(
    tiny_config: AppConfig, two_checkpoints: dict[str, Path], tmp_path: Path
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    tiny_config.to_yaml(cfg_path)
    args = ["compare", "--config", str(cfg_path), "--action", "homoglyph"]
    for name, path in two_checkpoints.items():
        args += ["--checkpoint", f"{name}={path}"]
    assert main(args) == 0
    assert main([*args, "--checkpoint", f"first={two_checkpoints['first']}"]) == 2
    with pytest.raises(SystemExit):
        main(["compare", "--config", str(cfg_path), "--action", "x", "--checkpoint", "no-equals"])
