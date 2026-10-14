"""End-to-end pipeline on synthetic data with a tiny model (CPU, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from red_queen.config import AppConfig
from red_queen.pipeline import Pipeline, Step
from red_queen.runs import Artifact, RunDirectory
from red_queen.stress import load_stress_cases

pytestmark = pytest.mark.slow


class NoSynonyms:
    def synonyms(self, word: str) -> list[str]:
        return []


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    from red_queen.config import apply_overrides, parse_config
    from tests.conftest import TINY_OVERRIDES

    root = tmp_path_factory.mktemp("pipeline")
    cfg = parse_config(apply_overrides(TINY_OVERRIDES, [f"paths.runs_dir={root}"]))
    run = RunDirectory.create(cfg.paths.runs_dir, "test")
    pipeline = Pipeline(cfg, run, torch.device("cpu"), synonyms=NoSynonyms())
    pipeline.run()
    return pipeline


def test_every_step_recorded(finished_run: Pipeline) -> None:
    manifest = finished_run.run_dir.manifest
    for artifact in (
        Artifact.GUARDIAN,
        Artifact.EVADER,
        Artifact.STRESS_TRAIN,
        Artifact.STRESS_EVAL,
        Artifact.ROBUST,
        Artifact.EVALUATION,
    ):
        assert manifest.get(artifact).exists(), artifact


def test_stress_sets_are_built_from_their_own_split(finished_run: Pipeline) -> None:
    splits = finished_run.splits
    manifest = finished_run.run_dir.manifest
    held_out = load_stress_cases(manifest.get(Artifact.STRESS_EVAL))
    train_stress = load_stress_cases(manifest.get(Artifact.STRESS_TRAIN))
    assert set(held_out["original_text"]) == set(splits.hateful_texts("test"))
    assert set(train_stress["original_text"]) == set(splits.hateful_texts("train"))
    # One row per hateful test text: the held-out set is exactly the test split.
    assert len(held_out) == len(splits.hateful_texts("test"))


def test_evaluation_report_uses_held_out_set(finished_run: Pipeline) -> None:
    out = finished_run.run_dir.manifest.get(Artifact.EVALUATION)
    payload = json.loads((out / "all_metrics.json").read_text())
    n_test_hate = len(finished_run.splits.hateful_texts("test"))
    assert payload["models"]["baseline"]["attack"]["n"] == n_test_hate
    assert payload["models"]["robust"]["attack"]["n"] == n_test_hate
    assert "best_of_all_actions_asr" in payload["policy_vs_brute_force"]
    assert set(payload["in_sample_train_stress_do_not_report"]) == {"baseline", "robust"}
    assert (out / "summary_table.csv").read_text().startswith("model,clean_macro_f1")


def test_rerun_skips_finished_steps(
    finished_run: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    finished_run.run([Step.STRESS, Step.EVALUATE])
    skipped = [r for r in caplog.records if r.getMessage() == "step already done, skipping"]
    assert len(skipped) == 2


def test_config_snapshot_saved_with_checkpoints(finished_run: Pipeline) -> None:
    best = finished_run.run_dir.manifest.get(Artifact.GUARDIAN)
    assert (best / "red_queen_config.yaml").exists()
    assert isinstance(finished_run.cfg, AppConfig)


def test_metrics_dirs_have_plots(finished_run: Pipeline) -> None:
    root: Path = finished_run.run_dir.root
    for rel in (
        "guardian/training_curves.png",
        "evader/training_curves.png",
        "robust/metrics.json",
    ):
        assert (root / rel).exists(), rel
