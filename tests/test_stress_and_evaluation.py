from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest
import torch

from red_queen.config import EvaluationConfig, PerturbationConfig
from red_queen.evader import EvaderPolicy
from red_queen.evaluation import Evaluator, summary_table, write_report
from red_queen.perturbations import PerturbationSet
from red_queen.stress import (
    SAMPLE_FILE,
    STRESS_COLUMNS,
    BruteForceBaseline,
    StressCaseGenerator,
    load_stress_cases,
    save_stress_cases,
)
from tests.fakes import KeywordClassifier

CPU = torch.device("cpu")
HATE = [f"zorblat says hello number {i}" for i in range(30)]


class NoSynonyms:
    def synonyms(self, word: str) -> list[str]:
        return []


@pytest.fixture
def actions() -> PerturbationSet:
    return PerturbationSet.from_config(PerturbationConfig(), synonyms=NoSynonyms())


@pytest.fixture
def generator(actions: PerturbationSet) -> StressCaseGenerator:
    return StressCaseGenerator(KeywordClassifier(["zorblat"]), actions, random.Random(0), 8)


def test_with_action_columns_and_outcomes(generator: StressCaseGenerator) -> None:
    df = generator.with_action(HATE, action=0)
    assert tuple(df.columns) == STRESS_COLUMNS
    assert (df["perturbed_text"] == df["original_text"]).all()
    assert df["attack_success"].sum() == 0
    leet = generator.with_action(HATE, action=generator.perturbations.names.index("leetspeak"))
    assert leet["attack_success"].mean() > 0.5
    assert (leet["pred_original"] == 1).all()


def test_with_policy_uses_greedy_actions(generator: StressCaseGenerator) -> None:
    policy = EvaderPolicy(len(generator.perturbations), hidden_dim=8)
    with torch.no_grad():
        policy.net[-1].bias.copy_(
            torch.tensor([0, 0, 10.0, 0, 0, 0, 0])
        )  # strongly prefer action 2
    df = generator.with_policy(HATE, policy, CPU)
    assert set(df["action_name"]) == {"leetspeak"}
    assert len(df) == len(HATE)


def test_classifier_is_called_in_batches_not_per_text(actions: PerturbationSet) -> None:
    clf = KeywordClassifier(["zorblat"])
    StressCaseGenerator(clf, actions, random.Random(0)).with_action(HATE, 1)
    assert clf.calls == 2  # once before, once after; the legacy loop called it per text


def test_brute_force_union_matches_per_action(generator: StressCaseGenerator) -> None:
    result = BruteForceBaseline(generator).run(HATE)
    assert set(result.per_action) == set(generator.perturbations.names)
    assert result.per_action["noop"].asr == 0.0
    best_single = max(m.asr for m in result.per_action.values())
    assert best_single <= result.best_of_all_asr <= 1.0
    assert json.dumps(result.to_dict())


def test_save_and_load_stress_cases(generator: StressCaseGenerator, tmp_path: Path) -> None:
    df = generator.with_action(["you b1tch zorblat"] * 3, action=0)
    path = save_stress_cases(df, tmp_path, mask_char="#")
    pd.testing.assert_frame_equal(load_stress_cases(path), df)
    sample = pd.read_csv(tmp_path / SAMPLE_FILE)
    assert "b1tch" not in " ".join(sample["original_text"])
    assert "b1tch" in " ".join(load_stress_cases(path)["original_text"])  # raw kept for training
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="not a stress-case file"):
        load_stress_cases(tmp_path / "bad.parquet")


def _test_frame() -> pd.DataFrame:
    rows = [("zorblat a", 1, 1.0), ("zorblat b", 1, 1.0), ("plain c", 0, 1.0), ("plain d", 0, 0.0)]
    rows += [(f"plain {i}", 0, 1.0) for i in range(12)] + [
        (f"zorblat {i}", 1, 1.0) for i in range(12)
    ]
    return pd.DataFrame(rows, columns=["text", "label", "female"])


def test_evaluator_rescores_stress_with_each_model(generator: StressCaseGenerator) -> None:
    stress = generator.with_action(HATE, generator.perturbations.names.index("leetspeak"))
    evaluator = Evaluator(EvaluationConfig(min_subgroup_size=5))
    weak = evaluator.evaluate(KeywordClassifier(["zorblat"]), _test_frame(), stress)
    always_hate = KeywordClassifier([])  # predicts safe for everything
    strong = evaluator.evaluate(always_hate, _test_frame(), stress)
    assert weak.clean.macro_f1 == 1.0
    assert weak.attack is not None and strong.attack is not None
    assert weak.attack.asr == pytest.approx(stress["attack_success"].mean())
    assert strong.attack.asr == 1.0
    assert set(weak.subgroups) == {"female"}


def test_report_files_and_no_macro_f1_on_stress(
    generator: StressCaseGenerator, tmp_path: Path
) -> None:
    stress = generator.with_action(HATE, 2)
    evaluator = Evaluator(EvaluationConfig(min_subgroup_size=5))
    reports = {
        "baseline": evaluator.evaluate(KeywordClassifier(["zorblat"]), _test_frame(), stress)
    }
    table = write_report(reports, tmp_path, extra={"protocol": "test"})
    assert "stress_macro_f1" not in table.columns
    assert {"stress_hate_recall", "stress_asr", "stress_conditional_asr"} <= set(table.columns)
    for name in ("summary_table.csv", "all_metrics.json", "cm_baseline.png", "asr_by_action.png"):
        assert (tmp_path / name).exists(), name
    assert json.loads((tmp_path / "all_metrics.json").read_text())["protocol"] == "test"
    assert summary_table({}).empty
