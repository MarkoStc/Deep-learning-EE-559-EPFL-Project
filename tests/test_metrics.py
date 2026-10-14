from __future__ import annotations

import pandas as pd
import pytest

from red_queen.metrics import AttackMetrics, ClassificationMetrics


def test_classification_metrics_known_values() -> None:
    m = ClassificationMetrics.from_predictions([1, 1, 0, 0], [1, 0, 0, 0])
    assert m.accuracy == pytest.approx(0.75)
    assert m.hate_precision == pytest.approx(1.0)
    assert m.hate_recall == pytest.approx(0.5)
    assert m.confusion == [[2, 0], [1, 1]]
    assert m.to_dict()["n"] == 4


def test_classification_metrics_validates_input() -> None:
    with pytest.raises(ValueError):
        ClassificationMetrics.from_predictions([1], [1, 0])
    with pytest.raises(ValueError):
        ClassificationMetrics.from_predictions([], [])


def _stress(before: list[int], after: list[int], actions: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"pred_original": before, "pred_perturbed": after, "action_name": actions})


def test_attack_metrics_separate_raw_and_conditional_asr() -> None:
    # Row 0: already missed before the attack. Row 1: flipped by the attack. Rows 2-3: held.
    df = _stress([0, 1, 1, 1], [0, 0, 1, 1], ["typo", "typo", "emoji", "emoji"])
    m = AttackMetrics.from_frame(df)
    assert m.asr == pytest.approx(0.5)
    assert m.n_detected_before == 3
    assert m.conditional_asr == pytest.approx(1 / 3)
    assert m.hate_recall == pytest.approx(0.5)
    assert m.per_action["typo"].asr == pytest.approx(1.0)
    assert m.per_action["emoji"].count == 2
    assert m.to_dict()["hate_recall"] == pytest.approx(0.5)


def test_attack_metrics_when_nothing_detected_before() -> None:
    m = AttackMetrics.from_frame(_stress([0, 0], [0, 0], ["noop", "noop"]))
    assert m.conditional_asr == 0.0


def test_attack_metrics_rejects_empty() -> None:
    with pytest.raises(ValueError):
        AttackMetrics.from_frame(_stress([], [], []))
