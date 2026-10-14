from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from red_queen.config import AppConfig
from red_queen.data import load_splits
from red_queen.errors import ArtifactError
from red_queen.guardian import CONFIG_SNAPSHOT, Guardian
from red_queen.training import GuardianTrainer, LabelledTexts, augment_with_stress

CPU = torch.device("cpu")


def _labelled(frame) -> LabelledTexts:  # type: ignore[no-untyped-def]
    return LabelledTexts(frame["text"].tolist(), frame["label"].tolist())


@pytest.fixture
def tiny_guardian(tiny_config: AppConfig) -> Guardian:
    splits = load_splits(tiny_config)
    return Guardian.create(tiny_config, CPU, splits.train["text"].tolist())


def test_tiny_guardian_predicts_probabilities(tiny_guardian: Guardian) -> None:
    probs = tiny_guardian.predict_proba(["the river looks calm", "zorblat today", ""])
    assert probs.shape == (3,)
    assert np.all((probs >= 0) & (probs <= 1))
    assert set(tiny_guardian.predict(["a", "b"]).tolist()) <= {0, 1}
    assert tiny_guardian.predict_proba([]).shape == (0,)


def test_save_and_reload_gives_same_predictions(
    tiny_guardian: Guardian, tiny_config: AppConfig, tmp_path: Path
) -> None:
    texts = ["our garden felt warm", "vexnor near the station"]
    path = tiny_guardian.save(tmp_path / "ckpt", tiny_config)
    assert (path / CONFIG_SNAPSHOT).exists()
    reloaded = Guardian.from_pretrained(path, tiny_config.guardian, CPU, tiny_config.hub)
    np.testing.assert_allclose(reloaded.predict_proba(texts), tiny_guardian.predict_proba(texts))


def test_loading_missing_checkpoint_fails(tiny_config: AppConfig, tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        Guardian.from_pretrained(tmp_path / "nope", tiny_config.guardian, CPU, tiny_config.hub)


def test_trainer_learns_marker_task_and_keeps_best(
    tiny_guardian: Guardian, tiny_config: AppConfig, tmp_path: Path
) -> None:
    splits = load_splits(tiny_config)
    schedule = tiny_config.guardian.schedule()
    trainer = GuardianTrainer(tiny_guardian, schedule, seed=0, stage="guardian")
    result = trainer.fit(_labelled(splits.train), _labelled(splits.val), tmp_path / "best")
    assert len(result.history) == schedule.epochs
    assert result.best_val_macro_f1 == max(h.val_macro_f1 for h in result.history)
    assert (tmp_path / "best" / "config.json").exists()
    # The synthetic positives carry a marker word, so even a tiny model should learn it.
    assert result.best_val_macro_f1 > 0.8


def test_training_is_repeatable(tiny_config: AppConfig, tmp_path: Path) -> None:
    splits = load_splits(tiny_config)
    losses = []
    for run in range(2):
        guardian = Guardian.create(tiny_config, CPU, splits.train["text"].tolist())
        trainer = GuardianTrainer(guardian, tiny_config.guardian.schedule(), seed=0, stage="g")
        fit = trainer.fit(_labelled(splits.train), _labelled(splits.val), tmp_path / str(run))
        losses.append([h.train_loss for h in fit.history])
    assert losses[0] == pytest.approx(losses[1])


def test_augment_with_stress() -> None:
    base = LabelledTexts(["a", "b"], [0, 0])
    out = augment_with_stress(base, ["s1", "s2", "s3", "s4"], 0.5, random.Random(0))
    assert len(out) == 4
    assert out.labels[2:] == [1, 1]
    assert set(out.texts[2:]) <= {"s1", "s2", "s3", "s4"}
    assert augment_with_stress(base, [], 0.5, random.Random(0)) is base
    assert len(augment_with_stress(base, ["s"], 0.1, random.Random(0))) == 3


def test_labelled_texts_checks_lengths() -> None:
    with pytest.raises(ValueError):
        LabelledTexts(["a"], [])
