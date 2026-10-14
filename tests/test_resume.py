from __future__ import annotations

from pathlib import Path

import pytest
import torch

from red_queen.config import AppConfig
from red_queen.data import DataSplits, load_splits
from red_queen.guardian import Guardian
from red_queen.training import RESUME_FILE, FitResult, GuardianTrainer, LabelledTexts

CPU = torch.device("cpu")


class InterruptedRunError(RuntimeError):
    pass


def _data(splits: DataSplits, name: str) -> LabelledTexts:
    frame = splits.split(name)
    return LabelledTexts(frame["text"].tolist(), frame["label"].tolist())


def _fit(cfg: AppConfig, splits: DataSplits, out: Path, stop_at: int | None = None) -> FitResult:
    torch.manual_seed(123)  # same global dropout stream for both runs
    guardian = Guardian.create(cfg, CPU, splits.train["text"].tolist())
    trainer = GuardianTrainer(guardian, cfg.guardian.schedule(), seed=cfg.seed, stage="g")
    if stop_at is not None:
        original = trainer._train_epoch

        def crash_at(data, epoch, optimizer, scheduler):  # type: ignore[no-untyped-def]
            if epoch == stop_at:
                raise InterruptedRunError
            return original(data, epoch, optimizer, scheduler)

        trainer._train_epoch = crash_at  # type: ignore[method-assign]
    return trainer.fit(
        _data(splits, "train"), _data(splits, "val"), out / "best", None, out / "last"
    )


def test_resumed_training_matches_uninterrupted(tiny_config: AppConfig, tmp_path: Path) -> None:
    splits = load_splits(tiny_config)
    reference = _fit(tiny_config, splits, tmp_path / "reference")

    with pytest.raises(InterruptedRunError):
        _fit(tiny_config, splits, tmp_path / "resumed", stop_at=3)
    assert (tmp_path / "resumed" / "last" / RESUME_FILE).exists()
    resumed = _fit(tiny_config, splits, tmp_path / "resumed")

    assert [h.epoch for h in resumed.history] == list(range(1, tiny_config.guardian.epochs + 1))
    assert [h.train_loss for h in resumed.history] == pytest.approx(
        [h.train_loss for h in reference.history], rel=1e-4
    )
    assert resumed.best_epoch == reference.best_epoch


def test_finished_state_is_not_retrained(tiny_config: AppConfig, tmp_path: Path) -> None:
    splits = load_splits(tiny_config)
    first = _fit(tiny_config, splits, tmp_path)
    again = _fit(tiny_config, splits, tmp_path)
    assert again.history == first.history
