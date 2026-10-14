from __future__ import annotations

from pathlib import Path

import pytest
import torch

from red_queen.config import AppConfig
from red_queen.data import DataSplits, load_splits
from red_queen.guardian import Guardian
from red_queen.training import (
    REPLACE_ATTEMPTS,
    RESUME_FILE,
    FitResult,
    GuardianTrainer,
    LabelledTexts,
    replace_with_retry,
)

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


def test_replace_retries_while_file_is_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, dst = tmp_path / "a.tmp", tmp_path / "a"
    src.write_text("new")
    dst.write_text("old")
    real_replace = Path.replace
    failures = iter([True, True, False])

    def flaky(self: Path, target: Path) -> Path:
        if next(failures):
            raise PermissionError("locked by another process")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    pauses: list[float] = []
    replace_with_retry(src, dst, sleep=pauses.append)
    assert dst.read_text() == "new"
    assert len(pauses) == 2


def test_replace_gives_up_after_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def locked(self: Path, target: Path) -> Path:
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "replace", locked)
    pauses: list[float] = []
    with pytest.raises(PermissionError):
        replace_with_retry(tmp_path / "a", tmp_path / "b", sleep=pauses.append)
    assert len(pauses) == REPLACE_ATTEMPTS - 1
