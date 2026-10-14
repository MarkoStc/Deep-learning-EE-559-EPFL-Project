"""Fine-tuning of the Guardian.

:class:`GuardianTrainer` runs both stages: the baseline fit and the robust
retraining on train data plus stress cases (which only differ in their
:class:`~red_queen.config.TrainSchedule` and training set). After every epoch
it evaluates on the validation set and keeps the checkpoint with the best
macro-F1.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from red_queen.config import AppConfig, TrainSchedule
from red_queen.errors import ArtifactError
from red_queen.guardian import Guardian
from red_queen.logging_setup import fields
from red_queen.metrics import ClassificationMetrics
from red_queen.tracking import NullTracker, Tracker

log = logging.getLogger(__name__)

RESUME_FILE = "trainer_state.pt"


@dataclass(frozen=True)
class LabelledTexts:
    texts: list[str]
    labels: list[int]

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length")

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    val_weighted_f1: float


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_val_macro_f1: float
    history: list[EpochResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_val_macro_f1,
            "history": [asdict(h) for h in self.history],
        }


def augment_with_stress(
    train: LabelledTexts, stress_texts: Sequence[str], ratio: float, rng: random.Random
) -> LabelledTexts:
    """Add a random ``ratio`` share of the stress texts, all labelled hate (1)."""
    if not stress_texts:
        return train
    n = max(1, int(len(stress_texts) * ratio))
    chosen = rng.sample(list(stress_texts), n)
    return LabelledTexts(train.texts + chosen, train.labels + [1] * n)


class GuardianTrainer:
    """AdamW with linear warmup and decay, gradient clipping, best-epoch checkpointing."""

    def __init__(
        self,
        guardian: Guardian,
        schedule: TrainSchedule,
        *,
        seed: int,
        stage: str,
        tracker: Tracker | None = None,
    ) -> None:
        self.guardian = guardian
        self.schedule = schedule
        self.seed = seed
        self.stage = stage
        self.tracker = tracker or NullTracker()

    def fit(
        self,
        train: LabelledTexts,
        val: LabelledTexts,
        best_dir: Path,
        cfg: AppConfig | None = None,
        resume_dir: Path | None = None,
    ) -> FitResult:
        """Train for ``schedule.epochs`` and save the best epoch to ``best_dir``.

        With ``resume_dir``, the full training state is saved there after
        every epoch, and a later call with the same directory continues from
        the next epoch instead of starting over.
        """
        if not len(train) or not len(val):
            raise ValueError("training and validation sets must not be empty")
        sched = self.schedule
        steps_per_epoch = -(-len(train) // sched.batch_size)
        optimizer = torch.optim.AdamW(
            self.guardian.model.parameters(),
            lr=sched.learning_rate,
            weight_decay=sched.weight_decay,
        )
        total = steps_per_epoch * sched.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total * sched.warmup_ratio), total
        )
        best: EpochResult | None = None
        history: list[EpochResult] = []
        start = 1
        if resume_dir is not None and (resume_dir / RESUME_FILE).exists():
            history, best = self._load_state(resume_dir, optimizer, scheduler)
            start = len(history) + 1
            log.info("resuming training", extra=fields(stage=self.stage, next_epoch=start))
        for epoch in range(start, sched.epochs + 1):
            train_loss = self._train_epoch(train, epoch, optimizer, scheduler)
            val_loss, metrics = self.evaluate(val)
            result = EpochResult(epoch, train_loss, val_loss, metrics.macro_f1, metrics.weighted_f1)
            history.append(result)
            self.tracker.log(
                self.stage, epoch, {k: v for k, v in asdict(result).items() if k != "epoch"}
            )
            log.info("epoch done", extra=fields(stage=self.stage, **asdict(result)))
            if best is None or result.val_macro_f1 > best.val_macro_f1:
                best = result
                self.guardian.save(best_dir, cfg)
            if resume_dir is not None:
                self._save_state(resume_dir, optimizer, scheduler, history, best)
        if best is None:
            raise ArtifactError(f"{resume_dir} records no finished epoch")
        return FitResult(best.epoch, best.val_macro_f1, history)

    def _save_state(
        self,
        resume_dir: Path,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        history: list[EpochResult],
        best: EpochResult,
    ) -> None:
        resume_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.guardian.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": [asdict(h) for h in history],
            "best": asdict(best),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        tmp = resume_dir / f"{RESUME_FILE}.tmp"
        torch.save(state, tmp)
        tmp.replace(resume_dir / RESUME_FILE)  # atomic: a crash mid-save keeps the old state

    def _load_state(
        self,
        resume_dir: Path,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> tuple[list[EpochResult], EpochResult]:
        state = torch.load(
            resume_dir / RESUME_FILE, map_location=self.guardian.device, weights_only=False
        )
        history = [EpochResult(**h) for h in state["history"]]
        if len(history) >= self.schedule.epochs + 1:
            raise ArtifactError(f"{resume_dir} has more epochs than the schedule allows")
        self.guardian.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return history, EpochResult(**state["best"])

    def _loader(self, data: LabelledTexts, shuffle: bool, epoch: int = 0) -> DataLoader[int]:
        # A fresh generator per epoch makes the batch order depend only on (seed, epoch).
        generator = torch.Generator().manual_seed(self.seed * 1000 + epoch)
        return DataLoader(
            range(len(data)),  # type: ignore[arg-type]
            batch_size=self.schedule.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def _batch(self, data: LabelledTexts, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        ids = idx.tolist()
        batch = self.guardian.encode([data.texts[i] for i in ids])
        batch["labels"] = torch.tensor([data.labels[i] for i in ids], device=self.guardian.device)
        return batch

    def _train_epoch(
        self,
        data: LabelledTexts,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> float:
        model = self.guardian.model
        model.train()
        losses = []
        for idx in tqdm(self._loader(data, True, epoch), desc=f"{self.stage} {epoch}", leave=False):
            loss = model(**self._batch(data, idx)).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.schedule.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def evaluate(self, data: LabelledTexts) -> tuple[float, ClassificationMetrics]:
        """Mean loss and classification metrics on ``data``."""
        model = self.guardian.model
        model.eval()
        losses, preds = [], []
        for idx in self._loader(data, shuffle=False):
            out = model(**self._batch(data, idx))
            losses.append(out.loss.item())
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
        return float(np.mean(losses)), ClassificationMetrics.from_predictions(data.labels, preds)
