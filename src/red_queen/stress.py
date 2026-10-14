"""Stress cases: perturbed hateful texts, and a brute-force baseline.

A stress-case frame has one row per hateful text with the columns in
:data:`STRESS_COLUMNS`. Predictions before and after the perturbation are both
kept, so :class:`~red_queen.metrics.AttackMetrics` can report raw and
conditional attack success.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from red_queen.evader import EvaderPolicy
from red_queen.features import batch_features
from red_queen.guardian import TextClassifier
from red_queen.logging_setup import fields
from red_queen.metrics import AttackMetrics
from red_queen.perturbations import PerturbationSet
from red_queen.text_safety import sanitize_text

log = logging.getLogger(__name__)

STRESS_COLUMNS = (
    "original_text",
    "perturbed_text",
    "action",
    "action_name",
    "prob_original",
    "pred_original",
    "prob_perturbed",
    "pred_perturbed",
    "attack_success",
)
STRESS_FILE = "stress_cases.parquet"
SAMPLE_FILE = "stress_cases_sample.csv"
SAMPLE_ROWS = 100


class StressCaseGenerator:
    """Applies perturbations to hateful texts and records the Guardian's response."""

    def __init__(
        self,
        classifier: TextClassifier,
        perturbations: PerturbationSet,
        rng: random.Random,
        policy_batch_size: int = 64,
    ) -> None:
        self.classifier = classifier
        self.perturbations = perturbations
        self.rng = rng
        self.policy_batch_size = policy_batch_size

    def with_policy(
        self, texts: Sequence[str], policy: EvaderPolicy, device: torch.device
    ) -> pd.DataFrame:
        """Perturb each text with the policy's greedy action."""
        probs = self.classifier.predict_proba(texts)
        actions: list[int] = []
        for start in range(0, len(texts), self.policy_batch_size):
            stop = start + self.policy_batch_size
            states = batch_features(texts[start:stop], [float(p) for p in probs[start:stop]])
            actions.extend(policy.greedy(torch.from_numpy(states).to(device)).cpu().tolist())
        return self._build(texts, actions, probs)

    def with_action(self, texts: Sequence[str], action: int) -> pd.DataFrame:
        """Perturb every text with the same action."""
        return self._build(texts, [action] * len(texts), self.classifier.predict_proba(texts))

    def _build(
        self, texts: Sequence[str], actions: Sequence[int], probs_before: np.ndarray
    ) -> pd.DataFrame:
        perturbed = [
            self.perturbations.apply(t, a, self.rng) for t, a in zip(texts, actions, strict=True)
        ]
        probs_after = self.classifier.predict_proba(perturbed)
        pred_after = (probs_after > 0.5).astype(np.int64)
        return pd.DataFrame(
            {
                "original_text": list(texts),
                "perturbed_text": perturbed,
                "action": list(actions),
                "action_name": [self.perturbations.names[a] for a in actions],
                "prob_original": probs_before,
                "pred_original": (probs_before > 0.5).astype(np.int64),
                "prob_perturbed": probs_after,
                "pred_perturbed": pred_after,
                "attack_success": (pred_after == 0).astype(np.int64),
            },
            columns=list(STRESS_COLUMNS),
        )


@dataclass(frozen=True)
class BaselineResult:
    """Per-action attack metrics, plus the share of texts that at least one action fools."""

    per_action: dict[str, AttackMetrics]
    best_of_all_asr: float

    def to_dict(self) -> dict[str, object]:
        return {
            "per_action": {k: v.to_dict() for k, v in self.per_action.items()},
            "best_of_all_asr": self.best_of_all_asr,
        }


class BruteForceBaseline:
    """Tries every action on every text: the bar a learned policy must clear."""

    def __init__(self, generator: StressCaseGenerator) -> None:
        self.generator = generator

    def run(self, texts: Sequence[str]) -> BaselineResult:
        frames = {
            name: self.generator.with_action(texts, action)
            for action, name in enumerate(self.generator.perturbations.names)
        }
        # Union over the same perturbed texts that produced the per-action numbers.
        fooled_by_any = np.stack(
            [f["attack_success"].to_numpy() == 1 for f in frames.values()]
        ).any(axis=0)
        result = BaselineResult(
            per_action={name: AttackMetrics.from_frame(f) for name, f in frames.items()},
            best_of_all_asr=float(fooled_by_any.mean()),
        )
        log.info(
            "brute-force baseline",
            extra=fields(
                best_of_all_asr=result.best_of_all_asr,
                **{f"asr_{k}": v.asr for k, v in result.per_action.items()},
            ),
        )
        return result


def save_stress_cases(df: pd.DataFrame, out_dir: Path, mask_char: str | None) -> Path:
    """Write the full frame as parquet and a short CSV sample for reading.

    The parquet keeps raw text because robust retraining needs it; it belongs in
    a git-ignored run directory. The CSV sample is sanitised unless
    ``mask_char`` is None.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / STRESS_FILE
    df.to_parquet(path, index=False)
    sample = df.head(SAMPLE_ROWS)[
        ["original_text", "perturbed_text", "action_name", "attack_success", "prob_original"]
    ].copy()
    if mask_char is not None:
        for col in ("original_text", "perturbed_text"):
            sample[col] = sample[col].map(lambda t: sanitize_text(t, mask_char))
    sample.to_csv(out_dir / SAMPLE_FILE, index=False)
    return path


def load_stress_cases(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in STRESS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is not a stress-case file (missing {missing})")
    return df
