"""Multi-round arms race between the Evader and the Guardian.

Round k: train Evader_k against Guardian_k, generate stress cases from hateful
training texts, retrain Guardian_k on the union of all stress cases so far to
get Guardian_{k+1}, then measure clean test performance and per-action attack
success on hateful *test* texts with the brute-force baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from red_queen import plots
from red_queen.guardian import Guardian
from red_queen.logging_setup import fields
from red_queen.metrics import AttackMetrics, ClassificationMetrics
from red_queen.pipeline import Pipeline, labelled, write_json
from red_queen.runs import Artifact
from red_queen.runtime import make_rng
from red_queen.stress import BaselineResult, BruteForceBaseline, save_stress_cases
from red_queen.training import augment_with_stress

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoundResult:
    round: int
    clean_before: ClassificationMetrics
    clean_after: ClassificationMetrics
    evader_train_asr: float
    action_distribution: dict[str, float]
    test_baseline_before: BaselineResult
    test_baseline_after: BaselineResult

    def summary_row(self) -> dict[str, Any]:
        dominant = max(self.action_distribution, key=lambda k: self.action_distribution[k])
        return {
            "round": self.round,
            "clean_macro_f1_before": self.clean_before.macro_f1,
            "clean_macro_f1_after": self.clean_after.macro_f1,
            "evader_train_asr": self.evader_train_asr,
            "dominant_action": dominant,
            "test_best_of_all_asr_before": self.test_baseline_before.best_of_all_asr,
            "test_best_of_all_asr_after": self.test_baseline_after.best_of_all_asr,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary_row(),
            "clean_before": self.clean_before.to_dict(),
            "clean_after": self.clean_after.to_dict(),
            "action_distribution": self.action_distribution,
            "test_baseline_before": self.test_baseline_before.to_dict(),
            "test_baseline_after": self.test_baseline_after.to_dict(),
        }


class ArmsRace:
    """Runs ``cfg.outer_loop.num_rounds`` rounds starting from the run's baseline Guardian."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.cfg = pipeline.cfg

    def _clean(self, guardian: Guardian) -> ClassificationMetrics:
        test = self.pipeline.splits.test
        preds = guardian.predict(test["text"].tolist())
        return ClassificationMetrics.from_predictions(test["label"].tolist(), preds.tolist())

    def _baseline(self, guardian: Guardian, stream: str) -> BaselineResult:
        generator = self.pipeline.stress_generator(guardian, stream)
        return BruteForceBaseline(generator).run(self.pipeline.splits.hateful_texts("test"))

    def run(self, num_rounds: int | None = None) -> pd.DataFrame:
        p = self.pipeline
        rounds = num_rounds or self.cfg.outer_loop.num_rounds
        out = p.run_dir.subdir("arms_race")
        guardian = p.load_guardian(Artifact.GUARDIAN)
        hate_train = p.splits.hateful_texts("train")
        cumulative: list[pd.DataFrame] = []
        results: list[RoundResult] = []

        for k in range(1, rounds + 1):
            rdir = p.run_dir.subdir("arms_race", f"round_{k:02d}")
            clean_before = self._clean(guardian)
            baseline_before = self._baseline(guardian, f"race-{k}-test-before")

            p.train_policy(guardian, rdir / "evader", stage=f"race_{k}_evader")
            policy = p.load_policy_from(rdir / "evader")
            stress = p.stress_generator(guardian, f"race-{k}-stress").with_policy(
                hate_train, policy, p.device
            )
            save_stress_cases(stress, rdir / "stress", p.mask_char())
            cumulative.append(stress)

            train = augment_with_stress(
                labelled(p.splits.train),
                pd.concat(cumulative)["perturbed_text"].tolist(),
                self.cfg.robust.augmentation_ratio,
                make_rng(self.cfg.seed, f"race-{k}-augment"),
            )
            p.fit_guardian(
                guardian, train, rdir / "guardian", robust=True, stage=f"race_{k}_robust"
            )
            guardian = Guardian.from_pretrained(
                rdir / "guardian" / "best", self.cfg.guardian, p.device, self.cfg.hub
            )

            result = RoundResult(
                round=k,
                clean_before=clean_before,
                clean_after=self._clean(guardian),
                evader_train_asr=AttackMetrics.from_frame(stress).asr,
                action_distribution={
                    str(a): float(s)
                    for a, s in stress["action_name"].value_counts(normalize=True).items()
                },
                test_baseline_before=baseline_before,
                test_baseline_after=self._baseline(guardian, f"race-{k}-test-after"),
            )
            write_json(rdir / "round_metrics.json", result.to_dict())
            log.info("arms race round done", extra=fields(**result.summary_row()))
            results.append(result)

        summary = pd.DataFrame([r.summary_row() for r in results])
        summary.to_csv(out / "summary_table.csv", index=False, float_format="%.4f")
        write_json(out / "all_rounds.json", [r.to_dict() for r in results])
        self._plot(results, out)
        p.run_dir.manifest.record(Artifact.OUTER_LOOP, out)
        return summary

    def _plot(self, results: list[RoundResult], out: Path) -> None:
        x = [r.round for r in results]
        names = self.pipeline.perturbations.names
        plots.line_panels(
            x,
            {
                "Evader ASR on train hate texts": {"evader": [r.evader_train_asr for r in results]},
                "clean test macro-F1": {
                    "before retrain": [r.clean_before.macro_f1 for r in results],
                    "after retrain": [r.clean_after.macro_f1 for r in results],
                },
                "best-of-all ASR on test hate texts": {
                    "before retrain": [r.test_baseline_before.best_of_all_asr for r in results],
                    "after retrain": [r.test_baseline_after.best_of_all_asr for r in results],
                },
            },
            "round",
            out / "arms_race_dynamics.png",
        )
        plots.grouped_bars(
            list(names),
            {
                "initial Guardian": [
                    results[0].test_baseline_before.per_action[n].asr for n in names
                ],
                "final Guardian": [
                    results[-1].test_baseline_after.per_action[n].asr for n in names
                ],
            },
            "Per-action ASR on hateful test texts",
            "ASR",
            out / "initial_vs_final_vulnerability.png",
        )
