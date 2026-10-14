"""Evaluation of one or more Guardians on clean test data and on a stress set.

The stress set is scored again with each model, so all models are compared on
the same perturbed texts. Subgroup metrics use the Civil Comments identity
columns when present.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from red_queen import plots
from red_queen.config import EvaluationConfig
from red_queen.guardian import TextClassifier
from red_queen.logging_setup import fields
from red_queen.metrics import AttackMetrics, ClassificationMetrics

log = logging.getLogger(__name__)

SUMMARY_FILE = "summary_table.csv"
METRICS_FILE = "all_metrics.json"


@dataclass(frozen=True)
class SubgroupMetrics:
    count: int
    macro_f1: float
    hate_f1: float


@dataclass
class ModelReport:
    clean: ClassificationMetrics
    attack: AttackMetrics | None = None
    subgroups: dict[str, SubgroupMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean.to_dict(),
            "attack": self.attack.to_dict() if self.attack else None,
            "subgroups": {k: vars(v) for k, v in self.subgroups.items()},
        }


class Evaluator:
    def __init__(self, cfg: EvaluationConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self, classifier: TextClassifier, test: pd.DataFrame, stress: pd.DataFrame | None
    ) -> ModelReport:
        preds = classifier.predict(test["text"].tolist())
        report = ModelReport(
            clean=ClassificationMetrics.from_predictions(test["label"].tolist(), preds.tolist()),
            subgroups=self.subgroups(test, preds.tolist()),
        )
        if stress is not None and not stress.empty:
            rescored = stress.assign(
                pred_original=classifier.predict(stress["original_text"].tolist()),
                pred_perturbed=classifier.predict(stress["perturbed_text"].tolist()),
            )
            report.attack = AttackMetrics.from_frame(rescored)
        return report

    def subgroups(self, test: pd.DataFrame, preds: list[int]) -> dict[str, SubgroupMetrics]:
        out: dict[str, SubgroupMetrics] = {}
        labels = test["label"].to_numpy()
        pred_arr = pd.Series(preds).to_numpy()
        for col in self.cfg.identity_columns:
            if col not in test.columns:
                continue
            mask = (test[col].fillna(0) >= self.cfg.identity_threshold).to_numpy()
            if mask.sum() < self.cfg.min_subgroup_size or len(set(labels[mask])) < 2:
                continue
            m = ClassificationMetrics.from_predictions(
                labels[mask].tolist(), pred_arr[mask].tolist()
            )
            out[col] = SubgroupMetrics(
                count=int(mask.sum()), macro_f1=m.macro_f1, hate_f1=m.hate_f1
            )
        return out


def summary_table(reports: Mapping[str, ModelReport]) -> pd.DataFrame:
    """One row per model. Stress columns report recall and ASR only, never macro-F1."""
    rows = []
    for name, r in reports.items():
        row: dict[str, Any] = {
            "model": name,
            "clean_macro_f1": r.clean.macro_f1,
            "clean_hate_f1": r.clean.hate_f1,
            "clean_hate_precision": r.clean.hate_precision,
            "clean_hate_recall": r.clean.hate_recall,
            "clean_accuracy": r.clean.accuracy,
        }
        if r.attack:
            row |= {
                "stress_n": r.attack.n,
                "stress_hate_recall": r.attack.hate_recall,
                "stress_asr": r.attack.asr,
                "stress_conditional_asr": r.attack.conditional_asr,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    reports: Mapping[str, ModelReport], out_dir: Path, extra: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Write the summary CSV, a JSON with every metric, and comparison plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = summary_table(reports)
    table.to_csv(out_dir / SUMMARY_FILE, index=False, float_format="%.4f")
    payload = {"models": {k: v.to_dict() for k, v in reports.items()}, **(extra or {})}
    (out_dir / METRICS_FILE).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    for name, r in reports.items():
        plots.confusion_matrix(r.clean.confusion, f"{name}: clean test", out_dir / f"cm_{name}.png")
    attacked = {k: v.attack for k, v in reports.items() if v.attack}
    if attacked:
        actions = sorted({a for m in attacked.values() for a in m.per_action})
        plots.grouped_bars(
            actions,
            {
                k: [m.per_action[a].asr if a in m.per_action else 0.0 for a in actions]
                for k, m in attacked.items()
            },
            "Attack success rate by perturbation (held-out stress set)",
            "ASR",
            out_dir / "asr_by_action.png",
        )
    groups = sorted({g for r in reports.values() for g in r.subgroups})
    if groups:
        plots.grouped_bars(
            groups,
            {
                k: [r.subgroups[g].hate_f1 if g in r.subgroups else 0.0 for g in groups]
                for k, r in reports.items()
            },
            "Hate-class F1 by identity subgroup",
            "hate F1",
            out_dir / "subgroup_hate_f1.png",
        )
    log.info("wrote evaluation report", extra=fields(path=str(out_dir), models=list(reports)))
    return table
