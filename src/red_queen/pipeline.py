"""The five-step Red Queen pipeline over one run directory.

Steps: train the Guardian, train the Evader against it, generate stress cases,
retrain the Guardian on them, evaluate. Each step records its output in the
run's manifest; re-running the pipeline on the same run skips steps whose
artifact already exists, so an interrupted run continues where it stopped.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from red_queen import plots
from red_queen.config import AppConfig
from red_queen.data import DatasetSource, DataSplits, load_splits
from red_queen.evader import (
    POLICY_FILE,
    BanditEnv,
    EvaderPolicy,
    EvaderResult,
    EvaderTrainer,
    effective_batch_size,
)
from red_queen.evaluation import Evaluator, ModelReport, write_report
from red_queen.guardian import Guardian
from red_queen.logging_setup import fields
from red_queen.metrics import AttackMetrics
from red_queen.perturbations import PerturbationSet, SynonymLookup
from red_queen.runs import Artifact, RunDirectory
from red_queen.runtime import make_rng
from red_queen.stress import (
    BruteForceBaseline,
    StressCaseGenerator,
    load_stress_cases,
    save_stress_cases,
)
from red_queen.tracking import NullTracker, Tracker
from red_queen.training import FitResult, GuardianTrainer, LabelledTexts, augment_with_stress

log = logging.getLogger(__name__)


class Step(StrEnum):
    GUARDIAN = "guardian"
    EVADER = "evader"
    STRESS = "stress"
    ROBUST = "robust"
    EVALUATE = "evaluate"


ALL_STEPS = tuple(Step)
_STEP_ARTIFACTS = {
    Step.GUARDIAN: (Artifact.GUARDIAN,),
    Step.EVADER: (Artifact.EVADER,),
    Step.STRESS: (Artifact.STRESS_TRAIN, Artifact.STRESS_EVAL),
    Step.ROBUST: (Artifact.ROBUST,),
    Step.EVALUATE: (Artifact.EVALUATION,),
}


def labelled(frame: pd.DataFrame) -> LabelledTexts:
    return LabelledTexts([str(t) for t in frame["text"]], [int(v) for v in frame["label"]])


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


class Pipeline:
    """Runs pipeline steps inside one :class:`RunDirectory`.

    ``source`` and ``synonyms`` exist for dependency injection in tests; by
    default the dataset comes from the config and synonyms from WordNet.
    """

    def __init__(
        self,
        cfg: AppConfig,
        run: RunDirectory,
        device: torch.device,
        *,
        source: DatasetSource | None = None,
        synonyms: SynonymLookup | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run
        self.device = device
        self.source = source
        self.tracker = tracker or NullTracker()
        self.perturbations = PerturbationSet.from_config(cfg.perturbation, synonyms)

    @cached_property
    def splits(self) -> DataSplits:
        return load_splits(self.cfg, self.source)

    # ── orchestration ───────────────────────────────────────────────

    def run(self, steps: Sequence[Step] = ALL_STEPS, force: bool = False) -> None:
        """Run ``steps`` in pipeline order, skipping finished ones unless ``force``."""
        actions = {
            Step.GUARDIAN: self.train_guardian,
            Step.EVADER: self.train_evader,
            Step.STRESS: self.generate_stress,
            Step.ROBUST: self.robust_retrain,
            Step.EVALUATE: self.evaluate,
        }
        for step in ALL_STEPS:
            if step not in steps:
                continue
            if not force and all(self.run_dir.manifest.has(a) for a in _STEP_ARTIFACTS[step]):
                log.info("step already done, skipping", extra=fields(step=step.value))
                continue
            log.info("running step", extra=fields(step=step.value))
            actions[step]()

    # ── reusable building blocks (also used by the arms race) ───────

    def load_guardian(self, artifact: Artifact) -> Guardian:
        path = self.run_dir.manifest.get(artifact)
        return Guardian.from_pretrained(path, self.cfg.guardian, self.device, self.cfg.hub)

    def load_policy(self) -> EvaderPolicy:
        return self.load_policy_from(self.run_dir.manifest.get(Artifact.EVADER))

    def load_policy_from(self, evader_dir: Path) -> EvaderPolicy:
        return EvaderPolicy.load(evader_dir / POLICY_FILE, self.perturbations.names, self.device)

    def fit_guardian(
        self, guardian: Guardian, train: LabelledTexts, out_dir: Path, *, robust: bool, stage: str
    ) -> FitResult:
        schedule = (
            self.cfg.robust.schedule(self.cfg.guardian) if robust else self.cfg.guardian.schedule()
        )
        trainer = GuardianTrainer(
            guardian, schedule, seed=self.cfg.seed, stage=stage, tracker=self.tracker
        )
        resume_dir = out_dir / "last"
        result = trainer.fit(
            train, labelled(self.splits.val), out_dir / "best", self.cfg, resume_dir
        )
        if not self.cfg.guardian.keep_resume_state:
            shutil.rmtree(resume_dir, ignore_errors=True)
        write_json(out_dir / "metrics.json", result.to_dict())
        epochs = [h.epoch for h in result.history]
        plots.line_panels(
            epochs,
            {
                "loss": {
                    "train": [h.train_loss for h in result.history],
                    "val": [h.val_loss for h in result.history],
                },
                "validation F1": {
                    "macro": [h.val_macro_f1 for h in result.history],
                    "weighted": [h.val_weighted_f1 for h in result.history],
                },
            },
            "epoch",
            out_dir / "training_curves.png",
        )
        return result

    def train_policy(self, guardian: Guardian, out_dir: Path, stage: str) -> EvaderResult:
        hate = self.splits.hateful_texts("train")
        env = BanditEnv(
            hate,
            guardian,
            self.perturbations,
            make_rng(self.cfg.seed, f"{stage}-env"),
            effective_batch_size(len(hate), self.cfg.evader),
        )
        policy = EvaderPolicy(len(self.perturbations), self.cfg.evader.hidden_dim)
        trainer = EvaderTrainer(
            env,
            policy,
            self.cfg.evader,
            self.device,
            seed=self.cfg.seed,
            tracker=self.tracker,
            stage=stage,
        )
        result = trainer.train()
        policy.save(out_dir / POLICY_FILE, self.perturbations.names)
        write_json(out_dir / "metrics.json", result.to_dict())
        history = result.history
        names = self.perturbations.names
        plots.line_panels(
            [h["episode"] for h in history],
            {
                "attack success rate": {"batch": [h["attack_success_rate"] for h in history]},
                "policy entropy": {"entropy": [h["entropy"] for h in history]},
                "action share": {n: [h[f"action_{n}"] for h in history] for n in names},
            },
            "episode",
            out_dir / "training_curves.png",
        )
        return result

    def stress_generator(self, guardian: Guardian, stream: str) -> StressCaseGenerator:
        return StressCaseGenerator(
            guardian,
            self.perturbations,
            make_rng(self.cfg.seed, stream),
            self.cfg.stress.batch_size,
        )

    def mask_char(self) -> str | None:
        """Mask for saved text samples, or None when sanitising is off."""
        ev = self.cfg.evaluation
        return ev.mask_char if ev.sanitize_logs else None

    # ── steps ───────────────────────────────────────────────────────

    def train_guardian(self) -> Path:
        out = self.run_dir.subdir("guardian")
        guardian = Guardian.create(self.cfg, self.device, self.splits.train["text"].tolist())
        self.fit_guardian(
            guardian, labelled(self.splits.train), out, robust=False, stage="guardian"
        )
        self.run_dir.manifest.record(Artifact.GUARDIAN, out / "best")
        return out / "best"

    def train_evader(self) -> Path:
        out = self.run_dir.subdir("evader")
        self.train_policy(self.load_guardian(Artifact.GUARDIAN), out, stage="evader")
        self.run_dir.manifest.record(Artifact.EVADER, out)
        return out

    def generate_stress(self) -> Path:
        """Two stress sets from the baseline Guardian and the trained Evader.

        * ``stress/train``: hateful *training* texts. Robust retraining learns
          from these.
        * ``stress/eval``: hateful *test* texts, with a separate RNG stream.
          Neither model ever trains on them, so this is the set evaluation
          reports. The course project scored the robust model on the training
          stress set, half of which it had been retrained on.
        """
        guardian = self.load_guardian(Artifact.GUARDIAN)
        policy = self.load_policy()
        paths = {}
        for artifact, split, stream in (
            (Artifact.STRESS_TRAIN, "train", "stress-train"),
            (Artifact.STRESS_EVAL, "test", "stress-eval"),
        ):
            df = self.stress_generator(guardian, stream).with_policy(
                self.splits.hateful_texts(split), policy, self.device
            )
            subdir = "train" if split == "train" else "eval"
            paths[artifact] = save_stress_cases(
                df, self.run_dir.subdir("stress", subdir), self.mask_char()
            )
            self.run_dir.manifest.record(artifact, paths[artifact])
            log.info(
                "stress cases generated",
                extra=fields(split=split, n=len(df), asr=float(df["attack_success"].mean())),
            )
        return paths[Artifact.STRESS_EVAL]

    def robust_retrain(self) -> Path:
        out = self.run_dir.subdir("robust")
        guardian = self.load_guardian(Artifact.GUARDIAN)
        stress = load_stress_cases(self.run_dir.manifest.get(Artifact.STRESS_TRAIN))
        train = augment_with_stress(
            labelled(self.splits.train),
            stress["perturbed_text"].tolist(),
            self.cfg.robust.augmentation_ratio,
            make_rng(self.cfg.seed, "augment"),
        )
        self.fit_guardian(guardian, train, out, robust=True, stage="robust")
        self.run_dir.manifest.record(Artifact.ROBUST, out / "best")
        return out / "best"

    def evaluate(self) -> Path:
        """Score every Guardian on the clean test set and the held-out stress set.

        Also records, for transparency, the in-sample number the course
        project reported (robust model on the training stress set) and a
        brute-force baseline on the same test texts, so the learned policy can
        be compared with simply trying every action.
        """
        out = self.run_dir.subdir("evaluation")
        held_out = load_stress_cases(self.run_dir.manifest.get(Artifact.STRESS_EVAL))
        train_stress = load_stress_cases(self.run_dir.manifest.get(Artifact.STRESS_TRAIN))
        evaluator = Evaluator(self.cfg.evaluation)
        models = {"baseline": self.load_guardian(Artifact.GUARDIAN)}
        if self.run_dir.manifest.has(Artifact.ROBUST):
            models["robust"] = self.load_guardian(Artifact.ROBUST)
        reports: dict[str, ModelReport] = {
            name: evaluator.evaluate(model, self.splits.test, held_out)
            for name, model in models.items()
        }
        in_sample = {
            name: AttackMetrics.from_frame(
                train_stress.assign(
                    pred_original=model.predict(train_stress["original_text"].tolist()),
                    pred_perturbed=model.predict(train_stress["perturbed_text"].tolist()),
                )
            ).to_dict()
            for name, model in models.items()
        }
        brute = BruteForceBaseline(self.stress_generator(models["baseline"], "eval-brute-force"))
        brute_result = brute.run(self.splits.hateful_texts("test"))
        policy_asr = AttackMetrics.from_frame(held_out).asr
        write_report(
            reports,
            out,
            extra={
                "protocol": (
                    "Stress metrics use stress/eval: hateful test texts perturbed by the Evader "
                    "trained against the baseline. No model trained on these texts."
                ),
                "policy_vs_brute_force": {
                    "evader_policy_asr": policy_asr,
                    "best_single_action_asr": max(m.asr for m in brute_result.per_action.values()),
                    "best_of_all_actions_asr": brute_result.best_of_all_asr,
                    "per_action": brute_result.to_dict()["per_action"],
                },
                "in_sample_train_stress_do_not_report": in_sample,
                "seed": self.cfg.seed,
            },
        )
        self.run_dir.manifest.record(Artifact.EVALUATION, out)
        return out
