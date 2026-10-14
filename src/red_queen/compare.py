"""Compare saved Guardian checkpoints under the held-out stress protocol.

Useful for checkpoints trained elsewhere, for example the archived course
models under ``outputs/``. One attack (a saved Evader policy or a fixed action)
is run against the *reference* checkpoint on the hateful test texts; every
checkpoint is then scored on that same perturbed set, on the clean test set,
and with the brute-force baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from red_queen.errors import ConfigError
from red_queen.evader import EvaderPolicy
from red_queen.evaluation import Evaluator, ModelReport, write_report
from red_queen.guardian import Guardian
from red_queen.logging_setup import fields
from red_queen.pipeline import Pipeline
from red_queen.stress import BruteForceBaseline, save_stress_cases

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Attack:
    """Exactly one of a policy file or a fixed action name."""

    policy_path: Path | None = None
    action: str | None = None

    def __post_init__(self) -> None:
        if (self.policy_path is None) == (self.action is None):
            raise ValueError("give exactly one of policy_path or action")

    def describe(self) -> str:
        return f"policy {self.policy_path}" if self.policy_path else f"fixed action {self.action}"


class CheckpointComparison:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def run(self, checkpoints: dict[str, Path], attack: Attack) -> Path:
        """Write a report to ``<run>/compare``; the first checkpoint is the attack target."""
        if not checkpoints:
            raise ValueError("need at least one checkpoint")
        p = self.pipeline
        cfg = p.cfg
        models = {
            name: Guardian.from_pretrained(path, cfg.guardian, p.device, cfg.hub)
            for name, path in checkpoints.items()
        }
        reference = next(iter(checkpoints))
        test_hate = p.splits.hateful_texts("test")
        generator = p.stress_generator(models[reference], "compare-held-out")
        if attack.policy_path is not None:
            policy = EvaderPolicy.load(attack.policy_path, p.perturbations.names, p.device)
            held_out = generator.with_policy(test_hate, policy, p.device)
        else:
            if attack.action not in p.perturbations.names:
                raise ConfigError(
                    f"unknown action {attack.action!r}; use one of {p.perturbations.names}"
                )
            held_out = generator.with_action(test_hate, p.perturbations.names.index(attack.action))

        out = p.run_dir.subdir("compare")
        save_stress_cases(held_out, out / "stress_eval", p.mask_char())
        evaluator = Evaluator(cfg.evaluation)
        reports: dict[str, ModelReport] = {
            name: evaluator.evaluate(model, p.splits.test, held_out)
            for name, model in models.items()
        }
        brute = {
            name: BruteForceBaseline(p.stress_generator(model, f"compare-brute-{name}"))
            .run(test_hate)
            .to_dict()
            for name, model in models.items()
        }
        write_report(
            reports,
            out,
            extra={
                "protocol": (
                    f"Held-out stress set: hateful test texts perturbed by {attack.describe()} "
                    f"against '{reference}'. Every model is scored on the same perturbed texts."
                ),
                "checkpoints": {k: str(v) for k, v in checkpoints.items()},
                "brute_force": brute,
                "seed": cfg.seed,
            },
        )
        log.info("comparison written", extra=fields(path=str(out), models=list(models)))
        return out
