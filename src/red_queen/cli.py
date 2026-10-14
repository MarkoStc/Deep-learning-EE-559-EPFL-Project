"""Command-line interface: ``red-queen <command>``.

Every command works on a run directory. ``pipeline`` and ``train`` create one
when ``--run-dir`` is not given; later steps need ``--run-dir`` to find the
artifacts of earlier ones. Inside an existing run the saved ``config.yaml`` is
reused unless ``--config`` is passed, so all steps of a run share one config.

Examples::

    red-queen pipeline --config configs/smoke.yaml
    red-queen train --set guardian.epochs=1
    red-queen evader --run-dir runs/20261014-120000-pipeline
    red-queen predict --checkpoint runs/<run>/guardian/best "some text"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from red_queen import __version__
from red_queen.arms_race import ArmsRace
from red_queen.config import AppConfig, HubConfig, load_config
from red_queen.errors import RedQueenError
from red_queen.guardian import Guardian
from red_queen.logging_setup import configure_logging, fields
from red_queen.pipeline import ALL_STEPS, Pipeline, Step, write_json
from red_queen.runs import CONFIG_FILE, Artifact, RunDirectory
from red_queen.runtime import seed_everything, select_device
from red_queen.stress import BruteForceBaseline
from red_queen.text_safety import sanitize_text
from red_queen.tracking import build_tracker

log = logging.getLogger("red_queen.cli")

DEFAULT_CONFIG = Path("configs/default.yaml")
EXIT_EXPECTED_ERROR = 2
SINGLE_STEP_COMMANDS = {
    "train": Step.GUARDIAN,
    "evader": Step.EVADER,
    "stress": Step.STRESS,
    "robust": Step.ROBUST,
    "evaluate": Step.EVALUATE,
}


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help=f"YAML config (default: {DEFAULT_CONFIG})")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set guardian.epochs=1 (repeatable)",
    )
    parser.add_argument("--run-dir", type=Path, help="existing run directory to continue")
    parser.add_argument("--force", action="store_true", help="redo steps that already finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="red-queen", description="Adversarial stress-testing of a hate-speech classifier."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--log-format", default="text", choices=["text", "json"])
    sub = parser.add_subparsers(dest="command", required=True)

    pipe = sub.add_parser("pipeline", help="run all (or selected) steps")
    _add_run_options(pipe)
    pipe.add_argument(
        "--steps",
        type=lambda s: [Step(x) for x in s.split(",")],
        default=list(ALL_STEPS),
        help=f"comma-separated subset of: {','.join(s.value for s in ALL_STEPS)}",
    )
    for name, step in SINGLE_STEP_COMMANDS.items():
        _add_run_options(sub.add_parser(name, help=f"run only the '{step.value}' step"))

    race = sub.add_parser("arms-race", help="multi-round Evader vs Guardian loop")
    _add_run_options(race)
    race.add_argument("--rounds", type=int, help="override outer_loop.num_rounds")

    base = sub.add_parser("baseline", help="per-action brute-force attack on hateful test texts")
    _add_run_options(base)
    base.add_argument("--model", choices=["guardian", "robust"], default="guardian")

    pred = sub.add_parser("predict", help="classify texts with a saved Guardian")
    pred.add_argument("--checkpoint", type=Path, required=True)
    pred.add_argument("--config", type=Path, help="config for max_length and batch size")
    pred.add_argument("--json", action="store_true", help="print one JSON object per text")
    pred.add_argument("--no-sanitize", action="store_true", help="echo texts without masking")
    pred.add_argument("texts", nargs="+")
    return parser


def resolve_config(args: argparse.Namespace) -> AppConfig:
    run_config = args.run_dir / CONFIG_FILE if getattr(args, "run_dir", None) else None
    if args.config is not None:
        path = args.config
    elif run_config is not None and run_config.exists():
        path = run_config
    else:
        path = DEFAULT_CONFIG
    return load_config(path, getattr(args, "overrides", []))


def open_pipeline(args: argparse.Namespace, label: str) -> Pipeline:
    cfg = resolve_config(args)
    seed_everything(cfg.seed, cfg.deterministic)
    if args.run_dir is not None:
        run = RunDirectory.open(args.run_dir)
    else:
        run = RunDirectory.create(cfg.paths.runs_dir, label)
        run.write_metadata()
    run.save_config(cfg)
    tracker = build_tracker(cfg.tracking, run.root, cfg)
    return Pipeline(cfg, run, select_device(cfg.device), tracker=tracker)


def cmd_steps(args: argparse.Namespace, steps: Sequence[Step], label: str) -> int:
    pipeline = open_pipeline(args, label)
    try:
        pipeline.run(steps, force=args.force)
    finally:
        pipeline.tracker.close()
    print(pipeline.run_dir.root)
    return 0


def cmd_arms_race(args: argparse.Namespace) -> int:
    pipeline = open_pipeline(args, "arms-race")
    try:
        pipeline.run([Step.GUARDIAN])
        summary = ArmsRace(pipeline).run(args.rounds)
    finally:
        pipeline.tracker.close()
    print(summary.to_string(index=False))
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    pipeline = open_pipeline(args, "baseline")
    artifact = Artifact.ROBUST if args.model == "robust" else Artifact.GUARDIAN
    guardian = pipeline.load_guardian(artifact)
    result = BruteForceBaseline(pipeline.stress_generator(guardian, "baseline")).run(
        pipeline.splits.hateful_texts("test")
    )
    path = write_json(pipeline.run_dir.root / "baseline" / f"{args.model}.json", result.to_dict())
    for name, metrics in result.per_action.items():
        print(f"{name:12s} asr={metrics.asr:.4f} conditional_asr={metrics.conditional_asr:.4f}")
    print(f"{'best_of_all':12s} asr={result.best_of_all_asr:.4f}")
    log.info("baseline written", extra=fields(path=str(path)))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else AppConfig()
    guardian = Guardian.from_pretrained(
        args.checkpoint, cfg.guardian, select_device(cfg.device), HubConfig()
    )
    probs = guardian.predict_proba(args.texts)
    for text, prob in zip(args.texts, probs, strict=True):
        shown = text if args.no_sanitize else sanitize_text(text, cfg.evaluation.mask_char)
        label = "hate" if prob > 0.5 else "safe"
        if args.json:
            print(json.dumps({"text": shown, "label": label, "p_hate": round(float(prob), 4)}))
        else:
            print(f"{label}\t{prob:.4f}\t{shown}")
    return 0


def _utf8_stdout() -> None:
    # Windows consoles default to a legacy code page that cannot print the mask character.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_format)
    try:
        if args.command == "pipeline":
            return cmd_steps(args, args.steps, "pipeline")
        if args.command in SINGLE_STEP_COMMANDS:
            return cmd_steps(args, [SINGLE_STEP_COMMANDS[args.command]], args.command)
        if args.command == "arms-race":
            return cmd_arms_race(args)
        if args.command == "baseline":
            return cmd_baseline(args)
        return cmd_predict(args)
    except RedQueenError as exc:
        log.error("%s", exc)
        return EXIT_EXPECTED_ERROR


if __name__ == "__main__":
    sys.exit(main())
