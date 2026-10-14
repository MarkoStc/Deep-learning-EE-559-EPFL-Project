# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Everything runs through uv; no virtualenv activation needed. torch comes from an extra, so always pass one.

```bash
uv sync --extra cpu --extra tensorboard   # dev install (use --extra cu126 for CUDA 12.6 instead of cpu)
make check                                # ruff check + format check, strict mypy, pytest with coverage
make test-fast                            # pytest -m "not slow" (skips end-to-end pipeline tests)
uv run pytest tests/test_config.py::test_unknown_key_is_rejected   # single test
make smoke                                # CLI pipeline on synthetic data + tiny model, ~1 min CPU
uv run red-queen --help                   # all commands
```

On Windows, pytest prints "Windows fatal exception: access violation" stack dumps from torch CPU kernels while tests still pass; add `-p no:faulthandler` to silence them.

## Architecture

The experiment is a pipeline of five steps over one **run directory** (`runs/<YYYYmmdd-HHMMSS>-<label>/`): `guardian` (fine-tune DistilBERT classifier) → `evader` (REINFORCE contextual bandit choosing one of 7 perturbations per hateful text) → `stress` (apply the greedy policy) → `robust` (retrain Guardian on train + stress cases) → `evaluate`. `pipeline.Pipeline` owns the steps; `arms_race.ArmsRace` and `compare.CheckpointComparison` reuse its building blocks (`fit_guardian`, `train_policy`, `stress_generator`) instead of duplicating them.

- **Manifest, not paths.** Each step records its output in `<run>/manifest.json` (`runs.Artifact` names). Later steps read inputs from the manifest; a step whose artifacts exist is skipped unless `--force`. Guardian training also resumes per epoch from `<stage>/last/trainer_state.pt`.
- **Config.** One YAML (`configs/default.yaml`) parsed into frozen pydantic models in `config.py` with `extra="forbid"`. New hyperparameters go into the model and the YAML, never module constants. CLI `--set a.b=value` overrides are YAML-parsed. Inside an existing run, the run's saved `config.yaml` is reused unless `--config` is given.
- **Evaluation protocol (ADR 0003).** The stress step writes `stress/train` (from hateful *train* texts, used for robust retraining) and `stress/eval` (hateful *test* texts, separate RNG stream). All stress metrics use `stress/eval`. Report recall, raw ASR and conditional ASR (`metrics.AttackMetrics`); never macro-F1 on a stress set (single class). The leaky course-style number is kept only under `in_sample_train_stress_do_not_report`.
- **Randomness.** No global RNG in library code: components get `runtime.make_rng(seed, "<stream>")` or a `torch.Generator`. Perturbations take the RNG as an argument. Trainer batch order depends only on `(seed, epoch)`, which is what makes resume exact.
- **Action space order** (`perturbations.DEFAULT_ORDER`) is stored in saved Evader policies; `EvaderPolicy.load` refuses a mismatch. Do not reorder it.
- **Seams for tests.** `guardian.TextClassifier` is a protocol; `tests/fakes.KeywordClassifier` stands in for the model. `Guardian.tiny` + `dataset.name: synthetic` give an offline real model. Hub loaders (`data.HubLoader`) and WordNet (`perturbations.SynonymLookup`) are injected. Tests must not touch the network or a GPU; shared tiny config is `TINY_OVERRIDES` / `tiny_config` in `tests/conftest.py`.
- **Errors.** Expected failures raise `errors.RedQueenError` subclasses; the CLI maps them to exit code 2. Logging uses `logging_setup.fields(...)` structured extras; only `cli.py` prints.

## Repo-specific rules

- Never log or commit raw hateful text. Use `text_safety.sanitize_text`; stress parquet files with raw text belong in git-ignored `runs/`.
- `outputs/` is the archived course run (checkpoints in Git LFS). Do not overwrite it; its stress numbers are leaky (see `outputs/README.md`).
- Numbers in README or docs must come from a command in this repo, stated next to the number. Smoke-config numbers are never results.
- Cluster-specific values (RunAI uid, PVC paths) come from `.env`, read by `scripts/run_cluster.sh`.
- The course project had three authors; keep that credit in the README. There is no open-source licence yet.
- Conventional Commits, one branch per change, CHANGELOG under *Unreleased* (see CONTRIBUTING.md).
