# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `red-queen` CLI: `pipeline`, `train`, `evader`, `stress`, `robust`,
  `evaluate`, `arms-race`, `baseline` and `predict`, with `--set key=value`
  config overrides.
- Typed config (`red_queen.config`, pydantic) that rejects unknown keys and
  invalid values.
- Run directories named `<timestamp>-<label>` with a saved config, metadata
  (git commit, package versions) and an artifact manifest, so steps find the
  outputs of earlier steps.
- Per-epoch resumable Guardian training, and step-level resume of the pipeline.
- Held-out stress set built from hateful test texts; conditional attack success
  rate; brute-force per-action baseline next to the learned policy.
- CSV metrics for every training stage; optional TensorBoard and W&B.
- Synthetic dataset and an offline tiny DistilBERT, used by the tests and by
  `configs/smoke.yaml` (full pipeline in about a minute on CPU).
- Test suite (unit and end-to-end, CPU only, no network), GitHub Actions CI,
  dependency audit, Dependabot, multi-stage Docker image (CPU and CUDA 12.6).
- Structured logging (text or JSON), retries with backoff for Hub downloads.
- `red-queen compare`: score saved checkpoints under the held-out protocol.
- `results/2026-10-14-course-checkpoints-held-out/`: the course checkpoints
  re-scored this way (clean numbers reproduce the course report exactly).
- README rewrite, model card, data card, ADRs, CONTRIBUTING, `CLAUDE.md`.

### Changed

- Packaged as `red_queen` under `src/` with `pyproject.toml` and `uv.lock`;
  torch comes from the `cpu` or `cu126` extra.
- One `GuardianTrainer` replaces the three copied training loops; perturbations
  are classes with an explicit RNG; stress generation is batched.
- The arms race (`outer_loop.py` in the course code) reuses the pipeline
  building blocks.

### Fixed

- **Evaluation leakage:** the robust Guardian was scored on stress cases built
  from training texts, about half of which it had been retrained on. Stress
  metrics now use the held-out set.
- Macro-F1 is no longer reported on single-class stress sets.
- `run_NNN` numbering reused (and overwrote) an existing run after any
  deletion.
- Loading a Guardian checkpoint downloaded and built the base model first.
- The Evader trainer changed `evader.batch_size` in the config in place.
- Stress generation called the Guardian once per text.
- The brute-force "best of all" number re-sampled the perturbations instead of
  using the ones behind the per-action numbers.
- HateXplain loading used a dataset script that current `datasets` refuses to
  run; it now reads the Hub parquet conversion.

### Removed

- Cluster user id and home paths from the config and scripts (now environment
  variables, see `.env.example`).
- `requirements.txt`, the legacy modules and the shell wrappers that hardcoded
  run directories.
- Empty `outputs/run_001` to `run_028` from failed cluster attempts.

## [0.1.0] - 2026-04-11

Original EE-559 course project code, as tagged `v0-original`.
