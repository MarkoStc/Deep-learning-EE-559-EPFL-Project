# Archived course run

Results and checkpoints of the original EE-559 run (EPFL RCP cluster,
2026-03-31 to 2026-04-09), produced by the code at tag `v0-original`.
New runs go to `runs/` and are not committed.

| Path | Content |
|---|---|
| `run_029/` | Baseline Guardian: `guardian_best/` checkpoint (Git LFS), training history and curve, config snapshot |
| `run_030/` | Evader training history and curves (the policy weights were not committed) |
| `run_031/` | Robust Guardian: `guardian_robust/` checkpoint (Git LFS), training history |
| `stress_cases/` | Stress cases from hateful **training** texts, 2,798 rows, all leetspeak |
| `evaluation/` | The course evaluation: summary table, metrics JSON, plots |

Notes:

- `evaluation/` scores the robust Guardian on `stress_cases/`, which overlaps
  its retraining data, so its "Robust Stress" row is not a held-out result
  ([ADR 0003](../docs/adr/0003-held-out-stress-evaluation.md)). Macro-F1 in the
  stress rows is computed on a one-class set and is not meaningful.
- `stress_cases/stress_cases.parquet` contains unmasked comments from Civil
  Comments. `stress_cases_sample.csv` is the masked sample.
- `run_001` to `run_028` were failed or empty cluster attempts (config and
  timestamp only) and were removed; they remain in git history.
- Cluster user id and home paths in the config snapshots are replaced with
  `REDACTED`.

Score the two checkpoints under the held-out protocol with:

```bash
uv run red-queen compare \
  --checkpoint baseline=outputs/run_029/guardian_best \
  --checkpoint robust=outputs/run_031/guardian_robust \
  --action leetspeak
```
