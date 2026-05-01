# Baseline-only additions

This repo copy only adds files with `baseline` in their names. No existing file is modified.

Added files:

- `src/baseline_evaluation.py`
- `scripts/baseline_run_cluster.sh`
- `scripts/baseline_submit_runai.sh`
- `scripts/baseline_push_to_cluster_from_laptop.sh`
- `baseline_README.md`

The baseline module implements:

1. Random Uniform
2. Exhaustive Oracle / best-of-all
3. Greedy Single-Action
4. Feature-Based Heuristic using the same 8 Evader features

Typical cluster command:

```bash
cd /home/mstojano/red_queen
bash scripts/baseline_submit_runai.sh outputs/run_029/guardian_best outputs/baseline_results
```

Smoke test:

```bash
cd /home/mstojano/red_queen
MAX_EXAMPLES=200 bash scripts/baseline_submit_runai.sh outputs/run_029/guardian_best outputs/baseline_smoke
```
