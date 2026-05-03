# Baseline Oracle Guardian Comparison

This add-only script compares how vulnerable different Guardian checkpoints are under an Oracle attack.

It is designed to answer:

> How much worse was the non-robust initial Guardian than the robustly trained Guardian?

## Metrics

- `raw_noop_asr`: fraction of hateful test examples the Guardian already misses with no perturbation.
- `raw_oracle_asr`: fraction of hateful test examples where any of the 7 actions fools the Guardian, including noop.
- `added_oracle_over_noop_pp`: `raw_oracle_asr - raw_noop_asr`; rough extra vulnerability from perturbations.
- `effective_oracle_asr_conditioned_on_noop_caught`: among hateful test examples the Guardian originally caught, fraction where at least one non-noop perturbation flips the prediction to safe.
- `clean_macro_f1`: normal performance on the full clean test split.

## Example

```bash
bash scripts/baseline_oracle_guardian_submit_runai.sh \
  outputs/baseline_oracle_guardian_comparison \
  initial=outputs/run_029/guardian_best \
  rl_final=outputs/baseline_outer_loop_rl/round_05/guardian_robust
```

You can also compare all five final ablation Guardians:

```bash
bash scripts/baseline_oracle_guardian_submit_runai.sh \
  outputs/baseline_oracle_guardian_all \
  initial=outputs/run_029/guardian_best \
  random=outputs/baseline_outer_loop_random/round_05/guardian_robust \
  greedy=outputs/baseline_outer_loop_greedy/round_05/guardian_robust \
  heuristic=outputs/baseline_outer_loop_heuristic/round_05/guardian_robust \
  oracle=outputs/baseline_outer_loop_oracle/round_05/guardian_robust \
  rl=outputs/baseline_outer_loop_rl/round_05/guardian_robust
```

Outputs:

- `baseline_oracle_guardian_summary.csv`
- `baseline_oracle_guardian_per_action.csv`
- `baseline_oracle_guardian_metrics.json`
- `baseline_oracle_guardian_asr_comparison.png`
- `baseline_oracle_guardian_per_action_raw_asr.png`
