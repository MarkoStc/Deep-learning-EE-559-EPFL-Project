# Baseline Outer-Loop Ablation

This add-on compares adversarial-training strategies over the same 5-round Guardian/Evader arms race.

The original project trains the Guardian against an RL Evader. This ablation asks whether RL is actually needed, or whether simpler stress-case generators make the final Guardian equally robust.

## Strategies

- `random`: choose one of the 7 perturbation actions uniformly for each hateful training text.
- `greedy`: find the best single action on validation hateful texts each round, then use it for every hateful training text.
- `heuristic`: use hand-written rules based on the same 8 features the RL Evader sees.
- `oracle`: try all 7 actions per hateful training text and keep a successful action if one exists; otherwise keep the action with the lowest Guardian hate probability.
- `rl`: reproduce the original learned Evader strategy, trained from scratch each round.

## Key metric

The main final metric is `final_oracle_asr`, measured after round 5. Lower is better.

This means: for each hateful test text, try all 7 perturbation actions. If any action fools the final Guardian, count it as a failure.

Also track `final_clean_macro_f1` to make sure robustness did not destroy normal classification performance.

## Run examples

```bash
cd /home/mstojano/red_queen_moderation

bash scripts/baseline_outer_loop_submit_runai.sh random outputs/run_029/guardian_best outputs/baseline_outer_loop_random 5
bash scripts/baseline_outer_loop_submit_runai.sh greedy outputs/run_029/guardian_best outputs/baseline_outer_loop_greedy 5
bash scripts/baseline_outer_loop_submit_runai.sh heuristic outputs/run_029/guardian_best outputs/baseline_outer_loop_heuristic 5
bash scripts/baseline_outer_loop_submit_runai.sh oracle outputs/run_029/guardian_best outputs/baseline_outer_loop_oracle 5
bash scripts/baseline_outer_loop_submit_runai.sh rl outputs/run_029/guardian_best outputs/baseline_outer_loop_rl 5
```

Smoke test with fewer hateful training examples:

```bash
bash scripts/baseline_outer_loop_submit_runai.sh random outputs/run_029/guardian_best outputs/baseline_outer_loop_random_smoke 1 200
```

Compare after all runs finish:

```bash
bash scripts/baseline_outer_loop_compare_results.sh
cat outputs/baseline_outer_loop_comparison/baseline_outer_loop_final_comparison.csv
```
