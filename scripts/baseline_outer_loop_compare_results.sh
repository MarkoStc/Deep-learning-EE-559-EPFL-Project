#!/usr/bin/env bash
set -euo pipefail

cd /home/mstojano/red_queen_moderation

python3 -m src.baseline_compare_outer_loop_results \
  --input_dirs \
    outputs/baseline_outer_loop_random \
    outputs/baseline_outer_loop_greedy \
    outputs/baseline_outer_loop_heuristic \
    outputs/baseline_outer_loop_oracle \
    outputs/baseline_outer_loop_rl \
  --output_dir outputs/baseline_outer_loop_comparison
