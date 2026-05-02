#!/usr/bin/env bash
set -euo pipefail

STRATEGY="${1:?Usage: bash scripts/baseline_outer_loop_run_cluster.sh STRATEGY INITIAL_GUARDIAN OUTPUT_DIR [NUM_ROUNDS] [MAX_HATE_TRAIN]}"
INITIAL_GUARDIAN="${2:?Missing initial guardian path}"
OUTPUT_DIR="${3:?Missing output dir}"
NUM_ROUNDS="${4:-5}"
MAX_HATE_TRAIN="${5:-}"
MAX_ARG=""
if [[ -n "${MAX_HATE_TRAIN}" ]]; then
  MAX_ARG="--max_hate_train ${MAX_HATE_TRAIN}"
fi

cd /home/mstojano/red_queen_moderation
python3 -m src.baseline_outer_loop_ablation \
  --config configs/default.yaml \
  --strategy "${STRATEGY}" \
  --initial_guardian "${INITIAL_GUARDIAN}" \
  --num_rounds "${NUM_ROUNDS}" \
  --output_dir "${OUTPUT_DIR}" \
  ${MAX_ARG}
