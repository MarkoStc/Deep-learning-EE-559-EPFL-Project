#!/usr/bin/env bash
set -euo pipefail

STRATEGY="${1:?Usage: bash scripts/baseline_outer_loop_submit_runai.sh STRATEGY INITIAL_GUARDIAN OUTPUT_DIR [NUM_ROUNDS] [MAX_HATE_TRAIN]}"
INITIAL_GUARDIAN="${2:?Missing initial guardian path}"
OUTPUT_DIR="${3:?Missing output dir}"
NUM_ROUNDS="${4:-5}"
MAX_HATE_TRAIN="${5:-}"

JOB_NAME="baseline-outer-${STRATEGY}-$(date +%Y%m%d-%H%M%S)"
MAX_ARG=""
if [[ -n "${MAX_HATE_TRAIN}" ]]; then
  MAX_ARG="--max_hate_train ${MAX_HATE_TRAIN}"
fi

runai submit "${JOB_NAME}" \
  --run-as-uid 309057 \
  --image registry.rcp.epfl.ch/ee559/environment-with-packages:latest \
  --gpu 1 \
  --existing-pvc claimname=home,path=/home/mstojano \
  --command -- bash -lc "export HOME=/home/mstojano; export TORCHINDUCTOR_CACHE_DIR=/home/mstojano/.torchinductor-cache; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd /home/mstojano/red_queen_moderation; pip install --break-system-packages -r requirements.txt 2>/dev/null; python3 -m src.baseline_outer_loop_ablation --config configs/default.yaml --strategy ${STRATEGY} --initial_guardian ${INITIAL_GUARDIAN} --num_rounds ${NUM_ROUNDS} --output_dir ${OUTPUT_DIR} ${MAX_ARG}"

echo "Submitted job: ${JOB_NAME}"
echo "Logs: runai logs ${JOB_NAME} -f"
