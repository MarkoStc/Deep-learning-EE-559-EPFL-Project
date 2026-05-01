#!/usr/bin/env bash
set -euo pipefail

GUARDIAN_DIR="${1:-outputs/run_029/guardian_best}"
OUTPUT_DIR="${2:-outputs/baseline_results}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
JOB_NAME="baseline-$(date +%Y%m%d-%H%M%S)"

MAX_ARG=""
if [[ -n "$MAX_EXAMPLES" ]]; then
  MAX_ARG="--max-examples $MAX_EXAMPLES"
fi

runai submit "$JOB_NAME" \
  --run-as-uid 309057 \
  --image registry.rcp.epfl.ch/ee559/environment-with-packages:latest \
  --gpu 1 \
  --existing-pvc claimname=home,path=/home/mstojano \
  --command -- bash -lc "export HOME=/home/mstojano; export TORCHINDUCTOR_CACHE_DIR=/home/mstojano/.torchinductor-cache; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd /home/mstojano/red_queen; pip install --break-system-packages -r requirements.txt 2>/dev/null; python3 -m src.baseline_evaluation --config configs/default.yaml --guardian-dir ${GUARDIAN_DIR} --output-dir ${OUTPUT_DIR} ${MAX_ARG}"

echo "Submitted RunAI job: $JOB_NAME"
echo "Logs: runai logs $JOB_NAME -f"
