#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/baseline_oracle_guardian_submit_runai.sh OUTPUT_DIR NAME=CHECKPOINT [NAME=CHECKPOINT ...]
#
# Example:
#   bash scripts/baseline_oracle_guardian_submit_runai.sh \
#     outputs/baseline_oracle_guardian_comparison \
#     initial=outputs/run_029/guardian_best \
#     rl_final=outputs/baseline_outer_loop_rl/round_05/guardian_robust

if [ "$#" -lt 2 ]; then
  echo "Usage: bash $0 OUTPUT_DIR NAME=CHECKPOINT [NAME=CHECKPOINT ...]"
  exit 1
fi

OUTPUT_DIR="$1"
shift

GUARDIAN_ARGS=""
for SPEC in "$@"; do
  GUARDIAN_ARGS="${GUARDIAN_ARGS} --guardian ${SPEC}"
done

JOB_NAME="baseline-oracle-$(date +%Y%m%d-%H%M%S)"

echo "[INFO] Submitting ${JOB_NAME}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Guardian args: ${GUARDIAN_ARGS}"

runai submit "${JOB_NAME}" \
  --run-as-uid 309057 \
  --image registry.rcp.epfl.ch/ee559/environment-with-packages:latest \
  --gpu 1 \
  --existing-pvc claimname=home,path=/home/mstojano \
  --command -- bash -lc "export HOME=/home/mstojano; export TORCHINDUCTOR_CACHE_DIR=/home/mstojano/.torchinductor-cache; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd /home/mstojano/red_queen_moderation; pip install --break-system-packages -r requirements.txt 2>/dev/null; python3 -m src.baseline_oracle_guardian_comparison --config configs/default.yaml ${GUARDIAN_ARGS} --output-dir ${OUTPUT_DIR}"

echo "[INFO] Submitted ${JOB_NAME}"
echo "[INFO] Logs: runai logs ${JOB_NAME} -f"
