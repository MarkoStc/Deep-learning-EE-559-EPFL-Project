#!/bin/bash
# ===================================================================
# Submit Cascade Red Queen as one RunAI job.
# Run this ON THE CLUSTER, from /home/mstojano/red_queen.
#
# Usage:
#   bash scripts/submit_cascade_redqueen.sh
#
# Faster smoke test:
#   MAX_EXAMPLES=50 MAX_DEPTH=2 BEAM_WIDTH=4 SAMPLES_PER_ACTION=1 \
#   OUTPUT_DIR=outputs/cascade_redqueen_smoke \
#   bash scripts/submit_cascade_redqueen.sh
# ===================================================================

set -euo pipefail

UID_NUM=309057
IMAGE="registry.rcp.epfl.ch/ee559/environment-with-packages:latest"
GPU=1
PVC="claimname=home,path=/home/mstojano"
PROJECT_DIR="/home/mstojano/red_queen"

CONFIG="${CONFIG:-configs/default.yaml}"
GUARDIAN_PATH="${GUARDIAN_PATH:-outputs/arms_race/round_05/guardian_robust}"
BASELINE_GUARDIAN_PATH="${BASELINE_GUARDIAN_PATH:-outputs/run_029/guardian_best}"
SPLIT="${SPLIT:-test}"
MAX_EXAMPLES="${MAX_EXAMPLES:-600}"
CLEAN_EVAL_EXAMPLES="${CLEAN_EVAL_EXAMPLES:-7500}"
MAX_DEPTH="${MAX_DEPTH:-4}"
BEAM_WIDTH="${BEAM_WIDTH:-8}"
SAMPLES_PER_ACTION="${SAMPLES_PER_ACTION:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SHIELD_VARIANTS="${SHIELD_VARIANTS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cascade_redqueen_final}"

INNER_CMD="export HOME=/home/mstojano; \
export TORCHINDUCTOR_CACHE_DIR=/home/mstojano/.torchinductor-cache; \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
cd ${PROJECT_DIR}; \
pip install --break-system-packages -r requirements.txt 2>/dev/null; \
CONFIG='${CONFIG}' \
GUARDIAN_PATH='${GUARDIAN_PATH}' \
BASELINE_GUARDIAN_PATH='${BASELINE_GUARDIAN_PATH}' \
SPLIT='${SPLIT}' \
MAX_EXAMPLES='${MAX_EXAMPLES}' \
CLEAN_EVAL_EXAMPLES='${CLEAN_EVAL_EXAMPLES}' \
MAX_DEPTH='${MAX_DEPTH}' \
BEAM_WIDTH='${BEAM_WIDTH}' \
SAMPLES_PER_ACTION='${SAMPLES_PER_ACTION}' \
BATCH_SIZE='${BATCH_SIZE}' \
SHIELD_VARIANTS='${SHIELD_VARIANTS}' \
OUTPUT_DIR='${OUTPUT_DIR}' \
bash scripts/run_cascade_redqueen.sh"

echo "Submitting Cascade Red Queen"
echo "Guardian:   ${GUARDIAN_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

runai submit \
  --run-as-uid ${UID_NUM} \
  --image ${IMAGE} \
  --gpu ${GPU} \
  --existing-pvc ${PVC} \
  --command -- bash -lc "${INNER_CMD}"

echo ""
echo "Job submitted. Monitor with:"
echo "  runai list"
echo "  runai logs <job-name> -f"
echo ""
echo "When done, download with:"
echo "  scp -r rcp:/home/mstojano/red_queen/${OUTPUT_DIR} ~/Desktop/EPFL/Deep\\ Learning/red_queen/outputs/"
