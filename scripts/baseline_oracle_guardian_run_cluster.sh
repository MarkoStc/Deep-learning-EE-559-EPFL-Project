#!/usr/bin/env bash
set -euo pipefail

# Runs directly inside the cluster/container environment, without submitting RunAI.
# Usage:
#   bash scripts/baseline_oracle_guardian_run_cluster.sh OUTPUT_DIR NAME=CHECKPOINT [NAME=CHECKPOINT ...]

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

export HOME=/home/mstojano
export TORCHINDUCTOR_CACHE_DIR=/home/mstojano/.torchinductor-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/mstojano/red_queen_moderation
python3 -m src.baseline_oracle_guardian_comparison \
  --config configs/default.yaml \
  ${GUARDIAN_ARGS} \
  --output-dir "${OUTPUT_DIR}"
