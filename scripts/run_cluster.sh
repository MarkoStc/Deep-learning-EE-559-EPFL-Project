#!/usr/bin/env bash
# Submit one command as a RunAI job on a GPU cluster (written for EPFL RCP).
#
# Run this on the cluster login node, from the project root:
#   cp .env.example .env    # then fill in your own values
#   bash scripts/run_cluster.sh python -m red_queen.train_guardian --config configs/legacy.yaml
#
# Everything user-specific comes from the environment or .env, never from this file.
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${RUNAI_UID:?set RUNAI_UID (your numeric cluster user id) in .env}"
: "${RUNAI_IMAGE:?set RUNAI_IMAGE in .env}"
: "${RUNAI_PVC_CLAIM:?set RUNAI_PVC_CLAIM in .env}"
: "${CLUSTER_HOME:?set CLUSTER_HOME (home directory on the PVC) in .env}"
: "${CLUSTER_PROJECT_DIR:?set CLUSTER_PROJECT_DIR in .env}"
RUNAI_GPU="${RUNAI_GPU:-1}"

if [[ $# -eq 0 ]]; then
  echo "usage: bash scripts/run_cluster.sh <command> [args...]" >&2
  exit 2
fi

# HOME and the inductor cache must point at the PVC: the container user has no passwd entry.
setup="export HOME=${CLUSTER_HOME}; export TORCHINDUCTOR_CACHE_DIR=${CLUSTER_HOME}/.torchinductor-cache"
setup="${setup}; cd ${CLUSTER_PROJECT_DIR}; pip install --break-system-packages -q -e ."

runai submit \
  --run-as-uid "${RUNAI_UID}" \
  --image "${RUNAI_IMAGE}" \
  --gpu "${RUNAI_GPU}" \
  --existing-pvc "claimname=${RUNAI_PVC_CLAIM},path=${CLUSTER_HOME}" \
  --command -- bash -lc "${setup}; $(printf '%q ' "$@")"

echo "Submitted. Follow with: runai list && runai logs <job-name> -f"
