#!/bin/bash
# ===================================================================
# RunAI submission script for EPFL RCP cluster.
# Run this ON THE CLUSTER (after ssh rcp), NOT on your laptop.
#
# Usage:
#   bash scripts/run_cluster.sh              # full pipeline
#   bash scripts/run_cluster.sh train        # Guardian only
#   bash scripts/run_cluster.sh evader       # Evader only
#   bash scripts/run_cluster.sh stress       # stress cases only
#   bash scripts/run_cluster.sh robust       # robust training only
#   bash scripts/run_cluster.sh eval         # evaluation only
# ===================================================================

# ── Fixed cluster settings ───────────────────────────────────
UID_NUM=309057
IMAGE="registry.rcp.epfl.ch/ee559/environment-with-packages:latest"
GPU=1
PVC="claimname=home,path=/home/mstojano"
PROJECT_DIR="/home/mstojano/red_queen"

# Export HOME and cache dir to avoid the UID/passwd container bug
ENV_SETUP="export HOME=/home/mstojano; export TORCHINDUCTOR_CACHE_DIR=/home/mstojano/.torchinductor-cache; cd ${PROJECT_DIR}"

# ── Select which step to run ─────────────────────────────────
STEP="${1:-full}"

case $STEP in

  train)
    echo "Submitting: Guardian training"
    runai submit \
      --run-as-uid $UID_NUM \
      --image $IMAGE \
      --gpu $GPU \
      --existing-pvc $PVC \
      --command -- bash -lc "${ENV_SETUP}; pip install --break-system-packages -e . 2>/dev/null; python -m red_queen.train_guardian --config configs/default.yaml"
    ;;

  evader)
    # IMPORTANT: Update GUARDIAN_PATH to your actual run directory
    GUARDIAN_PATH="outputs/run_001/guardian_best"
    echo "Submitting: Evader training"
    runai submit \
      --run-as-uid $UID_NUM \
      --image $IMAGE \
      --gpu $GPU \
      --existing-pvc $PVC \
      --command -- bash -lc "${ENV_SETUP}; pip install --break-system-packages -e . 2>/dev/null; python -m red_queen.evader_policy --config configs/default.yaml --guardian_path ${GUARDIAN_PATH}"
    ;;

  stress)
    GUARDIAN_PATH="outputs/run_001/guardian_best"
    EVADER_PATH="outputs/run_002/evader_policy.pt"
    echo "Submitting: Generate stress cases"
    runai submit \
      --run-as-uid $UID_NUM \
      --image $IMAGE \
      --gpu $GPU \
      --existing-pvc $PVC \
      --command -- bash -lc "${ENV_SETUP}; pip install --break-system-packages -e . 2>/dev/null; python -m red_queen.generate_stress_cases --config configs/default.yaml --guardian_path ${GUARDIAN_PATH} --evader_path ${EVADER_PATH} --output_dir outputs/stress_cases"
    ;;

  robust)
    GUARDIAN_PATH="outputs/run_001/guardian_best"
    STRESS_CASES="outputs/stress_cases/stress_cases.parquet"
    echo "Submitting: Robust retraining"
    runai submit \
      --run-as-uid $UID_NUM \
      --image $IMAGE \
      --gpu $GPU \
      --existing-pvc $PVC \
      --command -- bash -lc "${ENV_SETUP}; pip install --break-system-packages -e . 2>/dev/null; python -m red_queen.robust_train --config configs/default.yaml --guardian_path ${GUARDIAN_PATH} --stress_cases ${STRESS_CASES}"
    ;;

  eval)
    BASELINE="outputs/run_001/guardian_best"
    ROBUST="outputs/run_003/guardian_robust"
    STRESS="outputs/stress_cases/stress_cases.parquet"
    echo "Submitting: Evaluation"
    runai submit \
      --run-as-uid $UID_NUM \
      --image $IMAGE \
      --gpu $GPU \
      --existing-pvc $PVC \
      --command -- bash -lc "${ENV_SETUP}; pip install --break-system-packages -e . 2>/dev/null; python -m red_queen.evaluate --config configs/default.yaml --baseline_path ${BASELINE} --robust_path ${ROBUST} --stress_cases ${STRESS} --output_dir outputs/evaluation"
    ;;

  full)
    echo "Submitting: Full pipeline"
    runai submit \
      --run-as-uid $UID_NUM \
      --image $IMAGE \
      --gpu $GPU \
      --existing-pvc $PVC \
      --command -- bash -lc "${ENV_SETUP}; pip install --break-system-packages -e . 2>/dev/null; bash scripts/run_full_pipeline.sh"
    ;;

  *)
    echo "Unknown step: $STEP"
    echo "Usage: bash scripts/run_cluster.sh [train|evader|stress|robust|eval|full]"
    exit 1
    ;;
esac

echo ""
echo "Job submitted. Check status with:"
echo "  runai list"
echo "  runai logs <job-name>"
