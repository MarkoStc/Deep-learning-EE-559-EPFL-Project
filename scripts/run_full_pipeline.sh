#!/bin/bash
# Full Red Queen pipeline: train → evade → stress → robust → evaluate
# Run from the red_queen/ project root.
# Adjust paths below after each step produces its run directory.

set -e

# ── Configuration ────────────────────────────────────────────
CONFIG="configs/default.yaml"

# These paths will be created by each step. Update if your run numbers differ.
GUARDIAN_RUN="outputs/run_001"
EVADER_RUN="outputs/run_002"
STRESS_DIR="outputs/stress_cases"
ROBUST_RUN="outputs/run_003"
EVAL_DIR="outputs/evaluation"

# ── Step 1: Train baseline Guardian ──────────────────────────
echo "============================================"
echo "  STEP 1: Train Baseline Guardian"
echo "============================================"
python3 -m src.train_guardian --config $CONFIG
GUARDIAN_PATH="${GUARDIAN_RUN}/guardian_best"

# ── Step 2: Train Evader policy ──────────────────────────────
echo ""
echo "============================================"
echo "  STEP 2: Train Evader Policy"
echo "============================================"
python3 -m src.evader_policy \
    --config $CONFIG \
    --guardian_path $GUARDIAN_PATH
EVADER_PATH="${EVADER_RUN}/evader_policy.pt"

# ── Step 3: Generate stress cases ────────────────────────────
echo ""
echo "============================================"
echo "  STEP 3: Generate Stress Cases"
echo "============================================"
python3 -m src.generate_stress_cases \
    --config $CONFIG \
    --guardian_path $GUARDIAN_PATH \
    --evader_path $EVADER_PATH \
    --output_dir $STRESS_DIR

# ── Step 4: Robust retrain Guardian ──────────────────────────
echo ""
echo "============================================"
echo "  STEP 4: Robust Retrain Guardian"
echo "============================================"
python3 -m src.robust_train \
    --config $CONFIG \
    --guardian_path $GUARDIAN_PATH \
    --stress_cases "${STRESS_DIR}/stress_cases.parquet"
ROBUST_PATH="${ROBUST_RUN}/guardian_robust"

# ── Step 5: Full evaluation ──────────────────────────────────
echo ""
echo "============================================"
echo "  STEP 5: Full Evaluation"
echo "============================================"
python3 -m src.evaluate \
    --config $CONFIG \
    --baseline_path $GUARDIAN_PATH \
    --robust_path $ROBUST_PATH \
    --stress_cases "${STRESS_DIR}/stress_cases.parquet" \
    --output_dir $EVAL_DIR

echo ""
echo "============================================"
echo "  PIPELINE COMPLETE"
echo "============================================"
echo "Results in: $EVAL_DIR"
echo "Key files:"
echo "  - ${EVAL_DIR}/summary_table.csv"
echo "  - ${EVAL_DIR}/all_metrics.json"
echo "  - ${EVAL_DIR}/clean_vs_stress_comparison.png"
echo "  - ${EVAL_DIR}/asr_by_action.png"
echo "  - ${EVAL_DIR}/fairness_subgroups.png"


