#!/bin/bash
# ===================================================================
# Run Steps 2–5 of the Red Queen pipeline in a SINGLE RunAI job.
# No manual waiting between steps — they run sequentially.
#
# USAGE (on the cluster, from /home/mstojano/red_queen):
#   bash scripts/submit_remaining_pipeline.sh
#
# PREREQUISITES:
#   - Guardian already trained at outputs/run_029/guardian_best
# ===================================================================

set -e  # stop on first error

CONFIG="configs/default.yaml"
GUARDIAN_PATH="outputs/run_029/guardian_best"
STRESS_DIR="outputs/stress_cases"

echo "============================================"
echo "  Red Queen Pipeline — Steps 2 through 5"
echo "  Guardian checkpoint: ${GUARDIAN_PATH}"
echo "============================================"

# ── Step 2: Train Evader ─────────────────────────────────────
echo ""
echo ">>> STEP 2: Training Evader policy..."
python3 -m src.evader_policy \
    --config $CONFIG \
    --guardian_path $GUARDIAN_PATH

# Find the evader checkpoint that was just created
# (latest run_XXX directory containing evader_policy.pt)
EVADER_PATH=$(find outputs/ -name "evader_policy.pt" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
echo ">>> Evader saved at: ${EVADER_PATH}"

if [ -z "$EVADER_PATH" ]; then
    echo "ERROR: Could not find evader_policy.pt"
    exit 1
fi

# ── Step 3: Generate stress cases ────────────────────────────
echo ""
echo ">>> STEP 3: Generating stress cases..."
python3 -m src.generate_stress_cases \
    --config $CONFIG \
    --guardian_path $GUARDIAN_PATH \
    --evader_path $EVADER_PATH \
    --output_dir $STRESS_DIR

if [ ! -f "${STRESS_DIR}/stress_cases.parquet" ]; then
    echo "ERROR: Stress cases not generated"
    exit 1
fi

# ── Step 4: Robust retrain Guardian ──────────────────────────
echo ""
echo ">>> STEP 4: Robust retraining Guardian..."
python3 -m src.robust_train \
    --config $CONFIG \
    --guardian_path $GUARDIAN_PATH \
    --stress_cases ${STRESS_DIR}/stress_cases.parquet

# Find the robust checkpoint that was just created
ROBUST_PATH=$(find outputs/ -name "guardian_robust" -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
echo ">>> Robust Guardian saved at: ${ROBUST_PATH}"

if [ -z "$ROBUST_PATH" ]; then
    echo "ERROR: Could not find guardian_robust directory"
    exit 1
fi

# ── Step 5: Full evaluation ──────────────────────────────────
echo ""
echo ">>> STEP 5: Running full evaluation..."
python3 -m src.evaluate \
    --config $CONFIG \
    --baseline_path $GUARDIAN_PATH \
    --robust_path $ROBUST_PATH \
    --stress_cases ${STRESS_DIR}/stress_cases.parquet \
    --output_dir outputs/evaluation

echo ""
echo "============================================"
echo "  PIPELINE COMPLETE"
echo "============================================"
echo ""
echo "Key outputs:"
echo "  Evader policy:    ${EVADER_PATH}"
echo "  Stress cases:     ${STRESS_DIR}/"
echo "  Robust Guardian:  ${ROBUST_PATH}"
echo "  Evaluation:       outputs/evaluation/"
echo ""
echo "Download results to your laptop with:"
echo "  scp -r rcp:/home/mstojano/red_queen/outputs ~/Desktop/EPFL/Deep\ Learning/red_queen/"
