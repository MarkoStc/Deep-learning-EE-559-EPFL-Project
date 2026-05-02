#!/bin/bash
# ===================================================================
# Run the additive Cascade Red Queen module from the project root.
# This assumes you are already inside the RunAI container or an
# equivalent environment with the project dependencies installed.
#
# Usage on the cluster inside /home/mstojano/red_queen:
#   bash scripts/run_cascade_redqueen.sh
#
# Optional overrides:
#   GUARDIAN_PATH=outputs/arms_race/round_04/guardian_robust \
#   MAX_DEPTH=5 BEAM_WIDTH=10 SAMPLES_PER_ACTION=2 \
#   bash scripts/run_cascade_redqueen.sh
# ===================================================================

set -euo pipefail

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

if [ ! -d "$GUARDIAN_PATH" ]; then
  echo "ERROR: Guardian checkpoint not found: $GUARDIAN_PATH"
  echo "Set GUARDIAN_PATH=... or finish the outer loop first."
  exit 1
fi

BASELINE_ARG=""
if [ -n "$BASELINE_GUARDIAN_PATH" ] && [ -d "$BASELINE_GUARDIAN_PATH" ]; then
  BASELINE_ARG="--baseline_guardian_path ${BASELINE_GUARDIAN_PATH}"
else
  echo "[WARN] Baseline checkpoint not found or empty: ${BASELINE_GUARDIAN_PATH}"
  echo "[WARN] Continuing without baseline comparison."
fi

echo "============================================"
echo "  Cascade Red Queen"
echo "============================================"
echo "Guardian:        ${GUARDIAN_PATH}"
echo "Baseline:        ${BASELINE_GUARDIAN_PATH}"
echo "Split:           ${SPLIT}"
echo "Attack examples: ${MAX_EXAMPLES}"
echo "Depth/beam/samp: ${MAX_DEPTH}/${BEAM_WIDTH}/${SAMPLES_PER_ACTION}"
echo "Output dir:      ${OUTPUT_DIR}"
echo ""

python3 -m src.rq_cascade_redqueen \
  --config "$CONFIG" \
  --guardian_path "$GUARDIAN_PATH" \
  ${BASELINE_ARG} \
  --split "$SPLIT" \
  --max_examples "$MAX_EXAMPLES" \
  --clean_eval_examples "$CLEAN_EVAL_EXAMPLES" \
  --max_depth "$MAX_DEPTH" \
  --beam_width "$BEAM_WIDTH" \
  --samples_per_action "$SAMPLES_PER_ACTION" \
  --batch_size "$BATCH_SIZE" \
  --shield_variants "$SHIELD_VARIANTS" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "Done. Main outputs:"
echo "  ${OUTPUT_DIR}/cascade_report.md"
echo "  ${OUTPUT_DIR}/cascade_metrics.json"
echo "  ${OUTPUT_DIR}/cascade_attacks_sample.csv"
echo "  ${OUTPUT_DIR}/cascade_vulnerability_dashboard.png"
