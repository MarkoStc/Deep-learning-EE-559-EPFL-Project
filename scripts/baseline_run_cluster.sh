#!/usr/bin/env bash
set -euo pipefail

GUARDIAN_DIR="${1:-outputs/run_029/guardian_best}"
OUTPUT_DIR="${2:-outputs/baseline_results}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

cd /home/mstojano/red_queen

CMD=(python3 -m src.baseline_evaluation \
  --config configs/default.yaml \
  --guardian-dir "$GUARDIAN_DIR" \
  --output-dir "$OUTPUT_DIR")

if [[ -n "$MAX_EXAMPLES" ]]; then
  CMD+=(--max-examples "$MAX_EXAMPLES")
fi

"${CMD[@]}"
