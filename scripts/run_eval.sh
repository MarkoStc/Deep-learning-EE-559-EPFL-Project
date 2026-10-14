#!/bin/bash
# Run evaluation only (after training is done).
# Adjust paths to match your actual run directories.

set -e

CONFIG="configs/legacy.yaml"
BASELINE="outputs/run_001/guardian_best"
ROBUST="outputs/run_003/guardian_robust"
STRESS="outputs/stress_cases/stress_cases.parquet"
EVAL_DIR="outputs/evaluation"

python -m red_queen.evaluate \
    --config $CONFIG \
    --baseline_path $BASELINE \
    --robust_path $ROBUST \
    --stress_cases $STRESS \
    --output_dir $EVAL_DIR

echo "Evaluation complete. Results in: $EVAL_DIR"
