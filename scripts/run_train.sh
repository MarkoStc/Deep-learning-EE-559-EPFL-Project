#!/bin/bash
# Train the baseline Guardian locally.
# Run from the red_queen/ project root.

set -e

echo "=== Training Baseline Guardian ==="
python -m red_queen.train_guardian --config configs/default.yaml

echo ""
echo "Done. Check outputs/ for the latest run directory."
echo "The Guardian checkpoint is in outputs/run_XXX/guardian_best/"
