#!/bin/bash
# ===================================================================
# Submit the Red Queen outer loop (5-round arms race) as one RunAI job.
#
# USAGE (on the cluster):
#   cd /home/mstojano/red_queen
#   bash scripts/submit_outer_loop.sh
#
# PREREQUISITE: Guardian trained at outputs/run_029/guardian_best
# ===================================================================

NUM_ROUNDS="${1:-5}"
GUARDIAN_PATH="outputs/run_029/guardian_best"

echo "Submitting outer loop: ${NUM_ROUNDS} rounds"
echo "Initial Guardian: ${GUARDIAN_PATH}"

runai submit \
  --run-as-uid 309057 \
  --image registry.rcp.epfl.ch/ee559/environment-with-packages:latest \
  --gpu 1 \
  --existing-pvc claimname=home,path=/home/mstojano \
  --command -- bash -lc "export HOME=/home/mstojano; export TORCHINDUCTOR_CACHE_DIR=/home/mstojano/.torchinductor-cache; cd /home/mstojano/red_queen; pip install --break-system-packages -e . 2>/dev/null; python -m red_queen.outer_loop --config configs/default.yaml --initial_guardian ${GUARDIAN_PATH} --num_rounds ${NUM_ROUNDS} --output_dir outputs/arms_race"

echo ""
echo "Job submitted. Monitor with:"
echo "  runai list"
echo "  runai logs <job-name> -f"
echo ""
echo "Expected runtime: ~2-3 hours for ${NUM_ROUNDS} rounds on A100"
echo ""
echo "When done, download results:"
echo "  scp -r rcp:/home/mstojano/red_queen/outputs/arms_race ~/Desktop/EPFL/Deep\ Learning/red_queen/outputs/"
