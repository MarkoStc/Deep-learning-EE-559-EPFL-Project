#!/bin/bash
# ===================================================================
# Submit the remaining pipeline (Steps 2–5) as a SINGLE RunAI job.
#
# USAGE (on the cluster):
#   cd /home/mstojano/red_queen
#   bash scripts/submit_remaining_pipeline.sh
# ===================================================================

runai submit \
  --run-as-uid 309057 \
  --image registry.rcp.epfl.ch/ee559/environment-with-packages:latest \
  --gpu 1 \
  --existing-pvc claimname=home,path=/home/mstojano \
  --command -- bash -lc 'export HOME=/home/mstojano; export TORCHINDUCTOR_CACHE_DIR=/home/mstojano/.torchinductor-cache; cd /home/mstojano/red_queen; pip install --break-system-packages -r requirements.txt 2>/dev/null; bash scripts/run_remaining_pipeline.sh'

echo ""
echo "Job submitted. Monitor with:"
echo "  runai list"
echo "  runai logs <job-name>"
echo "  runai logs <job-name> -f    (live follow)"
