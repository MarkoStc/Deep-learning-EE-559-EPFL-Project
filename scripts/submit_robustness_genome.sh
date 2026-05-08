#!/bin/bash
set -euo pipefail

UID_NUM=309057
IMAGE="registry.rcp.epfl.ch/ee559/environment-with-packages:latest"
GPU="${GPU:-1}"
PVC="claimname=home,path=/home/mstojano"
PROJECT_DIR="${PROJECT_DIR:-/home/mstojano/red_queen_moderation_sequence}"

CONFIG="${CONFIG:-configs/default.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/robustness_genome_final}"
BASELINE_GUARDIAN_PATH="${BASELINE_GUARDIAN_PATH:-outputs/run_029/guardian_best}"
FINAL_GUARDIAN_PATH="${FINAL_GUARDIAN_PATH:-outputs/arms_race/round_05/guardian_robust}"
ARMS_RACE_DIR="${ARMS_RACE_DIR:-outputs/arms_race}"
LINEAGE="${LINEAGE:-all}"
MAX_RAW_ROWS="${MAX_RAW_ROWS:-0}"
HATE_SEED_EXAMPLES="${HATE_SEED_EXAMPLES:-5000}"
SAFE_SEED_EXAMPLES="${SAFE_SEED_EXAMPLES:-5000}"
REQUIRE_IDENTITY_FRACTION="${REQUIRE_IDENTITY_FRACTION:-0.65}"
GENERATIONS="${GENERATIONS:-120}"
CANDIDATES_PER_GENERATION="${CANDIDATES_PER_GENERATION:-4096}"
MAX_DEPTH="${MAX_DEPTH:-8}"
MAX_EVALUATIONS="${MAX_EVALUATIONS:-0}"
WALL_TIME_MINUTES="${WALL_TIME_MINUTES:-0}"
BATCH_SIZE="${BATCH_SIZE:-192}"
THRESHOLD="${THRESHOLD:-0.5}"
SEED="${SEED:-42}"
CACHE_DIR="${CACHE_DIR:-data/robustness_genome}"
COVERAGE_TOP_K="${COVERAGE_TOP_K:-12}"
LINEAGE_TOP_K="${LINEAGE_TOP_K:-30000}"
MIN_GROUP_EXAMPLES="${MIN_GROUP_EXAMPLES:-25}"
CLUSTER_MAX_CASES="${CLUSTER_MAX_CASES:-5000}"
FAILURE_CLUSTERS="${FAILURE_CLUSTERS:-16}"
LOG_EVERY="${LOG_EVERY:-5}"

JOB_SCRIPT="${PROJECT_DIR}/scripts/.run_robustness_genome_job.sh"

cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
set -euxo pipefail

export HOME=/home/mstojano
export TORCHINDUCTOR_CACHE_DIR=/home/mstojano/.torchinductor-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ${PROJECT_DIR}

echo "PWD: \$(pwd)"
echo "Checking files..."
ls -lah scripts/run_robustness_genome.sh
ls -lah src/rq_robustness_genome.py
ls -lah ${FINAL_GUARDIAN_PATH}
ls -lah ${BASELINE_GUARDIAN_PATH}

pip install --break-system-packages -r requirements.txt 2>/dev/null

CONFIG='${CONFIG}' \\
OUTPUT_DIR='${OUTPUT_DIR}' \\
BASELINE_GUARDIAN_PATH='${BASELINE_GUARDIAN_PATH}' \\
FINAL_GUARDIAN_PATH='${FINAL_GUARDIAN_PATH}' \\
ARMS_RACE_DIR='${ARMS_RACE_DIR}' \\
LINEAGE='${LINEAGE}' \\
MAX_RAW_ROWS='${MAX_RAW_ROWS}' \\
HATE_SEED_EXAMPLES='${HATE_SEED_EXAMPLES}' \\
SAFE_SEED_EXAMPLES='${SAFE_SEED_EXAMPLES}' \\
REQUIRE_IDENTITY_FRACTION='${REQUIRE_IDENTITY_FRACTION}' \\
GENERATIONS='${GENERATIONS}' \\
CANDIDATES_PER_GENERATION='${CANDIDATES_PER_GENERATION}' \\
MAX_DEPTH='${MAX_DEPTH}' \\
MAX_EVALUATIONS='${MAX_EVALUATIONS}' \\
WALL_TIME_MINUTES='${WALL_TIME_MINUTES}' \\
BATCH_SIZE='${BATCH_SIZE}' \\
THRESHOLD='${THRESHOLD}' \\
SEED='${SEED}' \\
CACHE_DIR='${CACHE_DIR}' \\
COVERAGE_TOP_K='${COVERAGE_TOP_K}' \\
LINEAGE_TOP_K='${LINEAGE_TOP_K}' \\
MIN_GROUP_EXAMPLES='${MIN_GROUP_EXAMPLES}' \\
CLUSTER_MAX_CASES='${CLUSTER_MAX_CASES}' \\
FAILURE_CLUSTERS='${FAILURE_CLUSTERS}' \\
LOG_EVERY='${LOG_EVERY}' \\
bash scripts/run_robustness_genome.sh
EOF

chmod +x "${JOB_SCRIPT}"

echo "Submitting RQ Robustness Genome"
echo "Project:      ${PROJECT_DIR}"
echo "Lineage:      ${LINEAGE}"
echo "Guardian:     ${FINAL_GUARDIAN_PATH}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Generations:  ${GENERATIONS}"
echo "Candidates:   ${CANDIDATES_PER_GENERATION} per generation"
echo "Job script:   ${JOB_SCRIPT}"
echo ""

runai submit \
  --run-as-uid ${UID_NUM} \
  --image ${IMAGE} \
  --gpu ${GPU} \
  --existing-pvc ${PVC} \
  --command -- bash "${JOB_SCRIPT}"

echo ""
echo "Job submitted. Monitor with:"
echo "  runai list"
echo "  runai logs <job-name> -f"
echo ""
echo "When done, download with:"
echo "  scp -r rcp:${PROJECT_DIR}/${OUTPUT_DIR} ~/Desktop/EPFL/red_queen_moderation/outputs/"
