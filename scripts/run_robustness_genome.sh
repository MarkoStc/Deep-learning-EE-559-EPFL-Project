#!/bin/bash
set -euo pipefail

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

printf '%s\n' "============================================"
printf '%s\n' "  RQ Robustness Genome"
printf '%s\n' "============================================"
printf 'Project pwd:       %s\n' "$(pwd)"
printf 'Lineage:           %s\n' "${LINEAGE}"
printf 'Baseline Guardian: %s\n' "${BASELINE_GUARDIAN_PATH}"
printf 'Final Guardian:    %s\n' "${FINAL_GUARDIAN_PATH}"
printf 'Seeds hate/safe:   %s/%s\n' "${HATE_SEED_EXAMPLES}" "${SAFE_SEED_EXAMPLES}"
printf 'Gen/candidates:    %s/%s\n' "${GENERATIONS}" "${CANDIDATES_PER_GENERATION}"
printf 'Output dir:        %s\n\n' "${OUTPUT_DIR}"

python3 -m src.rq_robustness_genome \
  --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --baseline_guardian_path "${BASELINE_GUARDIAN_PATH}" \
  --final_guardian_path "${FINAL_GUARDIAN_PATH}" \
  --arms_race_dir "${ARMS_RACE_DIR}" \
  --lineage "${LINEAGE}" \
  --max_raw_rows "${MAX_RAW_ROWS}" \
  --hate_seed_examples "${HATE_SEED_EXAMPLES}" \
  --safe_seed_examples "${SAFE_SEED_EXAMPLES}" \
  --require_identity_fraction "${REQUIRE_IDENTITY_FRACTION}" \
  --generations "${GENERATIONS}" \
  --candidates_per_generation "${CANDIDATES_PER_GENERATION}" \
  --max_depth "${MAX_DEPTH}" \
  --max_evaluations "${MAX_EVALUATIONS}" \
  --wall_time_minutes "${WALL_TIME_MINUTES}" \
  --batch_size "${BATCH_SIZE}" \
  --threshold "${THRESHOLD}" \
  --seed "${SEED}" \
  --cache_dir "${CACHE_DIR}" \
  --coverage_top_k "${COVERAGE_TOP_K}" \
  --lineage_top_k "${LINEAGE_TOP_K}" \
  --min_group_examples "${MIN_GROUP_EXAMPLES}" \
  --cluster_max_cases "${CLUSTER_MAX_CASES}" \
  --failure_clusters "${FAILURE_CLUSTERS}" \
  --log_every "${LOG_EVERY}"

printf '\nDone. Main outputs:\n'
printf '  %s/robustness_genome_report.md\n' "${OUTPUT_DIR}"
printf '  %s/genome_archive.csv\n' "${OUTPUT_DIR}"
printf '  %s/robustness_debt_by_group.csv\n' "${OUTPUT_DIR}"
printf '  %s/lineage_summary.csv\n' "${OUTPUT_DIR}"
printf '  %s/coverage_growth.png\n' "${OUTPUT_DIR}"
