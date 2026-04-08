# Red Queen Moderation — Makefile
# Usage: make <target>

CONFIG = configs/default.yaml

# ── Update these after each step creates its run dir ──────────
GUARDIAN_RUN   = outputs/run_001
EVADER_RUN     = outputs/run_002
ROBUST_RUN     = outputs/run_003
STRESS_DIR     = outputs/stress_cases
EVAL_DIR       = outputs/evaluation

GUARDIAN_PATH  = $(GUARDIAN_RUN)/guardian_best
EVADER_PATH    = $(EVADER_RUN)/evader_policy.pt
ROBUST_PATH    = $(ROBUST_RUN)/guardian_robust
STRESS_FILE    = $(STRESS_DIR)/stress_cases.parquet

.PHONY: install train evader stress robust eval all clean help

help:
	@echo "Targets:"
	@echo "  install  — install Python dependencies"
	@echo "  train    — train baseline Guardian"
	@echo "  evader   — train Evader policy"
	@echo "  stress   — generate stress-test cases"
	@echo "  robust   — robust-retrain Guardian"
	@echo "  eval     — full evaluation + plots"
	@echo "  all      — run the full pipeline"
	@echo "  clean    — remove all outputs"

install:
	pip install -r requirements.txt
	python3 -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

train:
	python3 -m src.train_guardian --config $(CONFIG)

evader:
	python3 -m src.evader_policy --config $(CONFIG) --guardian_path $(GUARDIAN_PATH)

stress:
	python3 -m src.generate_stress_cases --config $(CONFIG) \
		--guardian_path $(GUARDIAN_PATH) \
		--evader_path $(EVADER_PATH) \
		--output_dir $(STRESS_DIR)

robust:
	python3 -m src.robust_train --config $(CONFIG) \
		--guardian_path $(GUARDIAN_PATH) \
		--stress_cases $(STRESS_FILE)

eval:
	python3 -m src.evaluate --config $(CONFIG) \
		--baseline_path $(GUARDIAN_PATH) \
		--robust_path $(ROBUST_PATH) \
		--stress_cases $(STRESS_FILE) \
		--output_dir $(EVAL_DIR)

all: train evader stress robust eval
	@echo "Full pipeline complete. Results in $(EVAL_DIR)/"

clean:
	rm -rf outputs/ data/
	@echo "Cleaned outputs and data."
