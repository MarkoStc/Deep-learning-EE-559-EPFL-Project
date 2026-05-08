# RQ Robustness Genome

This is an additive-only research patch for the Red Queen Moderation project.

It adds one large, compute-heavy audit:

**RQ Robustness Genome** — a coverage-guided, dual-objective, MAP-Elites-style fuzzer for hate/toxicity moderation.

It does not change any existing project files.

## Why this exists

The original Red Queen story shows that a single-action RL Evader can be patched over five rounds. The Cascade and Frontier audits show stronger attack search still finds weaknesses.

This patch goes one level deeper: it tries to build a **map of the model's hidden failure space**, not just one best adversarial attack.

It asks:

1. Can toxic examples be perturbed so the Guardian misses them?
2. Can safe examples be perturbed so the Guardian falsely flags them?
3. Which identity mentions are associated with the biggest remaining safety debt?
4. Do discovered failures transfer across the full Guardian lineage from baseline to round 5?
5. Does the search keep discovering new internal/behavioral coverage, or does it saturate?

## Research inspiration

This combines ideas from:

- HateCheck-style functional testing for hate speech detection.
- CheckList-style behavioral NLP testing beyond average accuracy.
- Coverage-guided neural fuzzing / DeepHunter-style defect discovery.
- Quality-diversity / MAP-Elites style archive search.
- Civil Comments / WILDS-style subgroup and identity robustness thinking.

The exact contribution here is the combination:

```text
Red Queen Guardian lineage
+ coverage-guided neural fuzzing
+ dual moderation objectives: hide-hate and false-alarm
+ identity-conditioned archive cells
+ DistilBERT hidden-state coverage
+ lineage transfer audit
+ failure archetype clustering
```

## New files

```text
src/rq_robustness_genome.py
scripts/run_robustness_genome.sh
scripts/submit_robustness_genome.sh
README_ROBUSTNESS_GENOME.md
```

## Safety note

The script does not export raw comment text. Public CSV files contain stable hashes, actions, probabilities, identities, and metrics. This is intentional because the underlying dataset can contain harmful text.

## Default full run

```bash
OUTPUT_DIR=outputs/robustness_genome_final \
bash scripts/submit_robustness_genome.sh
```

Default settings:

```text
LINEAGE=all
MAX_RAW_ROWS=0
HATE_SEED_EXAMPLES=5000
SAFE_SEED_EXAMPLES=5000
GENERATIONS=120
CANDIDATES_PER_GENERATION=4096
MAX_DEPTH=8
LINEAGE_TOP_K=30000
```

This can evaluate roughly hundreds of thousands of generated candidates plus a large lineage audit.

## Smoke test

```bash
LINEAGE=baseline_final \
MAX_RAW_ROWS=50000 \
HATE_SEED_EXAMPLES=100 \
SAFE_SEED_EXAMPLES=100 \
GENERATIONS=3 \
CANDIDATES_PER_GENERATION=128 \
MAX_DEPTH=3 \
LINEAGE_TOP_K=500 \
BATCH_SIZE=128 \
OUTPUT_DIR=outputs/robustness_genome_smoke \
bash scripts/submit_robustness_genome.sh
```

## Heavy run

```bash
LINEAGE=all \
MAX_RAW_ROWS=0 \
HATE_SEED_EXAMPLES=10000 \
SAFE_SEED_EXAMPLES=10000 \
GENERATIONS=220 \
CANDIDATES_PER_GENERATION=8192 \
MAX_DEPTH=10 \
LINEAGE_TOP_K=80000 \
BATCH_SIZE=256 \
CLUSTER_MAX_CASES=10000 \
FAILURE_CLUSTERS=24 \
OUTPUT_DIR=outputs/robustness_genome_deep \
bash scripts/submit_robustness_genome.sh
```

This is the version intended to run for hours. It is not artificially slowed down; it spends compute on model queries, hidden-state coverage, archive construction, lineage transfer, and clustering.

## Main outputs

Open this first:

```text
robustness_genome_report.md
```

Important files:

```text
genome_archive.csv
coverage_growth.csv
lineage_summary.csv
lineage_case_audit.csv
robustness_debt_by_group.csv
lineage_debt_by_group.csv
failure_archetypes.csv
action_family_summary.csv
coverage_growth.png
fuzzer_pressure.png
genome_depth_edit_heatmap.png
lineage_failure_transfer.png
robustness_debt_leaderboard.png
failure_archetypes_pca.png
```

## How to interpret

- `hide_hate`: true toxic/hateful examples that become safe-looking to the Guardian.
- `false_alarm`: true safe examples that become hate-looking to the Guardian.
- `induced_failure`: the original text was classified correctly, but a generated variant caused a failure.
- `archive cells`: distinct discovered behaviors, not just raw examples.
- `coverage cells`: approximate DistilBERT hidden-state coverage buckets.
- `lineage transfer`: whether failures discovered against the final Guardian also break earlier or intermediate Guardians.

A strong result is not just high ASR. A strong result is a rich archive showing which failure types survive robust retraining, which ones are newly created, and which identity/objective regions have the biggest safety debt.
