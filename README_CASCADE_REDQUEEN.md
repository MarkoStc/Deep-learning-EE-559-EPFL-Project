# Cascade Red Queen — additive extension

This extension adds a stronger post-outer-loop experiment without modifying any existing project file.

## What is new

The original Red Queen loop trains an Evader that chooses one perturbation at a time. Cascade Red Queen asks a harder question:

> After five rounds of robust retraining, can a black-box attacker still find compact *sequences* of perturbations that fool the Guardian?

It adds four pieces:

1. **Compositional beam-search red-team**: searches perturbation programs like `homoglyph+typo+emoji`, not just one action.
2. **Multi-objective scoring**: rewards evasion but penalizes large edits, long sequences, and obvious artifacts.
3. **Canonicalization shield**: evaluates a no-retraining defense that runs the Guardian on normalized variants and takes the maximum hate probability.
4. **Failure archetyping**: separates examples the Guardian already missed from examples that became misses only after perturbation.

## Files

Place these files in your repo:

```text
src/rq_cascade_redqueen.py
scripts/run_cascade_redqueen.sh
scripts/submit_cascade_redqueen.sh
README_CASCADE_REDQUEEN.md
```

## Main command

```bash
python3 -m src.rq_cascade_redqueen \
  --config configs/default.yaml \
  --guardian_path outputs/arms_race/round_05/guardian_robust \
  --baseline_guardian_path outputs/run_029/guardian_best \
  --split test \
  --max_examples 600 \
  --beam_width 8 \
  --max_depth 4 \
  --samples_per_action 2 \
  --output_dir outputs/cascade_redqueen_final
```

## Expected outputs

```text
outputs/cascade_redqueen_final/cascade_report.md
outputs/cascade_redqueen_final/cascade_metrics.json
outputs/cascade_redqueen_final/cascade_attacks.parquet
outputs/cascade_redqueen_final/cascade_attacks_sample.csv
outputs/cascade_redqueen_final/top_sequences.csv
outputs/cascade_redqueen_final/asr_by_depth.csv
outputs/cascade_redqueen_final/cascade_vulnerability_dashboard.png
```

The CSV and Markdown reports are sanitized for human reading. The raw Parquet file contains original dataset text and should be handled as controlled research data.

## How to interpret the key numbers

- **Raw cascade ASR**: how often the composed attack fools the selected Guardian.
- **Induced evasion rate**: how often an example was detected before perturbation but missed after perturbation. This is the cleanest robustness-failure signal.
- **Intrinsic original miss rate**: how often the Guardian already missed the original hateful example. This is a base classifier recall issue, not a perturbation issue.
- **Shielded cascade ASR**: how many composed attacks still pass after canonical normalization.
- **Shield recovery on successes**: fraction of raw successful attacks recovered by the shield.

If raw cascade ASR is high but shielded ASR is low, your remaining weakness is mostly tokenization/canonicalization. If both are high, the final Guardian still has semantic blind spots and needs another robustness strategy.
