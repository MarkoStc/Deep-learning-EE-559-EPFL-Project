# Red Queen Moderation

**An offline RL framework for stress-testing hate speech classifiers.**

A lightweight "Evader" agent learns which benchmark-style text perturbations
fool a hate-speech "Guardian" classifier. The Guardian is then retrained on the
discovered stress cases to become more robust. The entire pipeline is offline,
safe, and reproducible.

## Team
- 3 EPFL students — EE-559 Deep Learning Mini-Project

## Quick Start

```bash
# 1. Install dependencies
uv sync --extra cpu   # or --extra cu126 for CUDA 12.6

# 2. Train the baseline Guardian
python -m red_queen.train_guardian --config configs/default.yaml

# 3. Train the Evader policy
python -m red_queen.evader_policy --config configs/default.yaml

# 4. Generate stress-test cases
python -m red_queen.generate_stress_cases --config configs/default.yaml

# 5. Robust-retrain the Guardian
python -m red_queen.robust_train --config configs/default.yaml

# 6. Evaluate everything
python -m red_queen.evaluate --config configs/default.yaml
```

## Project Structure

```
red_queen/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # all hyperparams & paths
├── src/
│   ├── __init__.py
│   ├── data_loading.py        # dataset download & splitting
│   ├── preprocess.py          # tokenization & perturbation functions
│   ├── models.py              # Guardian model definition
│   ├── train_guardian.py      # fine-tune DistilBERT classifier
│   ├── evader_env.py          # bandit environment
│   ├── evader_policy.py       # REINFORCE training for Evader
│   ├── generate_stress_cases.py  # apply learned policy to produce stress set
│   ├── robust_train.py        # retrain Guardian with augmented data
│   ├── evaluate.py            # full evaluation + plots
│   └── utils.py               # seeds, logging, config loading, masking
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── run_full_pipeline.sh
│   └── run_cluster.sh         # RunAI submission for EPFL RCP
├── data/                      # auto-populated by data_loading.py
└── outputs/                   # metrics, plots, checkpoints per run
```

## Datasets
- **Civil Comments** (primary) — via HuggingFace `google/civil_comments`
- **HateXplain** (secondary) — via HuggingFace `hatexplain`
- **Hatemoji** (adversarial test) — from GitHub

## Ethical Considerations
- All perturbations are surface-level transformations of existing labeled data.
- No new hateful content is generated.
- Logs and outputs mask/sanitize hateful text by default.
- The project follows the UN definition of hate speech and EPFL ethics guidelines.

## License
Academic use only — EPFL EE-559 course project.
