"""
Compare raw and effective Oracle ASR across Guardian checkpoints.

This is meant to answer one simple question:
    How vulnerable was the non-robust Guardian under Oracle attack,
    and how much did robust training reduce that vulnerability?

Raw Oracle ASR:
    Fraction of held-out hateful test examples where ANY action, including noop,
    is predicted safe by the Guardian.

Effective Oracle ASR:
    Among held-out hateful test examples that the Guardian correctly catches in
    their original/noop form, fraction where at least one NON-noop perturbation
    flips the prediction to safe.

Usage:
    python3 -m src.baseline_oracle_guardian_comparison \
        --config configs/default.yaml \
        --guardian initial=outputs/run_029/guardian_best \
        --guardian rl_final=outputs/baseline_outer_loop_rl/round_05/guardian_robust \
        --output-dir outputs/baseline_oracle_guardian_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data_loading import load_and_split
from src.models import Guardian
from src.preprocess import ACTION_NAMES, NUM_ACTIONS, apply_perturbation
from src.utils import get_device, load_config, set_seed


def parse_guardian_specs(specs: List[str]) -> List[Tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Guardian spec must look like NAME=PATH, got: {spec!r}"
            )
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Bad guardian spec: {spec!r}")
        parsed.append((name, path))
    return parsed


def predict_actions(
    guardian: Guardian,
    hate_texts: List[str],
    cfg: dict,
) -> Dict[str, np.ndarray]:
    """Return predictions for each perturbation action on all hateful texts."""
    action_preds = {}
    for action_idx, action_name in enumerate(ACTION_NAMES):
        if action_idx == 0:
            perturbed = hate_texts
        else:
            perturbed = [apply_perturbation(t, action_idx, cfg) for t in hate_texts]
        preds = np.array(guardian.predict(perturbed), dtype=int)
        action_preds[action_name] = preds
    return action_preds


def evaluate_guardian(
    name: str,
    checkpoint_path: str,
    cfg: dict,
    test_texts: List[str],
    test_labels: List[int],
    hate_test_texts: List[str],
    device,
) -> Tuple[dict, List[dict]]:
    print(f"\n{'=' * 72}")
    print(f"Evaluating Guardian: {name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'=' * 72}")

    guardian = Guardian(cfg, device)
    guardian.load(checkpoint_path)

    # Clean full-test performance: safe + hateful, no perturbation.
    clean_preds = guardian.predict(test_texts)
    clean_macro_f1 = f1_score(test_labels, clean_preds, average="macro", zero_division=0)
    clean_hate_f1 = f1_score(test_labels, clean_preds, pos_label=1, zero_division=0)

    # Hateful-test adversarial performance.
    action_preds = predict_actions(guardian, hate_test_texts, cfg)
    pred_matrix = np.vstack([action_preds[a] for a in ACTION_NAMES])  # actions x N

    noop_preds = action_preds[ACTION_NAMES[0]]
    base_missed = noop_preds == 0
    base_caught = noop_preds == 1

    raw_success_by_action = pred_matrix == 0
    raw_oracle_success = raw_success_by_action.any(axis=0)

    # Effective attack excludes noop and only conditions on originally-caught texts.
    non_noop_success = raw_success_by_action[1:, :].any(axis=0)
    effective_success = base_caught & non_noop_success

    n_hate = len(hate_test_texts)
    n_base_caught = int(base_caught.sum())
    n_base_missed = int(base_missed.sum())
    n_raw_oracle_success = int(raw_oracle_success.sum())
    n_effective_success = int(effective_success.sum())

    raw_noop_asr = float(base_missed.mean()) if n_hate else 0.0
    raw_oracle_asr = float(raw_oracle_success.mean()) if n_hate else 0.0
    effective_oracle_asr = (
        float(n_effective_success / n_base_caught) if n_base_caught else 0.0
    )
    added_oracle_pp = raw_oracle_asr - raw_noop_asr

    summary = {
        "guardian": name,
        "checkpoint_path": checkpoint_path,
        "clean_macro_f1": float(clean_macro_f1),
        "clean_hate_f1": float(clean_hate_f1),
        "n_test_total": int(len(test_texts)),
        "n_hate_test": int(n_hate),
        "n_base_missed_noop": n_base_missed,
        "n_base_caught_noop": n_base_caught,
        "raw_noop_asr": raw_noop_asr,
        "raw_oracle_asr": raw_oracle_asr,
        "raw_oracle_success_count": n_raw_oracle_success,
        "effective_oracle_asr_conditioned_on_noop_caught": effective_oracle_asr,
        "effective_oracle_success_count": n_effective_success,
        "added_oracle_over_noop_pp": float(added_oracle_pp),
    }

    per_action_rows = []
    for action_idx, action_name in enumerate(ACTION_NAMES):
        preds = action_preds[action_name]
        raw_asr = float(np.mean(preds == 0)) if n_hate else 0.0
        if action_idx == 0:
            effective_asr = 0.0
            effective_count = 0
        else:
            action_effective_success = base_caught & (preds == 0)
            effective_count = int(action_effective_success.sum())
            effective_asr = (
                float(effective_count / n_base_caught) if n_base_caught else 0.0
            )
        per_action_rows.append({
            "guardian": name,
            "action": action_name,
            "raw_asr": raw_asr,
            "raw_success_count": int(np.sum(preds == 0)),
            "effective_asr_conditioned_on_noop_caught": effective_asr,
            "effective_success_count": effective_count,
        })

    print(
        f"Clean macro-F1={clean_macro_f1:.4f}, hate-F1={clean_hate_f1:.4f}\n"
        f"Noop/raw clean-miss ASR={raw_noop_asr:.4f} "
        f"({n_base_missed}/{n_hate})\n"
        f"Raw Oracle ASR={raw_oracle_asr:.4f} "
        f"({n_raw_oracle_success}/{n_hate})\n"
        f"Added Oracle over noop={added_oracle_pp:.4f} percentage points\n"
        f"Effective Oracle ASR={effective_oracle_asr:.4f} "
        f"({n_effective_success}/{n_base_caught} originally caught hate texts)"
    )

    return summary, per_action_rows


def save_bar_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    labels = summary_df["guardian"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.8), 5))
    ax.bar(x - width, summary_df["raw_noop_asr"], width, label="Noop ASR")
    ax.bar(x, summary_df["raw_oracle_asr"], width, label="Raw Oracle ASR")
    ax.bar(
        x + width,
        summary_df["effective_oracle_asr_conditioned_on_noop_caught"],
        width,
        label="Effective Oracle ASR",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("ASR")
    ax.set_ylim(0, 1.05)
    ax.set_title("Noop vs Raw Oracle vs Effective Oracle ASR")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_per_action_plot(per_action_df: pd.DataFrame, output_path: Path) -> None:
    guardians = per_action_df["guardian"].unique().tolist()
    actions = ACTION_NAMES

    fig, ax = plt.subplots(figsize=(max(11, len(actions) * 1.2), 5))
    x = np.arange(len(actions))
    width = 0.8 / max(1, len(guardians))

    for i, guardian_name in enumerate(guardians):
        subset = per_action_df[per_action_df["guardian"] == guardian_name]
        values = [
            float(subset[subset["action"] == a]["raw_asr"].iloc[0])
            for a in actions
        ]
        offset = (i - (len(guardians) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=guardian_name)

    ax.set_xticks(x)
    ax.set_xticklabels(actions, rotation=25, ha="right")
    ax.set_ylabel("Raw ASR")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-action raw ASR on hateful test examples")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw/effective Oracle ASR across Guardian checkpoints."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--guardian",
        action="append",
        required=True,
        help="Guardian checkpoint as NAME=PATH. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="outputs/baseline_oracle_guardian_comparison",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_and_split(cfg)
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label"].tolist()
    hate_test_texts = splits["test"][splits["test"]["label"] == 1]["text"].tolist()

    print(f"[INFO] Full test examples: {len(test_texts)}")
    print(f"[INFO] Hateful test examples: {len(hate_test_texts)}")
    print(f"[INFO] Actions: {ACTION_NAMES}")

    summaries = []
    per_action_rows = []
    for name, checkpoint_path in parse_guardian_specs(args.guardian):
        summary, rows = evaluate_guardian(
            name=name,
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            test_texts=test_texts,
            test_labels=test_labels,
            hate_test_texts=hate_test_texts,
            device=device,
        )
        summaries.append(summary)
        per_action_rows.extend(rows)

    summary_df = pd.DataFrame(summaries).sort_values(
        "raw_oracle_asr", ascending=True
    )
    per_action_df = pd.DataFrame(per_action_rows)

    summary_csv = output_dir / "baseline_oracle_guardian_summary.csv"
    per_action_csv = output_dir / "baseline_oracle_guardian_per_action.csv"
    metrics_json = output_dir / "baseline_oracle_guardian_metrics.json"

    summary_df.to_csv(summary_csv, index=False)
    per_action_df.to_csv(per_action_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summaries, "per_action": per_action_rows}, f, indent=2)

    save_bar_plot(summary_df, output_dir / "baseline_oracle_guardian_asr_comparison.png")
    save_per_action_plot(per_action_df, output_dir / "baseline_oracle_guardian_per_action_raw_asr.png")

    print("\n[INFO] Final summary:")
    print(summary_df.to_string(index=False))
    print(f"\n[INFO] Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
