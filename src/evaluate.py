"""
Full evaluation pipeline: compare baseline vs robust Guardian on clean
and stressed test sets. Produces metrics, confusion matrices, plots.

Usage:
    python -m src.evaluate \
        --config configs/default.yaml \
        --baseline_path outputs/run_001/guardian_best \
        --robust_path outputs/run_003/guardian_robust \
        --stress_cases outputs/stress_cases/stress_cases.parquet \
        --output_dir outputs/evaluation
"""

import argparse
import json
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from src.utils import load_config, set_seed, get_device, save_metrics
from src.data_loading import load_and_split
from src.models import Guardian
from src.preprocess import apply_perturbation, ACTION_NAMES, NUM_ACTIONS


def eval_on_texts(guardian, texts, labels, tag=""):
    """Evaluate Guardian on a set of texts. Return metrics dict."""
    preds = guardian.predict(texts)
    report = classification_report(labels, preds, target_names=["safe", "hate"], output_dict=True)
    cm = confusion_matrix(labels, preds)
    return {
        "tag": tag,
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "hate_precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "hate_recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "hate_f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "accuracy": np.mean(np.array(labels) == np.array(preds)),
        "confusion_matrix": cm.tolist(),
        "report": report,
        "preds": preds,
    }


def eval_stress_by_action(guardian, stress_df):
    """Evaluate Guardian on stress cases, broken down by perturbation type."""
    results = {}
    for action_name in ACTION_NAMES:
        subset = stress_df[stress_df["action_name"] == action_name]
        if len(subset) == 0:
            continue
        preds = guardian.predict(subset["perturbed_text"].tolist())
        true_labels = [1] * len(subset)  # all hateful
        asr = np.mean(np.array(preds) == 0)  # fraction misclassified as safe
        f1 = f1_score(true_labels, preds, pos_label=1, zero_division=0)
        results[action_name] = {
            "count": len(subset),
            "attack_success_rate": float(asr),
            "hate_f1": float(f1),
        }
    return results


def eval_fairness(guardian, test_df, identity_cols=None):
    """
    Evaluate per-subgroup performance if identity columns exist.
    For Civil Comments, identity columns are float [0, 1]; threshold at 0.5.
    """
    if identity_cols is None:
        identity_cols = [
            "male", "female", "black", "white", "asian", "latino",
            "homosexual_gay_or_lesbian", "muslim", "christian", "jewish",
        ]
    available = [c for c in identity_cols if c in test_df.columns]
    if not available:
        return {}

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()
    preds = guardian.predict(texts)

    results = {}
    for col in available:
        mask = test_df[col].fillna(0) >= 0.5
        sub_labels = [labels[i] for i in range(len(labels)) if mask.iloc[i]]
        sub_preds = [preds[i] for i in range(len(preds)) if mask.iloc[i]]
        if len(sub_labels) < 10:
            continue
        results[col] = {
            "count": len(sub_labels),
            "macro_f1": f1_score(sub_labels, sub_preds, average="macro", zero_division=0),
            "hate_f1": f1_score(sub_labels, sub_preds, pos_label=1, zero_division=0),
        }
    return results


def plot_confusion_matrix(cm, title, path):
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["safe", "hate"],
                yticklabels=["safe", "hate"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_clean_vs_stress(baseline_clean, baseline_stress, robust_clean, robust_stress, path):
    """Bar chart comparing clean vs stressed performance for baseline and robust."""
    metrics = ["macro_f1", "hate_f1", "hate_precision", "hate_recall"]
    labels = ["Macro-F1", "Hate-F1", "Hate-Prec", "Hate-Rec"]

    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    vals_bc = [baseline_clean.get(m, 0) for m in metrics]
    vals_bs = [baseline_stress.get(m, 0) for m in metrics]
    vals_rc = [robust_clean.get(m, 0) for m in metrics]
    vals_rs = [robust_stress.get(m, 0) for m in metrics]

    ax.bar(x - 1.5 * width, vals_bc, width, label="Baseline (clean)", color="#4c72b0")
    ax.bar(x - 0.5 * width, vals_bs, width, label="Baseline (stressed)", color="#4c72b0", alpha=0.5)
    ax.bar(x + 0.5 * width, vals_rc, width, label="Robust (clean)", color="#dd8452")
    ax.bar(x + 1.5 * width, vals_rs, width, label="Robust (stressed)", color="#dd8452", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Guardian Performance: Clean vs Stressed")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_asr_by_action(action_results_baseline, action_results_robust, path):
    """Bar chart of ASR by perturbation type, baseline vs robust."""
    actions = sorted(set(list(action_results_baseline.keys()) +
                        list(action_results_robust.keys())))
    if not actions:
        return

    x = np.arange(len(actions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    baseline_asrs = [action_results_baseline.get(a, {}).get("attack_success_rate", 0) for a in actions]
    robust_asrs = [action_results_robust.get(a, {}).get("attack_success_rate", 0) for a in actions]

    ax.bar(x - width / 2, baseline_asrs, width, label="Baseline", color="#4c72b0")
    ax.bar(x + width / 2, robust_asrs, width, label="Robust", color="#dd8452")

    ax.set_xticks(x)
    ax.set_xticklabels(actions, rotation=30, ha="right")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("ASR by Perturbation Type")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_fairness(fairness_baseline, fairness_robust, path):
    """Grouped bar chart of per-subgroup F1."""
    groups = sorted(set(list(fairness_baseline.keys()) +
                       list(fairness_robust.keys())))
    if not groups:
        return

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    b_vals = [fairness_baseline.get(g, {}).get("hate_f1", 0) for g in groups]
    r_vals = [fairness_robust.get(g, {}).get("hate_f1", 0) for g in groups]

    ax.bar(x - width / 2, b_vals, width, label="Baseline", color="#4c72b0")
    ax.bar(x + width / 2, r_vals, width, label="Robust", color="#dd8452")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel("Hate-class F1")
    ax.set_title("Subgroup Fairness: Hate-F1 by Identity Group")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Full evaluation pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--baseline_path", type=str, required=True)
    parser.add_argument("--robust_path", type=str, default=None,
                        help="Path to robust Guardian (optional)")
    parser.add_argument("--stress_cases", type=str, default=None,
                        help="Path to stress_cases.parquet (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    splits = load_and_split(cfg)
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label"].tolist()

    # ── Evaluate baseline on clean test ──────────────────────
    print("\n=== Baseline Guardian on Clean Test ===")
    baseline = Guardian(cfg, device)
    baseline.load(args.baseline_path)
    bl_clean = eval_on_texts(baseline, test_texts, test_labels, "baseline_clean")
    print(f"  Macro-F1:  {bl_clean['macro_f1']:.4f}")
    print(f"  Hate-F1:   {bl_clean['hate_f1']:.4f}")
    print(f"  Hate-Prec: {bl_clean['hate_precision']:.4f}")
    print(f"  Hate-Rec:  {bl_clean['hate_recall']:.4f}")

    plot_confusion_matrix(
        np.array(bl_clean["confusion_matrix"]),
        "Baseline — Clean Test", out / "cm_baseline_clean.png",
    )

    # ── Evaluate on stress cases ─────────────────────────────
    bl_stress = None
    action_bl = {}
    stress_df = None

    if args.stress_cases and Path(args.stress_cases).exists():
        stress_df = pd.read_parquet(args.stress_cases)
        stress_texts = stress_df["perturbed_text"].tolist()
        stress_labels = [1] * len(stress_texts)

        print(f"\n=== Baseline Guardian on {len(stress_texts)} Stress Cases ===")
        bl_stress = eval_on_texts(baseline, stress_texts, stress_labels, "baseline_stress")
        print(f"  Macro-F1:  {bl_stress['macro_f1']:.4f}")
        print(f"  Hate-F1:   {bl_stress['hate_f1']:.4f}")
        print(f"  ASR:       {1 - bl_stress['hate_recall']:.4f}")

        action_bl = eval_stress_by_action(baseline, stress_df)
        print(f"  ASR by action:")
        for name, data in sorted(action_bl.items(), key=lambda x: -x[1]["attack_success_rate"]):
            print(f"    {name:15s}: ASR={data['attack_success_rate']:.3f} (n={data['count']})")

    # ── Fairness (baseline) ──────────────────────────────────
    fairness_bl = eval_fairness(baseline, splits["test"])
    if fairness_bl:
        print(f"\n=== Baseline Subgroup Fairness ===")
        for g, m in sorted(fairness_bl.items()):
            print(f"  {g:30s}: F1={m['hate_f1']:.3f} (n={m['count']})")

    # ── Evaluate robust Guardian ─────────────────────────────
    rb_clean = None
    rb_stress = None
    action_rb = {}
    fairness_rb = {}

    if args.robust_path and Path(args.robust_path).exists():
        robust = Guardian(cfg, device)
        robust.load(args.robust_path)

        print(f"\n=== Robust Guardian on Clean Test ===")
        rb_clean = eval_on_texts(robust, test_texts, test_labels, "robust_clean")
        print(f"  Macro-F1:  {rb_clean['macro_f1']:.4f}")
        print(f"  Hate-F1:   {rb_clean['hate_f1']:.4f}")

        plot_confusion_matrix(
            np.array(rb_clean["confusion_matrix"]),
            "Robust — Clean Test", out / "cm_robust_clean.png",
        )

        if stress_df is not None:
            stress_texts = stress_df["perturbed_text"].tolist()
            stress_labels = [1] * len(stress_texts)
            print(f"\n=== Robust Guardian on Stress Cases ===")
            rb_stress = eval_on_texts(robust, stress_texts, stress_labels, "robust_stress")
            print(f"  Macro-F1:  {rb_stress['macro_f1']:.4f}")
            print(f"  Hate-F1:   {rb_stress['hate_f1']:.4f}")
            print(f"  ASR:       {1 - rb_stress['hate_recall']:.4f}")

            action_rb = eval_stress_by_action(robust, stress_df)

            plot_confusion_matrix(
                np.array(rb_stress["confusion_matrix"]),
                "Robust — Stress Test", out / "cm_robust_stress.png",
            )

        fairness_rb = eval_fairness(robust, splits["test"])

    # ── Plots ────────────────────────────────────────────────
    if bl_stress and rb_clean and rb_stress:
        plot_clean_vs_stress(bl_clean, bl_stress, rb_clean, rb_stress,
                             out / "clean_vs_stress_comparison.png")

    if action_bl:
        plot_asr_by_action(action_bl, action_rb, out / "asr_by_action.png")

    if fairness_bl:
        plot_fairness(fairness_bl, fairness_rb, out / "fairness_subgroups.png")

    # ── Save all metrics ─────────────────────────────────────
    def _strip_preds(d):
        if d is None:
            return None
        return {k: v for k, v in d.items() if k != "preds"}

    all_metrics = {
        "baseline_clean": _strip_preds(bl_clean),
        "baseline_stress": _strip_preds(bl_stress),
        "robust_clean": _strip_preds(rb_clean),
        "robust_stress": _strip_preds(rb_stress),
        "asr_by_action_baseline": action_bl,
        "asr_by_action_robust": action_rb,
        "fairness_baseline": fairness_bl,
        "fairness_robust": fairness_rb,
    }
    save_metrics(all_metrics, out / "all_metrics.json")

    # ── Summary table ────────────────────────────────────────
    rows = []
    for name, m in [("Baseline Clean", bl_clean), ("Baseline Stress", bl_stress),
                    ("Robust Clean", rb_clean), ("Robust Stress", rb_stress)]:
        if m:
            rows.append({
                "Condition": name,
                "Macro-F1": f"{m['macro_f1']:.4f}",
                "Hate-F1": f"{m['hate_f1']:.4f}",
                "Hate-Prec": f"{m['hate_precision']:.4f}",
                "Hate-Rec": f"{m['hate_recall']:.4f}",
                "Accuracy": f"{m['accuracy']:.4f}",
            })
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(out / "summary_table.csv", index=False)
        print(f"\n{'='*60}")
        print(summary.to_string(index=False))
        print(f"{'='*60}")

    print(f"\n[INFO] All evaluation outputs saved to {out}")


if __name__ == "__main__":
    main()
