"""
Generate stress-test cases using the trained Evader policy, random baseline, or oracle.

Usage:
    # With trained RL evader:
    python -m src.generate_stress_cases --config configs/default.yaml \
        --guardian_path outputs/run_001/guardian_best \
        --evader_path outputs/run_002/evader_policy.pt

    # With random baseline:
    python -m src.generate_stress_cases --config configs/default.yaml \
        --guardian_path outputs/run_001/guardian_best --random

    # With oracle (upper bound):
    python -m src.generate_stress_cases --config configs/default.yaml \
        --guardian_path outputs/run_001/guardian_best --oracle
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils import (
    load_config, set_seed, get_device, save_metrics,
    safe_log_example, sanitize_text,
)
from src.data_loading import load_and_split
from src.models import Guardian, EvaderNetwork, RandomNetwork
from src.preprocess import (
    apply_perturbation, extract_text_features,
    ACTION_NAMES, NUM_ACTIONS,
)


def oracle_actions(texts, guardian, cfg, n_samples=30):
    """
    For each text, try each perturbation n_samples times and pick the action
    whose AVERAGE hate prob is lowest. Robust to stochastic perturbations.
    """
    avg_probs = np.zeros((len(texts), NUM_ACTIONS))

    for a in range(NUM_ACTIONS):
        action_probs = np.zeros((n_samples, len(texts)))
        for s in range(n_samples):
            perturbed = [apply_perturbation(t, a, cfg) for t in texts]
            action_probs[s] = guardian.predict_proba(perturbed)
        avg_probs[:, a] = action_probs.mean(axis=0)

    best_actions = np.argmin(avg_probs, axis=1).tolist()
    return best_actions


def oracle_apply_with_retries(text, action, guardian, cfg, n_retries=20):
    """
    For oracle: try multiple stochastic samples of the chosen action
    and keep the one that successfully fools the Guardian (or the best one).

    Returns: (perturbed_text, final_pred, hate_prob)
    """
    best_perturbed = apply_perturbation(text, action, cfg)
    best_pred = guardian.predict([best_perturbed])[0]
    best_prob = guardian.predict_proba([best_perturbed])[0]

    if best_pred == 0:
        return best_perturbed, best_pred, best_prob

    for _ in range(n_retries):
        candidate = apply_perturbation(text, action, cfg)
        cand_prob = guardian.predict_proba([candidate])[0]
        if cand_prob < best_prob:
            best_perturbed = candidate
            best_prob = cand_prob
            if cand_prob < 0.5:  # Attack succeeded
                best_pred = 0
                return best_perturbed, best_pred, best_prob

    best_pred = 1 if best_prob >= 0.5 else 0
    return best_perturbed, best_pred, best_prob


def generate(cfg, guardian_path, evader_path, output_dir):
    device = get_device()
    ecfg = cfg["evader"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────
    guardian = Guardian(cfg, device)
    guardian.load(guardian_path)

    mode = "rl"
    policy = None
    if cfg.get("random_evader", False):
        mode = "random"
        print("[INFO] Using Random Evader baseline")
    elif cfg.get("oracle_evader", False):
        mode = "oracle"
        print("[INFO] Using Oracle Evader (upper bound, with retries)")
    else:
        policy = EvaderNetwork(NUM_ACTIONS).to(device)
        policy.load_state_dict(torch.load(evader_path, map_location=device))
        policy.eval()
        print("[INFO] Using learned EvaderNetwork")

    # ── Load hateful examples from train set ─────────────────
    splits = load_and_split(cfg)
    hate_texts = splits["train"][splits["train"]["label"] == 1]["text"].tolist()
    print(f"[INFO] Generating stress cases from {len(hate_texts)} hateful examples")

    # ── Generate stress cases ────────────────────────────────
    records = []
    batch_size = 64

    for i in tqdm(range(0, len(hate_texts), batch_size), desc="Generating"):
        batch = hate_texts[i:i + batch_size]

        # Get Guardian probabilities for features
        probs = guardian.predict_proba(batch)

        # Get actions based on mode
        if mode == "random":
            actions = torch.randint(0, NUM_ACTIONS, (len(batch),)).tolist()
        elif mode == "oracle":
            actions = oracle_actions(batch, guardian, cfg)
        else:  # rl
            states = np.array([
                extract_text_features(t, p) for t, p in zip(batch, probs)
            ], dtype=np.float32)
            state_tensor = torch.tensor(states).to(device)
            with torch.no_grad():
                logits = policy(state_tensor)
                actions = torch.argmax(logits, dim=-1).cpu().tolist()

        # Apply perturbations
        for text, action, prob in zip(batch, actions, probs):
            if mode == "oracle":
                # Oracle: retry stochastic perturbations to find the best one
                perturbed, pred, _ = oracle_apply_with_retries(
                    text, action, guardian, cfg, n_retries=20
                )
            else:
                perturbed = apply_perturbation(text, action, cfg)
                pred = guardian.predict([perturbed])[0]

            records.append({
                "original_text": text,
                "perturbed_text": perturbed,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "original_label": 1,
                "guardian_pred_original": 1 if prob >= 0.5 else 0,
                "guardian_pred_perturbed": pred,
                "attack_success": 1 if pred == 0 else 0,
                "guardian_hate_prob": prob,
            })

    df = pd.DataFrame(records)

    # ── Summary stats ────────────────────────────────────────
    asr = df["attack_success"].mean()
    asr_by_action = df.groupby("action_name")["attack_success"].mean().to_dict()
    action_counts = df["action_name"].value_counts(normalize=True).to_dict()

    print(f"\n[RESULTS] Overall Attack Success Rate: {asr:.4f}")
    print(f"[RESULTS] ASR by action:")
    for name, rate in sorted(asr_by_action.items(), key=lambda x: -x[1]):
        print(f"  {name:15s}: {rate:.4f}")

    # ── Save ─────────────────────────────────────────────────
    sanitize = cfg["eval"].get("sanitize_logs", True)
    if sanitize:
        df["original_text_display"] = df["original_text"].apply(sanitize_text)
        df["perturbed_text_display"] = df["perturbed_text"].apply(sanitize_text)

    df.to_parquet(output_dir / "stress_cases.parquet", index=False)

    sample = df.head(100)
    if sanitize:
        cols = ["original_text_display", "perturbed_text_display",
                "action_name", "attack_success", "guardian_hate_prob"]
    else:
        cols = ["original_text", "perturbed_text",
                "action_name", "attack_success", "guardian_hate_prob"]
    sample[[c for c in cols if c in sample.columns]].to_csv(
        output_dir / "stress_cases_sample.csv", index=False
    )

    save_metrics({
        "overall_asr": asr,
        "asr_by_action": asr_by_action,
        "action_distribution": action_counts,
        "total_cases": len(df),
        "successful_attacks": int(df["attack_success"].sum()),
    }, output_dir / "stress_case_metrics.json")

    print(f"[INFO] Stress cases saved to {output_dir}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate stress-test cases")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--guardian_path", type=str, required=True)
    parser.add_argument("--evader_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/stress_cases")
    parser.add_argument("--random", action="store_true",
                        help="Use random baseline")
    parser.add_argument("--oracle", action="store_true",
                        help="Use oracle (upper bound: try all perturbations, pick best)")
    args = parser.parse_args()

    if args.random and args.oracle:
        raise ValueError("Cannot use --random and --oracle at the same time")

    cfg = load_config(args.config)
    cfg["random_evader"] = args.random
    cfg["oracle_evader"] = args.oracle
    set_seed(cfg["seed"])
    generate(cfg, args.guardian_path, args.evader_path, args.output_dir)


if __name__ == "__main__":
    main()
