"""
Red Queen Outer Loop: Multi-round adversarial arms race.

Round k:
  1. Train Evader_k against Guardian_k
  2. Generate stress cases using Evader_k
  3. Robust-retrain Guardian_k → Guardian_{k+1}
  4. Evaluate Guardian_{k+1} on ALL previous stress cases + clean test
  5. Also run random-baseline evaluation for comparison

The first Guardian (round 0) must already be trained.

Usage:
    python3 -m src.outer_loop \
        --config configs/default.yaml \
        --initial_guardian outputs/run_029/guardian_best \
        --num_rounds 5 \
        --output_dir outputs/arms_race
"""

import argparse
import json
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

from src.utils import load_config, set_seed, get_device, save_metrics, save_config_snapshot
from src.data_loading import load_and_split
from src.models import Guardian, EvaderNetwork
from src.evader_env import EvaderBanditEnv
from src.evader_policy import train_evader
from src.preprocess import (
    apply_perturbation, extract_text_features,
    ACTION_NAMES, NUM_ACTIONS,
)


# ── Random baseline: apply each action uniformly and measure ASR ──

def random_baseline_evaluation(guardian, hate_texts, cfg):
    """
    For each perturbation type, apply it to ALL hateful texts and measure ASR.
    This is the brute-force baseline the RL must beat.
    """
    results = {}
    for action_idx, action_name in enumerate(ACTION_NAMES):
        perturbed = [apply_perturbation(t, action_idx, cfg) for t in hate_texts]
        preds = guardian.predict(perturbed)
        asr = np.mean(np.array(preds) == 0)
        results[action_name] = {
            "asr": float(asr),
            "count": len(hate_texts),
        }
        print(f"  [Random baseline] {action_name:15s}: ASR = {asr:.4f}")

    # Also compute "best random" — for each text, success if ANY action works
    any_success = np.zeros(len(hate_texts), dtype=bool)
    for action_idx in range(NUM_ACTIONS):
        perturbed = [apply_perturbation(t, action_idx, cfg) for t in hate_texts]
        preds = guardian.predict(perturbed)
        any_success |= (np.array(preds) == 0)
    results["best_of_all"] = {
        "asr": float(any_success.mean()),
        "count": len(hate_texts),
    }
    print(f"  [Random baseline] {'best-of-all':15s}: ASR = {results['best_of_all']['asr']:.4f}")

    return results


# ── Stress test with learned policy (greedy) ─────────────────

def generate_stress_with_policy(guardian, policy, hate_texts, cfg, device):
    """Apply learned policy greedily to hateful texts. Return DataFrame."""
    ecfg = cfg["evader"]
    records = []
    batch_size = 64

    for i in range(0, len(hate_texts), batch_size):
        batch = hate_texts[i:i + batch_size]
        probs = guardian.predict_proba(batch)
        states = np.array([
            extract_text_features(t, p) for t, p in zip(batch, probs)
        ], dtype=np.float32)
        state_tensor = torch.tensor(states).to(device)

        with torch.no_grad():
            logits = policy(state_tensor)
            actions = torch.argmax(logits, dim=-1).cpu().tolist()

        for text, action, prob in zip(batch, actions, probs):
            perturbed = apply_perturbation(text, action, cfg)
            pred = guardian.predict([perturbed])[0]
            records.append({
                "original_text": text,
                "perturbed_text": perturbed,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "guardian_pred": pred,
                "attack_success": 1 if pred == 0 else 0,
            })

    return pd.DataFrame(records)


# ── Evaluate guardian on a stress DataFrame ───────────────────

def eval_on_stress_df(guardian, stress_df):
    """Evaluate guardian on stress cases. Returns dict of metrics."""
    texts = stress_df["perturbed_text"].tolist()
    true_labels = [1] * len(texts)  # all hateful
    preds = guardian.predict(texts)

    asr = np.mean(np.array(preds) == 0)
    hate_f1 = f1_score(true_labels, preds, pos_label=1, zero_division=0)

    # Per-action breakdown
    per_action = {}
    for action_name in ACTION_NAMES:
        subset = stress_df[stress_df["action_name"] == action_name]
        if len(subset) == 0:
            continue
        sub_preds = guardian.predict(subset["perturbed_text"].tolist())
        sub_asr = np.mean(np.array(sub_preds) == 0)
        per_action[action_name] = {
            "asr": float(sub_asr),
            "count": len(subset),
        }

    return {
        "overall_asr": float(asr),
        "hate_f1": float(hate_f1),
        "per_action": per_action,
        "total": len(texts),
    }


# ── Clean test evaluation ─────────────────────────────────────

def eval_clean(guardian, test_texts, test_labels):
    preds = guardian.predict(test_texts)
    macro_f1 = f1_score(test_labels, preds, average="macro")
    hate_f1 = f1_score(test_labels, preds, pos_label=1, zero_division=0)
    return {"macro_f1": float(macro_f1), "hate_f1": float(hate_f1)}


# ── Robust retrain (inline, returns guardian) ─────────────────

def robust_retrain(guardian, train_texts, train_labels, stress_df, cfg, device, save_path):
    """Retrain guardian with augmented data. Returns updated guardian."""
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    rcfg = cfg["robust"]
    gcfg = cfg["guardian"]

    # Augment
    stress_texts = stress_df["perturbed_text"].tolist()
    stress_labels = [1] * len(stress_texts)
    aug_ratio = rcfg["augmentation_ratio"]
    n_aug = int(len(stress_texts) * aug_ratio)
    if n_aug > 0:
        indices = np.random.choice(len(stress_texts), size=min(n_aug, len(stress_texts)), replace=False)
        aug_texts = [stress_texts[i] for i in indices]
        aug_labels = [stress_labels[i] for i in indices]
    else:
        aug_texts, aug_labels = stress_texts, stress_labels

    combined_texts = train_texts + aug_texts
    combined_labels = train_labels + aug_labels

    train_ds = guardian.make_dataset(combined_texts, combined_labels)
    train_loader = DataLoader(train_ds, batch_size=gcfg["batch_size"], shuffle=True)

    optimizer = AdamW(guardian.model.parameters(), lr=rcfg["learning_rate"], weight_decay=gcfg["weight_decay"])
    total_steps = len(train_loader) * rcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    for epoch in range(rcfg["epochs"]):
        guardian.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"  Robust epoch {epoch+1}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = guardian.model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(guardian.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"    Robust epoch {epoch+1} loss: {total_loss / len(train_loader):.4f}")

    guardian.save(str(save_path))
    return guardian


# ── Main outer loop ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Red Queen Outer Loop")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--initial_guardian", type=str, required=True,
                        help="Path to the initial trained Guardian checkpoint")
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/arms_race")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, out)

    # ── Load data ────────────────────────────────────────────
    splits = load_and_split(cfg)
    train_texts = splits["train"]["text"].tolist()
    train_labels = splits["train"]["label"].tolist()
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label"].tolist()

    hate_train = splits["train"][splits["train"]["label"] == 1]["text"].tolist()
    hate_test = splits["test"][splits["test"]["label"] == 1]["text"].tolist()
    print(f"[INFO] Hate train: {len(hate_train)}, Hate test: {len(hate_test)}")

    # ── Load initial Guardian ────────────────────────────────
    guardian = Guardian(cfg, device)
    guardian.load(args.initial_guardian)

    # Track results across rounds
    round_results = []
    all_stress_dfs = []  # cumulative stress cases

    for rnd in range(args.num_rounds):
        print(f"\n{'='*60}")
        print(f"  ROUND {rnd + 1} / {args.num_rounds}")
        print(f"{'='*60}")

        round_dir = out / f"round_{rnd + 1:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # ── 0. Evaluate current Guardian on clean test ───────
        print(f"\n[Round {rnd+1}] Evaluating Guardian on clean test...")
        clean_metrics = eval_clean(guardian, test_texts, test_labels)
        print(f"  Clean macro-F1: {clean_metrics['macro_f1']:.4f}, hate-F1: {clean_metrics['hate_f1']:.4f}")

        # ── 1. Random baseline (brute-force per-action ASR) ──
        print(f"\n[Round {rnd+1}] Running random baseline on test hate texts...")
        random_bl = random_baseline_evaluation(guardian, hate_test, cfg)

        # ── 2. Train Evader against current Guardian ─────────
        print(f"\n[Round {rnd+1}] Training Evader...")
        temp_guardian_path = round_dir / "guardian_input"
        guardian.save(str(temp_guardian_path))
        train_evader(cfg, str(temp_guardian_path), round_dir)

        # Load the policy back
        ecfg = cfg["evader"]
        policy_net = EvaderNetwork(ecfg["state_dim"], NUM_ACTIONS, ecfg["hidden_dim"]).to(device)
        policy_path = round_dir / "evader_policy.pt"
        policy_net.load_state_dict(torch.load(policy_path, map_location=device))
        policy_net.eval()

        # ── 3. Generate stress cases ─────────────────────────
        print(f"\n[Round {rnd+1}] Generating stress cases...")
        stress_df = generate_stress_with_policy(
            guardian, policy_net, hate_train, cfg, device
        )
        stress_df.to_parquet(round_dir / "stress_cases.parquet", index=False)

        # Log which actions the Evader chose
        action_dist = stress_df["action_name"].value_counts(normalize=True).to_dict()
        print(f"  Action distribution: {action_dist}")
        asr_this_round = stress_df["attack_success"].mean()
        print(f"  ASR this round: {asr_this_round:.4f}")

        # Per-action ASR from learned policy
        learned_per_action = {}
        for aname in ACTION_NAMES:
            sub = stress_df[stress_df["action_name"] == aname]
            if len(sub) > 0:
                learned_per_action[aname] = {
                    "asr": float(sub["attack_success"].mean()),
                    "count": len(sub),
                    "fraction_chosen": float(len(sub) / len(stress_df)),
                }

        # ── 4. Evaluate on ALL cumulative stress cases ───────
        all_stress_dfs.append(stress_df)
        cumulative_stress = pd.concat(all_stress_dfs, ignore_index=True)

        # Evaluate current guardian on cumulative stress (before robust retrain)
        pre_robust_cumulative = eval_on_stress_df(guardian, cumulative_stress)
        print(f"  Pre-robust cumulative ASR: {pre_robust_cumulative['overall_asr']:.4f}")

        # ── 5. Robust retrain ────────────────────────────────
        print(f"\n[Round {rnd+1}] Robust retraining...")
        # Use ALL cumulative stress cases for retraining
        guardian = robust_retrain(
            guardian, train_texts, train_labels,
            cumulative_stress, cfg, device,
            save_path=round_dir / "guardian_robust",
        )

        # ── 6. Post-retrain evaluation ───────────────────────
        print(f"\n[Round {rnd+1}] Post-retrain evaluation...")
        post_clean = eval_clean(guardian, test_texts, test_labels)
        print(f"  Post-robust clean macro-F1: {post_clean['macro_f1']:.4f}")

        post_cumulative = eval_on_stress_df(guardian, cumulative_stress)
        print(f"  Post-robust cumulative ASR: {post_cumulative['overall_asr']:.4f}")

        # Per-action ASR after robust retrain (using random application)
        print(f"\n[Round {rnd+1}] Post-retrain per-action ASR (random baseline)...")
        post_random_bl = random_baseline_evaluation(guardian, hate_test, cfg)

        # ── Record round results ─────────────────────────────
        round_data = {
            "round": rnd + 1,
            "pre_clean_macro_f1": clean_metrics["macro_f1"],
            "pre_clean_hate_f1": clean_metrics["hate_f1"],
            "evader_asr": float(asr_this_round),
            "evader_action_distribution": action_dist,
            "evader_per_action": learned_per_action,
            "random_baseline_pre": random_bl,
            "pre_robust_cumulative_asr": pre_robust_cumulative["overall_asr"],
            "post_clean_macro_f1": post_clean["macro_f1"],
            "post_clean_hate_f1": post_clean["hate_f1"],
            "post_robust_cumulative_asr": post_cumulative["overall_asr"],
            "random_baseline_post": post_random_bl,
        }
        round_results.append(round_data)

        # Save per-round metrics
        save_metrics(round_data, round_dir / "round_metrics.json")

    # ── Save global results ──────────────────────────────────
    save_metrics(round_results, out / "all_rounds.json")

    # ── Build summary table ──────────────────────────────────
    summary_rows = []
    for r in round_results:
        # Find dominant action
        dist = r["evader_action_distribution"]
        dominant = max(dist, key=dist.get) if dist else "none"

        summary_rows.append({
            "Round": r["round"],
            "Clean Macro-F1 (pre)": f"{r['pre_clean_macro_f1']:.4f}",
            "Clean Macro-F1 (post)": f"{r['post_clean_macro_f1']:.4f}",
            "Evader ASR": f"{r['evader_asr']:.4f}",
            "Dominant Action": dominant,
            "Cumul ASR (pre-robust)": f"{r['pre_robust_cumulative_asr']:.4f}",
            "Cumul ASR (post-robust)": f"{r['post_robust_cumulative_asr']:.4f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "summary_table.csv", index=False)
    print(f"\n{'='*60}")
    print("ARMS RACE SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    # ── Plot 1: ASR across rounds ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    rounds = [r["round"] for r in round_results]

    # Evader ASR per round
    axes[0].plot(rounds, [r["evader_asr"] for r in round_results], "o-", color="#c44e52", linewidth=2)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Attack Success Rate")
    axes[0].set_title("Evader ASR per Round")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.3)

    # Clean performance across rounds
    axes[1].plot(rounds, [r["pre_clean_macro_f1"] for r in round_results], "s-",
                 label="Pre-robust", color="#4c72b0", linewidth=2)
    axes[1].plot(rounds, [r["post_clean_macro_f1"] for r in round_results], "^-",
                 label="Post-robust", color="#dd8452", linewidth=2)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Clean Test Performance")
    axes[1].legend()
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(alpha=0.3)

    # Cumulative ASR pre vs post
    axes[2].plot(rounds, [r["pre_robust_cumulative_asr"] for r in round_results], "o-",
                 label="Pre-robust", color="#c44e52", linewidth=2)
    axes[2].plot(rounds, [r["post_robust_cumulative_asr"] for r in round_results], "^-",
                 label="Post-robust", color="#55a868", linewidth=2)
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Cumulative ASR")
    axes[2].set_title("Cumulative Stress Test ASR")
    axes[2].legend()
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "arms_race_dynamics.png", dpi=150)
    plt.close()

    # ── Plot 2: Per-action ASR heatmap across rounds ─────────
    # Build matrix: rows = actions, cols = rounds, values = ASR (from random baseline post-retrain)
    action_names = ACTION_NAMES
    heatmap_data = np.zeros((len(action_names), len(round_results)))

    for j, r in enumerate(round_results):
        for i, aname in enumerate(action_names):
            bl = r.get("random_baseline_post", {}).get(aname, {})
            heatmap_data[i, j] = bl.get("asr", 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(round_results) * 1.5), 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn_r",
                xticklabels=[f"Round {r['round']}" for r in round_results],
                yticklabels=action_names, ax=ax, vmin=0, vmax=1)
    ax.set_title("Per-Action ASR After Robust Retraining (Random Baseline)")
    ax.set_ylabel("Perturbation Type")
    ax.set_xlabel("Round")
    plt.tight_layout()
    plt.savefig(out / "per_action_heatmap.png", dpi=150)
    plt.close()

    # ── Plot 3: Dominant action per round ────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(round_results) * 1.5), 5))
    for aname in action_names:
        fracs = []
        for r in round_results:
            fracs.append(r["evader_action_distribution"].get(aname, 0.0))
        if max(fracs) > 0.01:  # only plot if ever chosen
            ax.plot(rounds, fracs, "o-", label=aname, linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction Chosen by Evader")
    ax.set_title("Evader Action Selection Across Rounds")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "action_selection_across_rounds.png", dpi=150)
    plt.close()

    # ── Plot 4: Random baseline ASR comparison (round 1 vs final round) ──
    if len(round_results) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(action_names))
        width = 0.35

        first_asrs = [round_results[0]["random_baseline_pre"].get(a, {}).get("asr", 0) for a in action_names]
        final_asrs = [round_results[-1]["random_baseline_post"].get(a, {}).get("asr", 0) for a in action_names]

        ax.bar(x - width / 2, first_asrs, width, label=f"Guardian v1 (before any robustness)", color="#c44e52")
        ax.bar(x + width / 2, final_asrs, width, label=f"Guardian v{len(round_results)+1} (after {len(round_results)} rounds)", color="#55a868")
        ax.set_xticks(x)
        ax.set_xticklabels(action_names, rotation=30, ha="right")
        ax.set_ylabel("Attack Success Rate")
        ax.set_title("Per-Action Vulnerability: Initial vs Final Guardian")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out / "initial_vs_final_vulnerability.png", dpi=150)
        plt.close()

    print(f"\n[INFO] All results saved to {out}")
    print(f"[INFO] Key plots:")
    print(f"  {out / 'arms_race_dynamics.png'}")
    print(f"  {out / 'per_action_heatmap.png'}")
    print(f"  {out / 'action_selection_across_rounds.png'}")
    print(f"  {out / 'initial_vs_final_vulnerability.png'}")


if __name__ == "__main__":
    main()
