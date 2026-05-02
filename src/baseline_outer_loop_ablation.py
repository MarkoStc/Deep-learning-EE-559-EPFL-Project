"""
Baseline outer-loop ablation for Red Queen Moderation.

This script runs the same 5-round adversarial-training loop as src.outer_loop,
but replaces the RL Evader with a chosen stress-case generator:

  random    - one random perturbation action per hateful training text
  greedy    - pick the best single action on validation hateful texts each round,
              then use that action for every hateful training text
  oracle    - try all actions per training text and keep a successful one; if none
              succeeds, keep the action that minimizes Guardian P(hate)
  heuristic - hand-written action rules using the same feature vector as the Evader
  rl        - original RL Evader, trained from scratch each round

The goal is not only to measure which attacker fools the current Guardian. The goal
is to answer: which stress-case generator makes the final Guardian most robust after
5 rounds of adversarial training?

Example:
  python3 -m src.baseline_outer_loop_ablation \
    --config configs/default.yaml \
    --strategy greedy \
    --initial_guardian outputs/run_029/guardian_best \
    --num_rounds 5 \
    --output_dir outputs/baseline_outer_loop_greedy
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.utils import load_config, set_seed, get_device, save_metrics, save_config_snapshot
from src.data_loading import load_and_split
from src.models import Guardian, EvaderNetwork
from src.evader_policy import train_evader
from src.preprocess import apply_perturbation, extract_text_features, ACTION_NAMES, NUM_ACTIONS


def eval_clean(guardian: Guardian, test_texts: List[str], test_labels: List[int]) -> Dict[str, float]:
    preds = guardian.predict(test_texts)
    return {
        "macro_f1": float(f1_score(test_labels, preds, average="macro", zero_division=0)),
        "hate_f1": float(f1_score(test_labels, preds, pos_label=1, zero_division=0)),
    }


def evaluate_actions_on_hate_texts(guardian: Guardian, hate_texts: List[str], cfg: dict) -> Dict[str, dict]:
    """Evaluate every action on the same hateful texts. Also returns oracle/best-of-all ASR."""
    results: Dict[str, dict] = {}
    any_success = np.zeros(len(hate_texts), dtype=bool)

    for action_idx, action_name in enumerate(ACTION_NAMES):
        perturbed = [apply_perturbation(t, action_idx, cfg) for t in hate_texts]
        preds = np.array(guardian.predict(perturbed))
        probs = np.array(guardian.predict_proba(perturbed))
        success = preds == 0
        any_success |= success
        results[action_name] = {
            "asr": float(success.mean()) if len(success) else 0.0,
            "mean_hate_probability": float(probs.mean()) if len(probs) else 0.0,
            "count": int(len(hate_texts)),
        }

    results["oracle_best_of_all"] = {
        "asr": float(any_success.mean()) if len(any_success) else 0.0,
        "count": int(len(hate_texts)),
    }
    return results


def best_single_action_on_validation(guardian: Guardian, val_hate_texts: List[str], cfg: dict) -> Tuple[int, Dict[str, dict]]:
    action_results = evaluate_actions_on_hate_texts(guardian, val_hate_texts, cfg)
    best_idx = max(range(NUM_ACTIONS), key=lambda i: action_results[ACTION_NAMES[i]]["asr"])
    return best_idx, action_results


def heuristic_action(text: str, guardian_prob: float) -> int:
    """
    Simple hand-written baseline using the same 8 features as the RL Evader.

    The rules are intentionally simple: they are not supposed to be perfect. They test
    whether human intuition can replace a learned policy.
    """
    features = extract_text_features(text, guardian_prob)
    char_len = features[0]
    word_count = features[1]
    upper_frac = features[2]
    punct_frac = features[3]
    has_emoji = features[4]
    has_url = features[5]
    avg_word_len = features[6]
    hate_prob = features[7]

    # Very confident Guardian: use the historically strong surface mutation.
    if hate_prob >= 0.80:
        return ACTION_NAMES.index("leetspeak")
    # URL/emoji-heavy texts often already contain weird tokens; add emoji noise.
    if has_url > 0.5 or has_emoji > 0.5:
        return ACTION_NAMES.index("emoji")
    # Long comments: typos spread across more words and are cheap/diverse.
    if char_len > 0.60 or word_count > 0.55:
        return ACTION_NAMES.index("typo")
    # Many long words: synonym swap has a better chance to do something.
    if avg_word_len > 0.45:
        return ACTION_NAMES.index("synonym")
    # Punctuation/uppercase-heavy text: homoglyphs add another tokenization shift.
    if punct_frac > 0.08 or upper_frac > 0.10:
        return ACTION_NAMES.index("homoglyph")
    # Default: leetspeak because it is usually the strongest single action.
    return ACTION_NAMES.index("leetspeak")


def build_records_for_actions(
    guardian: Guardian,
    texts: List[str],
    actions: List[int],
    cfg: dict,
    strategy: str,
) -> pd.DataFrame:
    perturbed = [apply_perturbation(t, a, cfg) for t, a in zip(texts, actions)]
    preds = guardian.predict(perturbed)
    probs = guardian.predict_proba(perturbed)
    return pd.DataFrame({
        "original_text": texts,
        "perturbed_text": perturbed,
        "action": actions,
        "action_name": [ACTION_NAMES[a] for a in actions],
        "guardian_pred": preds,
        "guardian_hate_probability": probs,
        "attack_success": [1 if p == 0 else 0 for p in preds],
        "strategy": strategy,
    })


def generate_random_stress(guardian: Guardian, hate_train: List[str], cfg: dict) -> Tuple[pd.DataFrame, Dict[str, object]]:
    actions = [random.randrange(NUM_ACTIONS) for _ in hate_train]
    df = build_records_for_actions(guardian, hate_train, actions, cfg, "random")
    return df, {"chosen_action": "random_uniform"}


def generate_greedy_stress(
    guardian: Guardian,
    hate_train: List[str],
    val_hate: List[str],
    cfg: dict,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    best_idx, val_action_results = best_single_action_on_validation(guardian, val_hate, cfg)
    actions = [best_idx] * len(hate_train)
    df = build_records_for_actions(guardian, hate_train, actions, cfg, "greedy")
    info = {
        "chosen_action": ACTION_NAMES[best_idx],
        "chosen_action_idx": int(best_idx),
        "validation_action_results": val_action_results,
    }
    return df, info


def generate_heuristic_stress(guardian: Guardian, hate_train: List[str], cfg: dict) -> Tuple[pd.DataFrame, Dict[str, object]]:
    probs = guardian.predict_proba(hate_train)
    actions = [heuristic_action(t, p) for t, p in zip(hate_train, probs)]
    df = build_records_for_actions(guardian, hate_train, actions, cfg, "heuristic")
    return df, {"chosen_action": "feature_rules"}


def generate_oracle_stress(guardian: Guardian, hate_train: List[str], cfg: dict) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Try all actions per text. Keep the first successful action. If no action fools the
    Guardian, keep the action with the lowest P(hate). This makes training examples as
    strong as possible under the fixed 7-action set.
    """
    all_preds = []
    all_probs = []
    all_perturbed = []

    for action_idx in range(NUM_ACTIONS):
        perturbed = [apply_perturbation(t, action_idx, cfg) for t in hate_train]
        all_perturbed.append(perturbed)
        all_preds.append(np.array(guardian.predict(perturbed)))
        all_probs.append(np.array(guardian.predict_proba(perturbed)))

    chosen_actions: List[int] = []
    chosen_texts: List[str] = []
    chosen_preds: List[int] = []
    chosen_probs: List[float] = []

    for i in range(len(hate_train)):
        success_actions = [a for a in range(NUM_ACTIONS) if all_preds[a][i] == 0]
        if success_actions:
            chosen = success_actions[0]
        else:
            chosen = int(np.argmin([all_probs[a][i] for a in range(NUM_ACTIONS)]))
        chosen_actions.append(chosen)
        chosen_texts.append(all_perturbed[chosen][i])
        chosen_preds.append(int(all_preds[chosen][i]))
        chosen_probs.append(float(all_probs[chosen][i]))

    df = pd.DataFrame({
        "original_text": hate_train,
        "perturbed_text": chosen_texts,
        "action": chosen_actions,
        "action_name": [ACTION_NAMES[a] for a in chosen_actions],
        "guardian_pred": chosen_preds,
        "guardian_hate_probability": chosen_probs,
        "attack_success": [1 if p == 0 else 0 for p in chosen_preds],
        "strategy": "oracle",
    })
    return df, {"chosen_action": "per_example_oracle"}


def generate_rl_stress(
    guardian: Guardian,
    hate_train: List[str],
    cfg: dict,
    device: torch.device,
    round_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    guardian_input = round_dir / "guardian_input"
    guardian.save(str(guardian_input))

    train_evader(cfg, str(guardian_input), round_dir)

    ecfg = cfg["evader"]
    policy = EvaderNetwork(ecfg["state_dim"], NUM_ACTIONS, ecfg["hidden_dim"]).to(device)
    policy.load_state_dict(torch.load(round_dir / "evader_policy.pt", map_location=device))
    policy.eval()

    records = []
    batch_size = 64
    for i in range(0, len(hate_train), batch_size):
        batch = hate_train[i:i + batch_size]
        probs = guardian.predict_proba(batch)
        states = np.array([extract_text_features(t, p) for t, p in zip(batch, probs)], dtype=np.float32)
        with torch.no_grad():
            logits = policy(torch.tensor(states, dtype=torch.float32).to(device))
            actions = torch.argmax(logits, dim=-1).cpu().tolist()
        sub = build_records_for_actions(guardian, batch, actions, cfg, "rl")
        records.append(sub)

    df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    return df, {"chosen_action": "rl_argmax_policy", "policy_path": str(round_dir / "evader_policy.pt")}


def eval_on_stress_df(guardian: Guardian, stress_df: pd.DataFrame) -> Dict[str, object]:
    if len(stress_df) == 0:
        return {"overall_asr": 0.0, "hate_f1": 0.0, "per_action": {}, "total": 0}
    texts = stress_df["perturbed_text"].tolist()
    preds = guardian.predict(texts)
    asr = float(np.mean(np.array(preds) == 0))
    hate_f1 = float(f1_score([1] * len(texts), preds, pos_label=1, zero_division=0))
    per_action = {}
    for action_name in ACTION_NAMES:
        subset = stress_df[stress_df["action_name"] == action_name]
        if len(subset) == 0:
            continue
        sub_preds = guardian.predict(subset["perturbed_text"].tolist())
        per_action[action_name] = {
            "asr": float(np.mean(np.array(sub_preds) == 0)),
            "count": int(len(subset)),
        }
    return {"overall_asr": asr, "hate_f1": hate_f1, "per_action": per_action, "total": int(len(texts))}


def robust_retrain(
    guardian: Guardian,
    train_texts: List[str],
    train_labels: List[int],
    stress_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    save_path: Path,
) -> Guardian:
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    rcfg = cfg["robust"]
    gcfg = cfg["guardian"]

    stress_texts = stress_df["perturbed_text"].tolist()
    stress_labels = [1] * len(stress_texts)
    aug_ratio = rcfg["augmentation_ratio"]
    n_aug = int(len(stress_texts) * aug_ratio)

    if n_aug > 0 and len(stress_texts) > 0:
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
    total_steps = max(1, len(train_loader) * rcfg["epochs"])
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    for epoch in range(rcfg["epochs"]):
        guardian.model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  Robust epoch {epoch + 1}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = guardian.model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(guardian.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        print(f"    Robust epoch {epoch + 1} loss: {total_loss / max(1, len(train_loader)):.4f}")

    guardian.save(str(save_path))
    return guardian


def action_distribution(stress_df: pd.DataFrame) -> Dict[str, float]:
    if len(stress_df) == 0:
        return {a: 0.0 for a in ACTION_NAMES}
    counts = stress_df["action_name"].value_counts(normalize=True).to_dict()
    return {a: float(counts.get(a, 0.0)) for a in ACTION_NAMES}


def save_summary_plots(round_results: List[dict], out: Path, strategy: str) -> None:
    if not round_results:
        return
    rounds = [r["round"] for r in round_results]

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, [r["train_attack_asr"] for r in round_results], "o-", label="Generated stress ASR")
    plt.plot(rounds, [r["post_oracle_asr"] for r in round_results], "s-", label="Post-round oracle ASR")
    plt.xlabel("Round")
    plt.ylabel("ASR")
    plt.title(f"Baseline outer loop: {strategy}")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "baseline_outer_loop_asr.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, [r["pre_clean_macro_f1"] for r in round_results], "o-", label="Pre-round clean macro-F1")
    plt.plot(rounds, [r["post_clean_macro_f1"] for r in round_results], "s-", label="Post-round clean macro-F1")
    plt.xlabel("Round")
    plt.ylabel("Macro-F1")
    plt.title(f"Clean performance: {strategy}")
    plt.ylim(0.45, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "baseline_outer_loop_clean_f1.png", dpi=150)
    plt.close()

    action_rows = []
    for r in round_results:
        row = {"round": r["round"]}
        row.update(r.get("action_distribution", {}))
        action_rows.append(row)
    adf = pd.DataFrame(action_rows).fillna(0.0)
    adf.to_csv(out / "baseline_outer_loop_action_distribution.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-round baseline outer-loop ablation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--strategy", type=str, required=True, choices=["random", "greedy", "oracle", "heuristic", "rl"])
    parser.add_argument("--initial_guardian", type=str, required=True)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_hate_train", type=int, default=None, help="Optional smoke-test limit for hateful train texts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, out)

    print(f"[INFO] Strategy: {args.strategy}")
    print(f"[INFO] Initial Guardian: {args.initial_guardian}")
    print(f"[INFO] Output dir: {out}")

    splits = load_and_split(cfg)
    train_texts = splits["train"]["text"].tolist()
    train_labels = splits["train"]["label"].tolist()
    val_hate = splits["val"][splits["val"]["label"] == 1]["text"].tolist()
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label"].tolist()
    hate_train = splits["train"][splits["train"]["label"] == 1]["text"].tolist()
    hate_test = splits["test"][splits["test"]["label"] == 1]["text"].tolist()

    if args.max_hate_train is not None:
        hate_train = hate_train[: args.max_hate_train]
        print(f"[INFO] Smoke-test mode: using {len(hate_train)} hateful train examples")

    print(f"[INFO] Train examples: {len(train_texts)} | Hate train: {len(hate_train)} | Hate val: {len(val_hate)} | Hate test: {len(hate_test)}")

    guardian = Guardian(cfg, device)
    guardian.load(args.initial_guardian)

    all_stress_dfs: List[pd.DataFrame] = []
    round_results: List[dict] = []

    for rnd in range(1, args.num_rounds + 1):
        print("\n" + "=" * 70)
        print(f"ROUND {rnd}/{args.num_rounds} | strategy={args.strategy}")
        print("=" * 70)

        round_dir = out / f"round_{rnd:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        pre_clean = eval_clean(guardian, test_texts, test_labels)
        pre_action_eval = evaluate_actions_on_hate_texts(guardian, hate_test, cfg)
        print(f"[Round {rnd}] Pre clean macro-F1={pre_clean['macro_f1']:.4f}; pre oracle ASR={pre_action_eval['oracle_best_of_all']['asr']:.4f}")

        if args.strategy == "random":
            stress_df, strategy_info = generate_random_stress(guardian, hate_train, cfg)
        elif args.strategy == "greedy":
            stress_df, strategy_info = generate_greedy_stress(guardian, hate_train, val_hate, cfg)
        elif args.strategy == "oracle":
            stress_df, strategy_info = generate_oracle_stress(guardian, hate_train, cfg)
        elif args.strategy == "heuristic":
            stress_df, strategy_info = generate_heuristic_stress(guardian, hate_train, cfg)
        elif args.strategy == "rl":
            stress_df, strategy_info = generate_rl_stress(guardian, hate_train, cfg, device, round_dir)
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        stress_df.to_parquet(round_dir / "baseline_stress_cases.parquet", index=False)
        stress_df.head(200).to_csv(round_dir / "baseline_stress_cases_sample.csv", index=False)

        train_attack_asr = float(stress_df["attack_success"].mean()) if len(stress_df) else 0.0
        dist = action_distribution(stress_df)
        dominant_action = max(dist, key=dist.get) if dist else "none"
        print(f"[Round {rnd}] Generated stress ASR={train_attack_asr:.4f}; dominant action={dominant_action}; dist={dist}")

        all_stress_dfs.append(stress_df)
        cumulative_stress = pd.concat(all_stress_dfs, ignore_index=True)
        pre_cumulative = eval_on_stress_df(guardian, cumulative_stress)
        print(f"[Round {rnd}] Pre-retrain cumulative stress ASR={pre_cumulative['overall_asr']:.4f}")

        guardian = robust_retrain(
            guardian,
            train_texts,
            train_labels,
            cumulative_stress,
            cfg,
            device,
            save_path=round_dir / "guardian_robust",
        )

        post_clean = eval_clean(guardian, test_texts, test_labels)
        post_cumulative = eval_on_stress_df(guardian, cumulative_stress)
        post_action_eval = evaluate_actions_on_hate_texts(guardian, hate_test, cfg)
        post_oracle_asr = post_action_eval["oracle_best_of_all"]["asr"]
        print(f"[Round {rnd}] Post clean macro-F1={post_clean['macro_f1']:.4f}; post oracle ASR={post_oracle_asr:.4f}")

        round_data = {
            "strategy": args.strategy,
            "round": rnd,
            "strategy_info": strategy_info,
            "pre_clean_macro_f1": pre_clean["macro_f1"],
            "pre_clean_hate_f1": pre_clean["hate_f1"],
            "pre_oracle_asr": pre_action_eval["oracle_best_of_all"]["asr"],
            "train_attack_asr": train_attack_asr,
            "action_distribution": dist,
            "dominant_action": dominant_action,
            "pre_robust_cumulative_asr": pre_cumulative["overall_asr"],
            "post_clean_macro_f1": post_clean["macro_f1"],
            "post_clean_hate_f1": post_clean["hate_f1"],
            "post_robust_cumulative_asr": post_cumulative["overall_asr"],
            "post_oracle_asr": post_oracle_asr,
            "post_action_eval": post_action_eval,
        }
        round_results.append(round_data)
        save_metrics(round_data, round_dir / "baseline_round_metrics.json")

    save_metrics(round_results, out / "baseline_all_rounds.json")

    summary = pd.DataFrame([
        {
            "strategy": r["strategy"],
            "round": r["round"],
            "dominant_action": r["dominant_action"],
            "train_attack_asr": r["train_attack_asr"],
            "pre_clean_macro_f1": r["pre_clean_macro_f1"],
            "post_clean_macro_f1": r["post_clean_macro_f1"],
            "pre_oracle_asr": r["pre_oracle_asr"],
            "post_oracle_asr": r["post_oracle_asr"],
            "post_robust_cumulative_asr": r["post_robust_cumulative_asr"],
        }
        for r in round_results
    ])
    summary.to_csv(out / "baseline_summary_table.csv", index=False)
    print("\n" + "=" * 70)
    print("BASELINE OUTER-LOOP SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    final = round_results[-1]
    final_report = {
        "strategy": args.strategy,
        "rounds": args.num_rounds,
        "final_clean_macro_f1": final["post_clean_macro_f1"],
        "final_clean_hate_f1": final["post_clean_hate_f1"],
        "final_oracle_asr": final["post_oracle_asr"],
        "final_cumulative_stress_asr": final["post_robust_cumulative_asr"],
        "final_dominant_action": final["dominant_action"],
        "final_guardian_path": str(out / f"round_{args.num_rounds:02d}" / "guardian_robust"),
    }
    save_metrics(final_report, out / "baseline_final_report.json")
    save_summary_plots(round_results, out, args.strategy)

    print("\n[INFO] Final report:")
    print(json.dumps(final_report, indent=2))
    print(f"[INFO] Wrote outputs to: {out}")


if __name__ == "__main__":
    main()
