"""
Standalone baseline evaluation for Red Queen Moderation.

This file is intentionally standalone and additive: it does not require modifying
any existing project file. It evaluates four baselines against a trained Guardian:

1. Random Uniform: choose one of the 7 actions uniformly per hateful text.
2. Exhaustive Oracle: try all 7 actions and count success if any action fools Guardian.
3. Greedy Single-Action: pick the best single action on validation hateful texts,
   then use that action for all test hateful texts.
4. Feature-Based Heuristic: hand-written rules over the same 8 Evader features.

Example:
python3 -m src.baseline_evaluation \
  --config configs/default.yaml \
  --guardian-dir outputs/run_029/guardian_best \
  --output-dir outputs/baseline_results
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loading import load_and_split
from src.models import Guardian
from src.preprocess import (
    ACTION_NAMES,
    NUM_ACTIONS,
    apply_perturbation,
    extract_text_features,
)
from src.utils import get_device, load_config, sanitize_text, set_seed


RANDOM_UNIFORM = "random_uniform"
EXHAUSTIVE_ORACLE = "exhaustive_oracle"
GREEDY_SINGLE_ACTION = "greedy_single_action"
FEATURE_HEURISTIC = "feature_based_heuristic"


def _sanitize_for_output(text: str, cfg: dict) -> str:
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("sanitize_logs", True):
        return sanitize_text(text, mask_char=eval_cfg.get("mask_char", "█"))
    return text


def _safe_rate(values: Sequence[int | bool | float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _predict_in_batches(guardian: Guardian, texts: Sequence[str], batch_size: int) -> List[int]:
    if not texts:
        return []
    return guardian.predict(list(texts), batch_size=batch_size)


def _predict_proba_in_batches(guardian: Guardian, texts: Sequence[str], batch_size: int) -> List[float]:
    if not texts:
        return []
    return guardian.predict_proba(list(texts), batch_size=batch_size)


def _apply_actions(texts: Sequence[str], actions: Sequence[int], cfg: dict) -> List[str]:
    return [apply_perturbation(text, int(action), cfg) for text, action in zip(texts, actions)]


def _evaluate_fixed_actions(
    *,
    baseline_name: str,
    texts: Sequence[str],
    actions: Sequence[int],
    guardian: Guardian,
    cfg: dict,
    batch_size: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    perturbed = _apply_actions(texts, actions, cfg)
    preds = _predict_in_batches(guardian, perturbed, batch_size)

    records: List[Dict[str, object]] = []
    successes: List[int] = []

    for idx, (original, new_text, action, pred) in enumerate(zip(texts, perturbed, actions, preds)):
        success = int(pred == 0)
        successes.append(success)
        records.append({
            "baseline": baseline_name,
            "example_index": idx,
            "action": int(action),
            "action_name": ACTION_NAMES[int(action)],
            "guardian_pred": int(pred),
            "attack_success": success,
            "original_text_sanitized": _sanitize_for_output(original, cfg),
            "perturbed_text_sanitized": _sanitize_for_output(new_text, cfg),
        })

    metrics = {
        "baseline": baseline_name,
        "asr": _safe_rate(successes),
        "count": len(texts),
        "action_distribution": _action_distribution(actions),
    }
    return metrics, records


def _action_distribution(actions: Sequence[int]) -> Dict[str, float]:
    if not actions:
        return {name: 0.0 for name in ACTION_NAMES}
    counts = {name: 0 for name in ACTION_NAMES}
    for action in actions:
        counts[ACTION_NAMES[int(action)]] += 1
    total = float(len(actions))
    return {name: count / total for name, count in counts.items()}


def random_uniform_baseline(
    texts: Sequence[str],
    guardian: Guardian,
    cfg: dict,
    batch_size: int,
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rng = random.Random(seed)
    actions = [rng.randrange(NUM_ACTIONS) for _ in texts]
    return _evaluate_fixed_actions(
        baseline_name=RANDOM_UNIFORM,
        texts=texts,
        actions=actions,
        guardian=guardian,
        cfg=cfg,
        batch_size=batch_size,
    )


def exhaustive_oracle_baseline(
    texts: Sequence[str],
    guardian: Guardian,
    cfg: dict,
    batch_size: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    """Try all actions. Succeeds for a text if any action is predicted safe."""
    per_action_success = np.zeros((len(texts), NUM_ACTIONS), dtype=np.int32)
    per_action_metrics: Dict[str, Dict[str, float]] = {}

    for action_idx, action_name in enumerate(ACTION_NAMES):
        perturbed = [apply_perturbation(text, action_idx, cfg) for text in texts]
        preds = _predict_in_batches(guardian, perturbed, batch_size)
        successes = np.asarray([int(pred == 0) for pred in preds], dtype=np.int32)
        per_action_success[:, action_idx] = successes
        per_action_metrics[action_name] = {
            "asr": float(successes.mean()) if len(successes) else 0.0,
            "count": len(texts),
        }

    records: List[Dict[str, object]] = []
    chosen_actions: List[int] = []
    successes_final: List[int] = []

    for idx, text in enumerate(texts):
        successful_actions = np.where(per_action_success[idx] == 1)[0].tolist()
        if successful_actions:
            chosen_action = int(successful_actions[0])
            success = 1
        else:
            # No action worked. Keep action 0 as a placeholder.
            chosen_action = 0
            success = 0
        chosen_actions.append(chosen_action)
        successes_final.append(success)
        perturbed_text = apply_perturbation(text, chosen_action, cfg)
        records.append({
            "baseline": EXHAUSTIVE_ORACLE,
            "example_index": idx,
            "action": chosen_action,
            "action_name": ACTION_NAMES[chosen_action],
            "guardian_pred": 0 if success else 1,
            "attack_success": success,
            "successful_actions": ",".join(ACTION_NAMES[a] for a in successful_actions),
            "original_text_sanitized": _sanitize_for_output(text, cfg),
            "perturbed_text_sanitized": _sanitize_for_output(perturbed_text, cfg),
        })

    metrics = {
        "baseline": EXHAUSTIVE_ORACLE,
        "asr": _safe_rate(successes_final),
        "count": len(texts),
        "action_distribution": _action_distribution(chosen_actions),
        "per_action_asr_when_forced": per_action_metrics,
    }
    return metrics, records, per_action_metrics


def choose_greedy_action_on_validation(
    val_texts: Sequence[str],
    guardian: Guardian,
    cfg: dict,
    batch_size: int,
) -> Tuple[int, Dict[str, Dict[str, float]]]:
    per_action: Dict[str, Dict[str, float]] = {}
    best_action = 0
    best_asr = -1.0

    for action_idx, action_name in enumerate(ACTION_NAMES):
        perturbed = [apply_perturbation(text, action_idx, cfg) for text in val_texts]
        preds = _predict_in_batches(guardian, perturbed, batch_size)
        successes = [int(pred == 0) for pred in preds]
        asr = _safe_rate(successes)
        per_action[action_name] = {"asr": asr, "count": len(val_texts)}
        if asr > best_asr:
            best_asr = asr
            best_action = action_idx

    return best_action, per_action


def greedy_single_action_baseline(
    test_texts: Sequence[str],
    val_texts: Sequence[str],
    guardian: Guardian,
    cfg: dict,
    batch_size: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    best_action, validation_action_metrics = choose_greedy_action_on_validation(
        val_texts=val_texts,
        guardian=guardian,
        cfg=cfg,
        batch_size=batch_size,
    )
    actions = [best_action for _ in test_texts]
    metrics, records = _evaluate_fixed_actions(
        baseline_name=GREEDY_SINGLE_ACTION,
        texts=test_texts,
        actions=actions,
        guardian=guardian,
        cfg=cfg,
        batch_size=batch_size,
    )
    metrics["chosen_action"] = int(best_action)
    metrics["chosen_action_name"] = ACTION_NAMES[best_action]
    metrics["validation_per_action_asr"] = validation_action_metrics
    return metrics, records, validation_action_metrics


def feature_based_action(text: str, guardian_prob: float) -> int:
    """
    Hand-crafted policy over the same 8 features used by the Evader.

    extract_text_features returns:
    0 char_len, 1 word_count, 2 uppercase fraction, 3 punctuation fraction,
    4 has emoji, 5 has URL, 6 average word length, 7 guardian hate probability.
    """
    features = extract_text_features(text, guardian_prob)
    char_len = float(features[0])
    word_count = float(features[1])
    upper_frac = float(features[2])
    punct_frac = float(features[3])
    has_emoji = float(features[4])
    has_url = float(features[5])
    avg_word_len = float(features[6])
    hate_prob = float(features[7])

    # Very confident Guardian: use stronger character-level obfuscation.
    if hate_prob >= 0.85:
        if avg_word_len >= 0.35:
            return ACTION_NAMES.index("homoglyph")
        return ACTION_NAMES.index("leetspeak")

    # Short/direct texts: leetspeak often changes obvious surface tokens.
    if word_count <= 0.08 or char_len <= 0.12:
        return ACTION_NAMES.index("leetspeak")

    # Lots of punctuation/uppercase: whitespace perturbations are less semantic.
    if punct_frac >= 0.08 or upper_frac >= 0.18:
        return ACTION_NAMES.index("whitespace")

    # Text already has benign-looking markers; preserve meaning and add another.
    if has_emoji >= 1.0:
        return ACTION_NAMES.index("emoji")

    # URLs or long text: typo injection is cheap and distributed.
    if has_url >= 1.0 or char_len >= 0.45 or word_count >= 0.30:
        return ACTION_NAMES.index("typo")

    # Less confident Guardian: synonym swaps can be enough without heavy obfuscation.
    if hate_prob < 0.65:
        return ACTION_NAMES.index("synonym")

    return ACTION_NAMES.index("homoglyph")


def feature_based_heuristic_baseline(
    texts: Sequence[str],
    guardian: Guardian,
    cfg: dict,
    batch_size: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    probs = _predict_proba_in_batches(guardian, texts, batch_size)
    actions = [feature_based_action(text, prob) for text, prob in zip(texts, probs)]
    metrics, records = _evaluate_fixed_actions(
        baseline_name=FEATURE_HEURISTIC,
        texts=texts,
        actions=actions,
        guardian=guardian,
        cfg=cfg,
        batch_size=batch_size,
    )
    return metrics, records


def _save_summary_outputs(
    *,
    output_dir: Path,
    metrics: List[Dict[str, object]],
    records: List[Dict[str, object]],
    extra: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for item in metrics:
        row = {
            "baseline": item["baseline"],
            "asr": item["asr"],
            "count": item["count"],
            "chosen_action_name": item.get("chosen_action_name", ""),
        }
        for action_name, frac in item.get("action_distribution", {}).items():
            row[f"action_frac_{action_name}"] = frac
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("asr", ascending=False)
    summary_df.to_csv(output_dir / "baseline_summary.csv", index=False)

    records_df = pd.DataFrame(records)
    records_df.to_parquet(output_dir / "baseline_decisions.parquet", index=False)
    records_df.head(200).to_csv(output_dir / "baseline_decisions_sample_sanitized.csv", index=False)

    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": metrics, "extra": extra}, f, indent=2, default=str)

    plt.figure(figsize=(8, 4))
    plt.bar(summary_df["baseline"], summary_df["asr"])
    plt.ylabel("Attack Success Rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_asr.png", dpi=200)
    plt.close()

    action_cols = [col for col in summary_df.columns if col.startswith("action_frac_")]
    if action_cols:
        dist_df = summary_df[["baseline"] + action_cols].copy()
        dist_df = dist_df.set_index("baseline")
        dist_df.columns = [col.replace("action_frac_", "") for col in dist_df.columns]
        dist_df.plot(kind="bar", stacked=True, figsize=(10, 5))
        plt.ylabel("Fraction of chosen actions")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "baseline_action_distribution.png", dpi=200)
        plt.close()

    print("\n[INFO] Baseline summary:")
    print(summary_df.to_string(index=False))
    print(f"\n[INFO] Wrote baseline outputs to: {output_dir}")


def _limit_examples(texts: Sequence[str], max_examples: int | None, seed: int) -> List[str]:
    texts = list(texts)
    if max_examples is None or max_examples <= 0 or len(texts) <= max_examples:
        return texts
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    chosen = sorted(indices[:max_examples])
    return [texts[i] for i in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run additive Red Queen baseline evaluations.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--guardian-dir", required=True, help="Path to trained Guardian checkpoint directory.")
    parser.add_argument("--output-dir", default="outputs/baseline_results", help="Where to save baseline outputs.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--batch-size", type=int, default=None, help="Prediction batch size. Defaults to guardian.batch_size.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    batch_size = int(args.batch_size or cfg.get("guardian", {}).get("batch_size", 32))

    device = get_device()
    guardian = Guardian(cfg, device)
    guardian.load(args.guardian_dir)

    splits = load_and_split(cfg)
    val_texts = splits["val"][splits["val"]["label"] == 1]["text"].tolist()
    test_texts = splits["test"][splits["test"]["label"] == 1]["text"].tolist()

    val_texts = _limit_examples(val_texts, args.max_examples, seed)
    test_texts = _limit_examples(test_texts, args.max_examples, seed + 1)

    print(f"[INFO] Validation hateful examples: {len(val_texts)}")
    print(f"[INFO] Test hateful examples: {len(test_texts)}")
    print(f"[INFO] Actions: {ACTION_NAMES}")

    all_metrics: List[Dict[str, object]] = []
    all_records: List[Dict[str, object]] = []
    extra: Dict[str, object] = {}

    metrics, records = random_uniform_baseline(test_texts, guardian, cfg, batch_size, seed)
    all_metrics.append(metrics)
    all_records.extend(records)

    metrics, records, forced_per_action = exhaustive_oracle_baseline(test_texts, guardian, cfg, batch_size)
    all_metrics.append(metrics)
    all_records.extend(records)
    extra["exhaustive_oracle_forced_per_action_asr"] = forced_per_action

    metrics, records, validation_per_action = greedy_single_action_baseline(
        test_texts, val_texts, guardian, cfg, batch_size
    )
    all_metrics.append(metrics)
    all_records.extend(records)
    extra["greedy_validation_per_action_asr"] = validation_per_action

    metrics, records = feature_based_heuristic_baseline(test_texts, guardian, cfg, batch_size)
    all_metrics.append(metrics)
    all_records.extend(records)

    _save_summary_outputs(
        output_dir=Path(args.output_dir),
        metrics=all_metrics,
        records=all_records,
        extra=extra,
    )


if __name__ == "__main__":
    main()
