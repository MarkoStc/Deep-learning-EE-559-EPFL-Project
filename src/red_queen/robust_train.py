"""
Robust training: retrain the Guardian by augmenting the training set
with stress cases discovered by the Evader.

Usage:
    python -m red_queen.robust_train \
        --config configs/default.yaml \
        --guardian_path outputs/run_001/guardian_best \
        --stress_cases outputs/stress_cases/stress_cases.parquet
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from red_queen.data_loading import load_and_split
from red_queen.models import Guardian
from red_queen.utils import (
    create_run_dir,
    get_device,
    load_config,
    save_config_snapshot,
    save_metrics,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Robust-retrain Guardian")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--guardian_path", type=str, required=True, help="Path to baseline Guardian checkpoint"
    )
    parser.add_argument(
        "--stress_cases", type=str, required=True, help="Path to stress_cases.parquet"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    run_dir = create_run_dir(cfg["output_dir"])
    save_config_snapshot(cfg, run_dir)

    rcfg = cfg["robust"]
    gcfg = cfg["guardian"]

    # ── Load baseline Guardian ───────────────────────────────
    guardian = Guardian(cfg, device)
    guardian.load(args.guardian_path)
    print("[INFO] Loaded baseline Guardian for robust retraining")

    # ── Load original training data ──────────────────────────
    splits = load_and_split(cfg)
    train_texts = splits["train"]["text"].tolist()
    train_labels = splits["train"]["label"].tolist()

    # ── Load stress cases ────────────────────────────────────
    stress_df = pd.read_parquet(args.stress_cases)
    # Use ALL stress cases (both successful and failed attacks)
    # with their TRUE labels (all hateful = 1)
    stress_texts = stress_df["perturbed_text"].tolist()
    stress_labels = [1] * len(stress_texts)  # ground truth: all hateful
    print(f"[INFO] Stress cases loaded: {len(stress_texts)}")

    # ── Augment training set ─────────────────────────────────
    aug_ratio = rcfg["augmentation_ratio"]
    n_aug = int(len(stress_texts) * aug_ratio)
    if n_aug > 0:
        indices = np.random.choice(len(stress_texts), size=n_aug, replace=False)
        aug_texts = [stress_texts[i] for i in indices]
        aug_labels = [stress_labels[i] for i in indices]
    else:
        aug_texts, aug_labels = stress_texts, stress_labels

    combined_texts = train_texts + aug_texts
    combined_labels = train_labels + aug_labels
    print(
        f"[INFO] Augmented training set: {len(combined_texts)} "
        f"(original: {len(train_texts)}, augmented: {len(aug_texts)})"
    )

    # ── Create datasets ──────────────────────────────────────
    train_ds = guardian.make_dataset(combined_texts, combined_labels)
    val_ds = guardian.make_dataset(
        splits["val"]["text"].tolist(),
        splits["val"]["label"].tolist(),
    )

    train_loader = DataLoader(train_ds, batch_size=gcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=gcfg["batch_size"])

    # ── Optimizer ────────────────────────────────────────────
    optimizer = AdamW(
        guardian.model.parameters(),
        lr=rcfg["learning_rate"],
        weight_decay=gcfg["weight_decay"],
    )
    total_steps = len(train_loader) * rcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    # ── Training loop ────────────────────────────────────────
    best_f1 = 0
    history = []

    for epoch in range(rcfg["epochs"]):
        guardian.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Robust epoch {epoch + 1}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = guardian.model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(guardian.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        guardian.model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = guardian.model(**batch).logits
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(batch["labels"].cpu().tolist())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"  Robust epoch {epoch + 1} — loss: {avg_loss:.4f}, val_macro_f1: {val_f1:.4f}")

        history.append({"epoch": epoch + 1, "loss": avg_loss, "val_macro_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            guardian.save(str(run_dir / "guardian_robust"))

    # ── Save ─────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(run_dir / "robust_training_history.csv", index=False)
    save_metrics({"best_val_macro_f1": best_f1}, run_dir / "robust_metrics.json")
    print(f"[INFO] Robust Guardian saved. Best val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
