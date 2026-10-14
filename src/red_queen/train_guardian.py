"""
Train the baseline Guardian (DistilBERT hate speech classifier).
Usage:
    python -m red_queen.train_guardian --config configs/legacy.yaml
"""

import argparse

import torch
from sklearn.metrics import classification_report, f1_score
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


def train_one_epoch(guardian, dataloader, optimizer, scheduler, device):
    guardian.model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = guardian.model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(guardian.model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(guardian, dataloader, device):
    guardian.model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = guardian.model(**batch)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(
        all_labels, all_preds, target_names=["safe", "hate"], output_dict=True
    )
    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline Guardian")
    parser.add_argument("--config", type=str, default="configs/legacy.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    run_dir = create_run_dir(cfg["output_dir"])
    save_config_snapshot(cfg, run_dir)

    gcfg = cfg["guardian"]

    # ── Load data ────────────────────────────────────────────
    splits = load_and_split(cfg)
    guardian = Guardian(cfg, device)

    train_ds = guardian.make_dataset(
        splits["train"]["text"].tolist(),
        splits["train"]["label"].tolist(),
    )
    val_ds = guardian.make_dataset(
        splits["val"]["text"].tolist(),
        splits["val"]["label"].tolist(),
    )

    train_loader = DataLoader(train_ds, batch_size=gcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=gcfg["batch_size"])

    # ── Optimizer & scheduler ────────────────────────────────
    optimizer = AdamW(
        guardian.model.parameters(),
        lr=gcfg["learning_rate"],
        weight_decay=gcfg["weight_decay"],
    )
    total_steps = len(train_loader) * gcfg["epochs"]
    warmup_steps = int(total_steps * gcfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ────────────────────────────────────────
    best_f1 = 0
    history = []

    for epoch in range(gcfg["epochs"]):
        train_loss = train_one_epoch(guardian, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(guardian, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{gcfg['epochs']} — "
            f"train_loss: {train_loss:.4f}, "
            f"val_loss: {val_metrics['loss']:.4f}, "
            f"val_macro_f1: {val_metrics['macro_f1']:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            ckpt_dir = run_dir / "guardian_best"
            guardian.save(str(ckpt_dir))

    # ── Save training history ────────────────────────────────
    import pandas as pd

    pd.DataFrame(history).to_csv(run_dir / "guardian_training_history.csv", index=False)
    save_metrics(
        {"best_val_macro_f1": best_f1, "history": history}, run_dir / "guardian_metrics.json"
    )

    # ── Plot training curve ──────────────────────────────────
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = [h["epoch"] for h in history]
    ax1.plot(epochs_range, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs_range, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Guardian Training Loss")
    ax1.legend()

    ax2.plot(epochs_range, [h["val_macro_f1"] for h in history], label="macro-F1")
    ax2.plot(epochs_range, [h["val_weighted_f1"] for h in history], label="weighted-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Guardian Validation F1")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(run_dir / "guardian_training_curve.png", dpi=150)
    plt.close()
    print(f"[INFO] Training curve saved to {run_dir / 'guardian_training_curve.png'}")
    print(f"[INFO] Guardian training complete. Best val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
