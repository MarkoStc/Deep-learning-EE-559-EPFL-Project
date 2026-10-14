"""
Train the Evader policy using REINFORCE (contextual bandit).
Usage:
    python -m red_queen.evader_policy --config configs/default.yaml --guardian_path outputs/run_001/guardian_best
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from red_queen.utils import load_config, set_seed, get_device, create_run_dir, save_metrics, save_config_snapshot
from red_queen.data_loading import load_and_split
from red_queen.models import Guardian, EvaderNetwork
from red_queen.evader_env import EvaderBanditEnv
from red_queen.preprocess import ACTION_NAMES, NUM_ACTIONS


def train_evader(cfg: dict, guardian_path: str, run_dir: Path):
    device = get_device()
    ecfg = cfg["evader"]

    # ── Load Guardian ────────────────────────────────────────
    guardian = Guardian(cfg, device)
    guardian.load(guardian_path)
    print("[INFO] Guardian loaded for Evader training")

    # ── Load data (only hateful examples) ────────────────────
    splits = load_and_split(cfg)
    hate_train = splits["train"][splits["train"]["label"] == 1]["text"].tolist()
    print(f"[INFO] Hateful training examples: {len(hate_train)}")

    if len(hate_train) < ecfg["batch_size"]:
        print(f"[WARN] Only {len(hate_train)} hate samples, reducing batch size")
        ecfg["batch_size"] = max(16, len(hate_train) // 2)

    # ── Setup ────────────────────────────────────────────────
    env = EvaderBanditEnv(hate_train, guardian, cfg)
    policy = EvaderNetwork(ecfg["state_dim"], NUM_ACTIONS, ecfg["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ecfg["learning_rate"])

    # Running baseline for variance reduction
    baseline = 0.0
    baseline_decay = ecfg["baseline_decay"]
    entropy_coeff = ecfg["entropy_coeff"]

    # ── Training loop ────────────────────────────────────────
    history = []

    for episode in tqdm(range(ecfg["episodes"]), desc="Evader training"):
        # 1. Sample batch of hateful texts
        texts = env.sample_batch()

        # 2. Get states
        states = env.get_states(texts)
        state_tensor = torch.tensor(states, dtype=torch.float32).to(device)

        # 3. Sample actions from policy
        actions_t, log_probs, entropies = policy.get_action_and_log_prob(state_tensor)
        actions = actions_t.cpu().tolist()

        # 4. Step environment
        result = env.step(texts, actions)
        rewards = torch.tensor(result["rewards"], dtype=torch.float32).to(device)

        # 5. REINFORCE update
        advantage = rewards - baseline
        policy_loss = -(log_probs * advantage).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + entropy_coeff * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Update baseline
        baseline = baseline_decay * baseline + (1 - baseline_decay) * rewards.mean().item()

        # Log
        if (episode + 1) % 50 == 0 or episode == 0:
            action_dist = env.get_action_distribution_stats(actions)
            history.append({
                "episode": episode + 1,
                "attack_success_rate": result["attack_success"],
                "mean_reward": rewards.mean().item(),
                "policy_loss": policy_loss.item(),
                "entropy": entropies.mean().item(),
                **{f"action_{k}": v for k, v in action_dist.items()},
            })
            if (episode + 1) % 200 == 0:
                print(
                    f"  Ep {episode+1}: ASR={result['attack_success']:.3f}, "
                    f"entropy={entropies.mean().item():.3f}"
                )

    # ── Save policy ──────────────────────────────────────────
    policy_path = run_dir / "evader_policy.pt"
    torch.save(policy.state_dict(), policy_path)
    print(f"[INFO] Evader policy saved to {policy_path}")

    # ── Save history ─────────────────────────────────────────
    df = pd.DataFrame(history)
    df.to_csv(run_dir / "evader_training_history.csv", index=False)
    save_metrics(
        {"final_asr": history[-1]["attack_success_rate"] if history else 0.0},
        run_dir / "evader_metrics.json",
    )

    # ── Plot ─────────────────────────────────────────────────
    if history:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        eps = [h["episode"] for h in history]

        axes[0].plot(eps, [h["attack_success_rate"] for h in history])
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Attack Success Rate")
        axes[0].set_title("Evader ASR over Training")

        axes[1].plot(eps, [h["entropy"] for h in history])
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Policy Entropy")
        axes[1].set_title("Policy Entropy")

        # Action distribution over time
        for action_name in ACTION_NAMES:
            key = f"action_{action_name}"
            if key in history[0]:
                axes[2].plot(eps, [h[key] for h in history], label=action_name)
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Action Frequency")
        axes[2].set_title("Action Distribution")
        axes[2].legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(run_dir / "evader_training_curves.png", dpi=150)
        plt.close()

    return policy


def main():
    parser = argparse.ArgumentParser(description="Train Evader policy")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--guardian_path", type=str, required=True,
                        help="Path to trained Guardian checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_dir = create_run_dir(cfg["output_dir"])
    save_config_snapshot(cfg, run_dir)

    train_evader(cfg, args.guardian_path, run_dir)


if __name__ == "__main__":
    main()
