"""The Evader: a contextual bandit that learns which perturbation fools the Guardian.

Each episode samples a batch of hateful texts. The policy sees the handcrafted
features of each text (:mod:`red_queen.features`), picks one perturbation per
text, and gets reward 1 when the Guardian labels the perturbed text safe. The
policy is trained with REINFORCE, a running-mean reward baseline and an entropy
bonus. There is one decision per text, so there is no discounting.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from tqdm import tqdm

from red_queen.config import EvaderConfig
from red_queen.errors import ArtifactError
from red_queen.features import STATE_DIM, batch_features
from red_queen.guardian import TextClassifier
from red_queen.logging_setup import fields
from red_queen.perturbations import PerturbationSet
from red_queen.tracking import NullTracker, Tracker

log = logging.getLogger(__name__)

POLICY_FILE = "evader_policy.pt"


def effective_batch_size(n_texts: int, cfg: EvaderConfig) -> int:
    """Texts per episode; shrinks for small datasets instead of editing the config."""
    if n_texts < cfg.batch_size:
        return max(1, min(n_texts, max(cfg.min_batch_size, n_texts // 2)))
    return cfg.batch_size


@dataclass(frozen=True)
class StepResult:
    perturbed: list[str]
    rewards: npt.NDArray[np.float32]

    @property
    def success_rate(self) -> float:
        return float(self.rewards.mean()) if len(self.rewards) else 0.0


class BanditEnv:
    """Contextual bandit over hateful texts with the Guardian as the judge."""

    def __init__(
        self,
        hate_texts: Sequence[str],
        classifier: TextClassifier,
        perturbations: PerturbationSet,
        rng: random.Random,
        batch_size: int,
    ) -> None:
        if not hate_texts:
            raise ValueError("the Evader needs at least one hateful text")
        self.texts = list(hate_texts)
        self.classifier = classifier
        self.perturbations = perturbations
        self.rng = rng
        self.batch_size = batch_size

    def sample(self) -> list[str]:
        return self.rng.sample(self.texts, min(self.batch_size, len(self.texts)))

    def observe(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """State vectors, including the Guardian's current P(hate) of each text."""
        probs = self.classifier.predict_proba(texts)
        return batch_features(texts, [float(p) for p in probs])

    def step(self, texts: Sequence[str], actions: Sequence[int]) -> StepResult:
        perturbed = [
            self.perturbations.apply(t, a, self.rng) for t, a in zip(texts, actions, strict=True)
        ]
        preds = self.classifier.predict(perturbed)
        return StepResult(perturbed, (preds == 0).astype(np.float32))


@dataclass(frozen=True)
class PolicySample:
    actions: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor


class EvaderPolicy(nn.Module):
    """Two-layer MLP from state features to action logits."""

    def __init__(self, num_actions: int, hidden_dim: int, state_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(states)
        return logits

    def sample(self, states: torch.Tensor, generator: torch.Generator) -> PolicySample:
        """Sample one action per state from the softmax policy."""
        log_p = torch.log_softmax(self.forward(states), dim=-1)
        actions = torch.multinomial(log_p.exp(), 1, generator=generator).squeeze(-1)
        return PolicySample(
            actions=actions,
            log_probs=log_p.gather(-1, actions.unsqueeze(-1)).squeeze(-1),
            entropy=-(log_p.exp() * log_p).sum(-1),
        )

    @torch.no_grad()
    def greedy(self, states: torch.Tensor) -> torch.Tensor:
        return self.forward(states).argmax(dim=-1)

    def save(self, path: Path, action_names: Sequence[str]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "action_names": list(action_names),
                "state_dim": self.state_dim,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path, action_names: Sequence[str], device: torch.device) -> EvaderPolicy:
        """Load a policy and check it was trained on the same action space."""
        if not path.is_file():
            raise ArtifactError(f"Evader policy {path} does not exist")
        blob = torch.load(path, map_location=device, weights_only=True)
        if list(blob["action_names"]) != list(action_names):
            raise ArtifactError(
                f"{path} was trained on actions {blob['action_names']}, "
                f"but the current perturbation set is {list(action_names)}"
            )
        policy = cls(len(action_names), int(blob["hidden_dim"]), int(blob["state_dim"]))
        policy.load_state_dict(blob["state_dict"])
        return policy.to(device).eval()


@dataclass
class EvaderResult:
    history: list[dict[str, float]] = field(default_factory=list)
    last_batch_asr: float = 0.0
    mean_asr_final_tenth: float = 0.0
    """Mean batch ASR over the last 10% of episodes; less noisy than the last batch."""

    def to_dict(self) -> dict[str, object]:
        return {
            "last_batch_asr": self.last_batch_asr,
            "mean_asr_final_tenth": self.mean_asr_final_tenth,
            "history": self.history,
        }


class EvaderTrainer:
    """REINFORCE with a running-mean baseline and an entropy bonus."""

    def __init__(
        self,
        env: BanditEnv,
        policy: EvaderPolicy,
        cfg: EvaderConfig,
        device: torch.device,
        *,
        seed: int,
        tracker: Tracker | None = None,
        stage: str = "evader",
    ) -> None:
        self.env = env
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.tracker = tracker or NullTracker()
        self.stage = stage

    def train(self) -> EvaderResult:
        cfg = self.cfg
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        names = self.env.perturbations.names
        baseline = 0.0
        batch_asrs: list[float] = []
        result = EvaderResult()
        self.policy.train()
        for episode in tqdm(range(1, cfg.episodes + 1), desc=self.stage, leave=False):
            texts = self.env.sample()
            states = torch.from_numpy(self.env.observe(texts))
            sample = self.policy.sample(states.to(self.device), self.generator)
            step = self.env.step(texts, sample.actions.cpu().tolist())
            rewards = torch.from_numpy(step.rewards).to(self.device)

            policy_loss = -(sample.log_probs * (rewards - baseline)).mean()
            loss = policy_loss - cfg.entropy_coeff * sample.entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            baseline = cfg.baseline_decay * baseline + (1 - cfg.baseline_decay) * step.success_rate
            batch_asrs.append(step.success_rate)
            if episode == 1 or episode % cfg.log_every == 0:
                row = {
                    "episode": float(episode),
                    "attack_success_rate": step.success_rate,
                    "policy_loss": float(policy_loss.item()),
                    "entropy": float(sample.entropy.mean().item()),
                    **action_shares(sample.actions.cpu().tolist(), names),
                }
                result.history.append(row)
                self.tracker.log(
                    self.stage, episode, {k: v for k, v in row.items() if k != "episode"}
                )

        tail = batch_asrs[-max(1, len(batch_asrs) // 10) :]
        result.last_batch_asr = batch_asrs[-1]
        result.mean_asr_final_tenth = float(np.mean(tail))
        log.info(
            "evader trained",
            extra=fields(
                episodes=cfg.episodes,
                last_batch_asr=result.last_batch_asr,
                mean_asr_final_tenth=result.mean_asr_final_tenth,
            ),
        )
        self.policy.eval()
        return result


def action_shares(actions: Sequence[int], names: Sequence[str]) -> dict[str, float]:
    """Share of each action in ``actions``, keyed ``action_<name>``."""
    total = max(len(actions), 1)
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=len(names))
    return {f"action_{name}": float(counts[i] / total) for i, name in enumerate(names)}
