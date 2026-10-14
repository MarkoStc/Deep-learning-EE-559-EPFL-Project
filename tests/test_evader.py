from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from red_queen.config import EvaderConfig, PerturbationConfig
from red_queen.errors import ArtifactError
from red_queen.evader import (
    BanditEnv,
    EvaderPolicy,
    EvaderTrainer,
    action_shares,
    effective_batch_size,
)
from red_queen.features import STATE_DIM
from red_queen.perturbations import PerturbationSet
from red_queen.tracking import CsvTracker
from tests.fakes import KeywordClassifier

CPU = torch.device("cpu")
MARKER = "zorblat"
HATE_TEXTS = [f"the river {MARKER} looks calm today number {i}" for i in range(40)]


class NoSynonyms:
    def synonyms(self, word: str) -> list[str]:
        return []


@pytest.fixture
def actions() -> PerturbationSet:
    return PerturbationSet.from_config(PerturbationConfig(), synonyms=NoSynonyms())


def make_env(actions: PerturbationSet, batch_size: int = 16) -> BanditEnv:
    return BanditEnv(HATE_TEXTS, KeywordClassifier([MARKER]), actions, random.Random(0), batch_size)


def test_effective_batch_size_shrinks_without_touching_config() -> None:
    cfg = EvaderConfig(batch_size=64, min_batch_size=16)
    assert effective_batch_size(1000, cfg) == 64
    assert effective_batch_size(40, cfg) == 20
    assert effective_batch_size(10, cfg) == 10
    assert cfg.batch_size == 64


def test_env_rewards(actions: PerturbationSet) -> None:
    env = make_env(actions)
    texts = env.sample()
    assert len(texts) == 16
    assert env.observe(texts).shape == (16, STATE_DIM)
    noop = env.step(texts, [0] * len(texts))
    assert noop.success_rate == 0.0  # untouched marker is always caught
    leet = env.step(texts, [actions.names.index("leetspeak")] * len(texts))
    assert leet.success_rate > 0.5


def test_env_rejects_empty_dataset(actions: PerturbationSet) -> None:
    with pytest.raises(ValueError):
        BanditEnv([], KeywordClassifier([]), actions, random.Random(0), 4)


def test_policy_sample_shapes_and_repeatability() -> None:
    policy = EvaderPolicy(num_actions=7, hidden_dim=8)
    states = torch.rand(5, STATE_DIM)
    a = policy.sample(states, torch.Generator().manual_seed(1))
    b = policy.sample(states, torch.Generator().manual_seed(1))
    assert a.actions.shape == (5,)
    assert torch.equal(a.actions, b.actions)
    assert torch.all(a.log_probs <= 0)
    assert torch.all(a.entropy > 0)
    assert policy.greedy(states).shape == (5,)


def test_policy_save_load_roundtrip(tmp_path: Path, actions: PerturbationSet) -> None:
    policy = EvaderPolicy(len(actions), hidden_dim=8)
    path = policy.save(tmp_path / "p.pt", actions.names)
    loaded = EvaderPolicy.load(path, actions.names, CPU)
    states = torch.rand(3, STATE_DIM)
    assert torch.equal(loaded.greedy(states), policy.greedy(states))


def test_policy_load_checks_action_space(tmp_path: Path, actions: PerturbationSet) -> None:
    path = EvaderPolicy(len(actions), hidden_dim=8).save(tmp_path / "p.pt", actions.names)
    with pytest.raises(ArtifactError, match="trained on actions"):
        EvaderPolicy.load(path, list(reversed(actions.names)), CPU)
    with pytest.raises(ArtifactError, match="does not exist"):
        EvaderPolicy.load(tmp_path / "missing.pt", actions.names, CPU)


def test_reinforce_learns_the_best_action(actions: PerturbationSet, tmp_path: Path) -> None:
    cfg = EvaderConfig(episodes=150, batch_size=16, learning_rate=0.02, log_every=25)
    policy = EvaderPolicy(len(actions), cfg.hidden_dim)
    tracker = CsvTracker(tmp_path)
    trainer = EvaderTrainer(make_env(actions), policy, cfg, CPU, seed=0, tracker=tracker)
    result = trainer.train()

    first, last = result.history[0], result.history[-1]
    assert last["attack_success_rate"] > first["attack_success_rate"]
    assert result.mean_asr_final_tenth > 0.7
    states = torch.from_numpy(make_env(actions).observe(HATE_TEXTS))
    chosen = {actions.names[a] for a in policy.greedy(states).tolist()}
    assert chosen <= {"leetspeak", "homoglyph", "typo", "whitespace"}
    assert "noop" not in chosen
    assert (tmp_path / "evader.csv").exists()


def test_action_shares() -> None:
    shares = action_shares([0, 0, 2], ["a", "b", "c"])
    assert shares == {
        "action_a": pytest.approx(2 / 3),
        "action_b": 0.0,
        "action_c": pytest.approx(1 / 3),
    }
    assert sum(action_shares([], ["a"]).values()) == 0.0
    assert np.isclose(sum(action_shares([1, 2], ["a", "b", "c"]).values()), 1.0)
