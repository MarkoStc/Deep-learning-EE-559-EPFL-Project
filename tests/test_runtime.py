from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from red_queen.runtime import make_rng, seed_everything, select_device


def _draw() -> tuple[float, float, float]:
    legacy_numpy = np.random.rand()  # noqa: NPY002  checks the global state seed_everything sets
    return random.random(), float(legacy_numpy), torch.rand(1).item()


def test_seed_everything_makes_all_generators_repeatable() -> None:
    seed_everything(3)
    first = _draw()
    seed_everything(3)
    assert _draw() == first


def test_named_rng_streams_are_independent_and_repeatable() -> None:
    a1, a2, b = make_rng(1, "a"), make_rng(1, "a"), make_rng(1, "b")
    seq = [a1.random() for _ in range(3)]
    assert seq == [a2.random() for _ in range(3)]
    assert seq != [b.random() for _ in range(3)]


def test_select_cpu() -> None:
    assert select_device("cpu").type == "cpu"


def test_select_cuda_without_cuda_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        select_device("cuda")
    assert select_device("auto").type == "cpu"
