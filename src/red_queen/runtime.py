"""Seeding, determinism and device selection."""

from __future__ import annotations

import logging
import os
import random
from typing import Literal

import numpy as np
import torch

from red_queen.logging_setup import fields

log = logging.getLogger(__name__)

DevicePreference = Literal["auto", "cpu", "cuda"]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch, and optionally force deterministic kernels.

    ``deterministic`` sets cuDNN to deterministic mode and asks torch to warn
    (not fail) on operations that have no deterministic implementation.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  global state is what libraries fall back to
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(preference: DevicePreference = "auto") -> torch.device:
    """Return the requested device; ``auto`` means CUDA when available."""
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' was requested but CUDA is not available")
    use_cuda = preference == "cuda" or (preference == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    log.info("selected device", extra=fields(device=str(device), name=name))
    return device


def make_rng(seed: int, stream: str) -> random.Random:
    """Independent, reproducible RNG for one named component (e.g. ``"stress"``).

    Giving each component its own stream means adding a random call in one
    place does not shift the random draws of every other component.
    """
    return random.Random(f"{seed}:{stream}")
