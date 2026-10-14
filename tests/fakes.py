"""Test doubles that stand in for the Guardian without loading a model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt


class KeywordClassifier:
    """Calls a text hate when it contains one of ``keywords`` as an exact word.

    Character-level perturbations (leetspeak, homoglyphs) break the exact match
    and fool it; word insertions (emoji) do not. That gives the Evader a clear
    best action to learn.
    """

    def __init__(self, keywords: Iterable[str]) -> None:
        self.keywords = frozenset(keywords)
        self.calls = 0

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        self.calls += 1
        return np.array(
            [0.9 if self.keywords & set(t.split()) else 0.1 for t in texts], dtype=np.float64
        )

    def predict(self, texts: Sequence[str]) -> npt.NDArray[np.int64]:
        return (self.predict_proba(texts) > 0.5).astype(np.int64)
