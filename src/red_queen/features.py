"""Handcrafted state features for the Evader.

The Evader sees a text only through this fixed-length vector. All features are
scaled into [0, 1]. The last one is the Guardian's own P(hate), which lets the
policy condition on how confident the classifier already is.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence

import emoji as emoji_lib
import numpy as np
import numpy.typing as npt

FEATURE_NAMES = (
    "char_length",
    "word_count",
    "uppercase_share",
    "punctuation_share",
    "has_emoji",
    "has_url",
    "mean_word_length",
    "guardian_hate_prob",
)
STATE_DIM = len(FEATURE_NAMES)

# Scales chosen so typical comments fall well inside [0, 1]; longer ones clip to 1.
CHAR_LENGTH_SCALE = 500.0
WORD_COUNT_SCALE = 100.0
WORD_LENGTH_SCALE = 15.0

_URL = re.compile(r"https?://")
_PUNCTUATION = frozenset(string.punctuation)


def text_features(text: str, guardian_prob: float) -> npt.NDArray[np.float32]:
    """Feature vector for one text, in the order of :data:`FEATURE_NAMES`."""
    words = text.split()
    n_chars = max(len(text), 1)
    mean_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    return np.array(
        [
            min(len(text) / CHAR_LENGTH_SCALE, 1.0),
            min(len(words) / WORD_COUNT_SCALE, 1.0),
            sum(c.isupper() for c in text) / n_chars,
            sum(c in _PUNCTUATION for c in text) / n_chars,
            1.0 if emoji_lib.emoji_count(text) > 0 else 0.0,
            1.0 if _URL.search(text) else 0.0,
            min(mean_word_len / WORD_LENGTH_SCALE, 1.0),
            guardian_prob,
        ],
        dtype=np.float32,
    )


def batch_features(
    texts: Sequence[str], guardian_probs: Sequence[float]
) -> npt.NDArray[np.float32]:
    """Stack :func:`text_features` into a ``(len(texts), STATE_DIM)`` array."""
    if len(texts) != len(guardian_probs):
        raise ValueError("texts and guardian_probs must have the same length")
    if not texts:
        return np.zeros((0, STATE_DIM), dtype=np.float32)
    return np.stack([text_features(t, p) for t, p in zip(texts, guardian_probs, strict=True)])
