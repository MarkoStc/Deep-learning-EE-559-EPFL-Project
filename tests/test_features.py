from __future__ import annotations

import numpy as np
import pytest

from red_queen.features import FEATURE_NAMES, STATE_DIM, batch_features, text_features


def test_vector_shape_and_range() -> None:
    vec = text_features("Hello, WORLD! see https://example.org \U0001f60a", 0.8)
    assert vec.shape == (STATE_DIM,)
    assert vec.dtype == np.float32
    assert np.all((vec >= 0) & (vec <= 1))


def test_individual_features() -> None:
    vec = dict(zip(FEATURE_NAMES, text_features("AB cd https://x.y", 0.25), strict=True))
    assert vec["has_url"] == 1.0
    assert vec["has_emoji"] == 0.0
    assert vec["guardian_hate_prob"] == pytest.approx(0.25)
    assert vec["uppercase_share"] == pytest.approx(2 / len("AB cd https://x.y"))


def test_long_text_clips_to_one() -> None:
    vec = text_features("word " * 1000, 0.5)
    assert vec[0] == 1.0
    assert vec[1] == 1.0


def test_empty_text() -> None:
    vec = text_features("", 0.0)
    assert np.all(vec == 0)


def test_batch_features() -> None:
    out = batch_features(["a", "b c"], [0.1, 0.9])
    assert out.shape == (2, STATE_DIM)
    assert batch_features([], []).shape == (0, STATE_DIM)
    with pytest.raises(ValueError):
        batch_features(["a"], [])
