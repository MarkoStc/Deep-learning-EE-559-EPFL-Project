from __future__ import annotations

import random

import pytest

from red_queen.config import PerturbationConfig
from red_queen.perturbations import (
    BENIGN_EMOJIS,
    DEFAULT_ORDER,
    HOMOGLYPHS,
    LEET_MAP,
    ZERO_WIDTH,
    Emoji,
    Homoglyph,
    Leetspeak,
    Noop,
    PerturbationSet,
    Synonym,
    Typo,
    Whitespace,
)

TEXT = "The quick brown fox jumps over the lazy sleeping dog again"


class FakeSynonyms:
    """Deterministic stand-in for WordNet: no corpus download in tests."""

    def __init__(self, table: dict[str, list[str]]) -> None:
        self.table = table
        self.calls: list[str] = []

    def synonyms(self, word: str) -> list[str]:
        self.calls.append(word)
        return self.table.get(word, [])


@pytest.fixture
def actions() -> PerturbationSet:
    return PerturbationSet.from_config(
        PerturbationConfig(), synonyms=FakeSynonyms({"quick": ["fast"], "lazy": ["idle"]})
    )


def test_action_order_is_stable(actions: PerturbationSet) -> None:
    # Saved Evader checkpoints index actions by position.
    assert actions.names == DEFAULT_ORDER
    assert len(actions) == 7


@pytest.mark.parametrize("action", range(7))
def test_same_seed_same_output(actions: PerturbationSet, action: int) -> None:
    a = actions.apply(TEXT, action, random.Random(5))
    b = actions.apply(TEXT, action, random.Random(5))
    assert a == b


def test_noop_is_identity() -> None:
    assert Noop().apply(TEXT, random.Random(0)) == TEXT


def test_typo_prob_zero_and_one() -> None:
    assert Typo(0.0).apply(TEXT, random.Random(0)) == TEXT
    out = Typo(1.0).apply("abcdef ghijkl", random.Random(0))
    assert out != "abcdef ghijkl"
    for orig, new in zip(["abcdef", "ghijkl"], out.split(), strict=True):
        assert new[0] == orig[0]  # first character is never touched
        assert abs(len(new) - len(orig)) <= 1


def test_typo_skips_short_words() -> None:
    assert Typo(1.0).apply("an ox is", random.Random(0)) == "an ox is"


def test_leetspeak_only_uses_map() -> None:
    out = Leetspeak(1.0).apply("glass toes", random.Random(0))
    assert out == "91455 7035"
    assert set(LEET_MAP) & set(out) == set()


def test_whitespace_inserts_invisible_chars_only() -> None:
    out = Whitespace(1.0).apply(TEXT, random.Random(0))
    assert out != TEXT
    assert "".join(ch for ch in out if ch not in ZERO_WIDTH) == TEXT


def test_homoglyph_preserves_length_and_looks() -> None:
    out = Homoglyph(1.0).apply("peace", random.Random(0))
    assert len(out) == len("peace")
    assert out == "".join(HOMOGLYPHS.get(c, c) for c in "peace")
    assert out != "peace"


def test_synonym_respects_replacement_budget() -> None:
    lookup = FakeSynonyms({"quick": ["fast"], "lazy": ["idle"], "brown": ["tan"]})
    out = Synonym(1, lookup).apply(TEXT, random.Random(1))
    changed = [a for a, b in zip(TEXT.split(), out.split(), strict=True) if a != b]
    assert len(changed) == 1


def test_synonym_without_matches_is_noop() -> None:
    assert Synonym(2, FakeSynonyms({})).apply(TEXT, random.Random(0)) == TEXT


def test_emoji_inserts_count_benign_emojis() -> None:
    out = Emoji(2).apply(TEXT, random.Random(0)).split()
    inserted = [w for w in out if w in BENIGN_EMOJIS]
    assert len(inserted) == 2
    assert [w for w in out if w not in BENIGN_EMOJIS] == TEXT.split()


def test_emoji_on_empty_text() -> None:
    assert Emoji(3).apply("", random.Random(0)) == ""


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        PerturbationSet([Noop(), Noop()])
