"""Surface-level text perturbations: the Evader's action space.

Each perturbation only rewrites characters or words of an existing, already
labelled text. None of them adds new hateful content. Every ``apply`` call
takes the RNG explicitly, so results depend only on the seed and not on how
many random numbers other code has drawn.

Action indices are part of saved Evader checkpoints; keep the order of
:data:`DEFAULT_ORDER` stable.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from red_queen.config import PerturbationConfig

log = logging.getLogger(__name__)

MIN_TYPO_WORD_LEN = 3
MIN_WHITESPACE_WORD_LEN = 4
MIN_SYNONYM_WORD_LEN = 4


class Perturbation(ABC):
    """One way of rewriting a text."""

    name: ClassVar[str]

    @abstractmethod
    def apply(self, text: str, rng: random.Random) -> str:
        """Return a perturbed copy of ``text``."""


class Noop(Perturbation):
    name = "noop"

    def apply(self, text: str, rng: random.Random) -> str:
        return text


@dataclass(frozen=True)
class Typo(Perturbation):
    """Swap, delete or duplicate one inner character in some words."""

    prob: float
    name: ClassVar[str] = "typo"

    def apply(self, text: str, rng: random.Random) -> str:
        out = []
        for word in text.split():
            if len(word) >= MIN_TYPO_WORD_LEN and rng.random() < self.prob:
                word = self._edit(word, rng)  # noqa: PLW2901
            out.append(word)
        return " ".join(out)

    @staticmethod
    def _edit(word: str, rng: random.Random) -> str:
        i = rng.randint(1, len(word) - 2)  # keep the first and last character
        op = rng.choice(("swap", "delete", "duplicate"))
        if op == "swap":
            return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if op == "delete":
            return word[:i] + word[i + 1 :]
        return word[:i] + word[i] + word[i:]


LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "l": "1", "g": "9"}


@dataclass(frozen=True)
class Leetspeak(Perturbation):
    """Replace some letters with digits (``e`` -> ``3``)."""

    prob: float
    name: ClassVar[str] = "leetspeak"

    def apply(self, text: str, rng: random.Random) -> str:
        return "".join(
            LEET_MAP[ch.lower()] if ch.lower() in LEET_MAP and rng.random() < self.prob else ch
            for ch in text
        )


ZERO_WIDTH = ("\u200b", "\u200c", "\u200d", "\ufeff")


@dataclass(frozen=True)
class Whitespace(Perturbation):
    """Insert one zero-width character inside some words."""

    prob: float
    name: ClassVar[str] = "whitespace"

    def apply(self, text: str, rng: random.Random) -> str:
        out = []
        for word in text.split():
            if len(word) >= MIN_WHITESPACE_WORD_LEN and rng.random() < self.prob:
                i = rng.randint(1, len(word) - 1)
                word = word[:i] + rng.choice(ZERO_WIDTH) + word[i:]  # noqa: PLW2901
            out.append(word)
        return " ".join(out)


# Latin letter -> visually identical Cyrillic letter.
HOMOGLYPHS = {
    "a": "\u0430",
    "c": "\u0441",
    "e": "\u0435",
    "o": "\u043e",
    "p": "\u0440",
    "x": "\u0445",
    "s": "\u0455",
    "i": "\u0456",
}


@dataclass(frozen=True)
class Homoglyph(Perturbation):
    """Replace some Latin letters with Cyrillic look-alikes."""

    prob: float
    name: ClassVar[str] = "homoglyph"

    def apply(self, text: str, rng: random.Random) -> str:
        return "".join(
            HOMOGLYPHS[ch.lower()] if ch.lower() in HOMOGLYPHS and rng.random() < self.prob else ch
            for ch in text
        )


class SynonymLookup(Protocol):
    def synonyms(self, word: str) -> list[str]: ...


@dataclass
class WordNetSynonyms:
    """Synonyms from NLTK WordNet; downloads the corpus on first use.

    If the corpus cannot be loaded (offline machine), every lookup returns an
    empty list and the synonym action becomes a no-op, with one warning.
    """

    max_synsets: int = 3
    _ready: bool | None = field(default=None, init=False, repr=False)

    def synonyms(self, word: str) -> list[str]:
        if not self._ensure_corpus():
            return []
        from nltk.corpus import wordnet  # noqa: PLC0415  heavy import, only when used

        found: set[str] = set()
        for synset in wordnet.synsets(word)[: self.max_synsets]:
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    found.add(name)
        return sorted(found)  # sorted: set order is not stable across processes

    def _ensure_corpus(self) -> bool:
        if self._ready is None:
            self._ready = _load_wordnet()
        return self._ready


def _load_wordnet() -> bool:
    import nltk  # noqa: PLC0415

    try:
        for resource in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True, raise_on_error=True)
        from nltk.corpus import wordnet  # noqa: PLC0415

        wordnet.ensure_loaded()
    except Exception as exc:  # nltk raises many unrelated types for network failures
        log.warning("WordNet unavailable, synonym perturbation disabled: %s", exc)
        return False
    return True


@dataclass(frozen=True)
class Synonym(Perturbation):
    """Replace up to ``max_replacements`` alphabetic words with a synonym."""

    max_replacements: int
    lookup: SynonymLookup
    name: ClassVar[str] = "synonym"

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        candidates = [
            i for i, w in enumerate(words) if len(w) >= MIN_SYNONYM_WORD_LEN and w.isalpha()
        ]
        rng.shuffle(candidates)
        replaced = 0
        for i in candidates:
            if replaced >= self.max_replacements:
                break
            options = self.lookup.synonyms(words[i].lower())
            if options:
                words[i] = rng.choice(options)
                replaced += 1
        return " ".join(words)


BENIGN_EMOJIS = (
    "\U0001f60a",
    "\U0001f44d",
    "\u2764\ufe0f",
    "\U0001f642",
    "\u2728",
    "\U0001f31f",
    "\U0001f4af",
    "\U0001f525",
    "\U0001f60e",
    "\U0001f389",
    "\U0001f44f",
    "\U0001f4aa",
    "\U0001f64f",
    "\U0001f607",
    "\U0001f308",
    "\u2600\ufe0f",
)


@dataclass(frozen=True)
class Emoji(Perturbation):
    """Insert ``count`` benign emojis at random word boundaries."""

    count: int
    name: ClassVar[str] = "emoji"

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        for _ in range(min(self.count, len(words))):
            words.insert(rng.randint(0, len(words)), rng.choice(BENIGN_EMOJIS))
        return " ".join(words)


DEFAULT_ORDER = ("noop", "typo", "leetspeak", "whitespace", "homoglyph", "synonym", "emoji")


class PerturbationSet:
    """Ordered action space; the index of a perturbation is its action id."""

    def __init__(self, perturbations: Sequence[Perturbation]) -> None:
        names = [p.name for p in perturbations]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate perturbation names: {names}")
        self._items = tuple(perturbations)

    @classmethod
    def from_config(
        cls, cfg: PerturbationConfig, synonyms: SynonymLookup | None = None
    ) -> PerturbationSet:
        lookup = synonyms or WordNetSynonyms(max_synsets=cfg.synonym_max_synsets)
        return cls(
            [
                Noop(),
                Typo(cfg.typo_prob),
                Leetspeak(cfg.leet_prob),
                Whitespace(cfg.whitespace_prob),
                Homoglyph(cfg.homoglyph_prob),
                Synonym(cfg.synonym_max_replacements, lookup),
                Emoji(cfg.emoji_insert_count),
            ]
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, action: int) -> Perturbation:
        return self._items[action]

    def apply(self, text: str, action: int, rng: random.Random) -> str:
        return self._items[action].apply(text, rng)
