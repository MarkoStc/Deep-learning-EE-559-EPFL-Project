"""
Perturbation functions (the Evader's action space) and tokenization utilities.

Action space:
  0 = noop
  1 = typo injection
  2 = leetspeak
  3 = whitespace manipulation
  4 = homoglyph substitution
  5 = synonym swap
  6 = emoji insertion

All perturbations operate on already-labeled text. No new hateful content
is generated — only surface-level transformations.
"""

import re
import random
import string
import numpy as np

# Action names for logging / display
ACTION_NAMES = [
    "noop",
    "typo",
    "leetspeak",
    "whitespace",
    "homoglyph",
    "synonym",
    "emoji",
]

NUM_ACTIONS = len(ACTION_NAMES)


# ── 0. Noop ─────────────────────────────────────────────────

def perturb_noop(text: str, cfg: dict = None) -> str:
    return text


# ── 1. Typo injection ───────────────────────────────────────

def perturb_typo(text: str, cfg: dict = None) -> str:
    """Randomly swap, delete, or duplicate a character in some words."""
    prob = (cfg or {}).get("perturbation", {}).get("typo_prob", 0.3)
    words = text.split()
    result = []
    for w in words:
        if len(w) > 2 and random.random() < prob:
            idx = random.randint(1, len(w) - 2)  # avoid first/last char
            op = random.choice(["swap", "delete", "duplicate"])
            if op == "swap" and idx < len(w) - 1:
                w = w[:idx] + w[idx + 1] + w[idx] + w[idx + 2:]
            elif op == "delete":
                w = w[:idx] + w[idx + 1:]
            else:  # duplicate
                w = w[:idx] + w[idx] + w[idx:]
        result.append(w)
    return " ".join(result)


# ── 2. Leetspeak ────────────────────────────────────────────

_LEET_MAP = {
    "a": "4", "e": "3", "i": "1", "o": "0",
    "s": "5", "t": "7", "l": "1", "g": "9",
}

def perturb_leetspeak(text: str, cfg: dict = None) -> str:
    """Replace some letters with leet equivalents."""
    prob = (cfg or {}).get("perturbation", {}).get("leet_prob", 0.4)
    result = []
    for ch in text:
        if ch.lower() in _LEET_MAP and random.random() < prob:
            result.append(_LEET_MAP[ch.lower()])
        else:
            result.append(ch)
    return "".join(result)


# ── 3. Whitespace manipulation ──────────────────────────────

_ZERO_WIDTH = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space
]

def perturb_whitespace(text: str, cfg: dict = None) -> str:
    """Insert zero-width characters or extra spaces into random words."""
    words = text.split()
    result = []
    for w in words:
        if len(w) > 3 and random.random() < 0.3:
            idx = random.randint(1, len(w) - 1)
            char = random.choice(_ZERO_WIDTH)
            w = w[:idx] + char + w[idx:]
        result.append(w)
    return " ".join(result)


# ── 4. Homoglyph substitution ──────────────────────────────

_HOMOGLYPHS = {
    "a": "а",   # cyrillic а
    "c": "с",   # cyrillic с
    "e": "е",   # cyrillic е
    "o": "о",   # cyrillic о
    "p": "р",   # cyrillic р
    "x": "х",   # cyrillic х
    "s": "ѕ",   # cyrillic ѕ
    "i": "і",   # cyrillic і
}

def perturb_homoglyph(text: str, cfg: dict = None) -> str:
    """Replace some ASCII characters with visually identical Unicode."""
    prob = (cfg or {}).get("perturbation", {}).get("homoglyph_prob", 0.3)
    result = []
    for ch in text:
        if ch.lower() in _HOMOGLYPHS and random.random() < prob:
            result.append(_HOMOGLYPHS[ch.lower()])
        else:
            result.append(ch)
    return "".join(result)


# ── 5. Synonym swap ─────────────────────────────────────────

def _get_synonyms(word: str) -> list:
    """Get WordNet synonyms for a word. Returns empty list if nltk unavailable."""
    try:
        from nltk.corpus import wordnet
        synsets = wordnet.synsets(word)
        synonyms = set()
        for syn in synsets[:3]:  # limit search
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    synonyms.add(name)
        return list(synonyms)
    except Exception:
        return []


def perturb_synonym(text: str, cfg: dict = None) -> str:
    """Replace 1-2 words with WordNet synonyms."""
    max_replace = (cfg or {}).get("perturbation", {}).get(
        "synonym_max_replacements", 2
    )
    words = text.split()
    # Pick candidate words (longer words more likely to have synonyms)
    candidates = [
        (i, w) for i, w in enumerate(words)
        if len(w) > 3 and w.isalpha()
    ]
    random.shuffle(candidates)

    replaced = 0
    for idx, word in candidates:
        if replaced >= max_replace:
            break
        syns = _get_synonyms(word.lower())
        if syns:
            words[idx] = random.choice(syns)
            replaced += 1

    return " ".join(words)


# ── 6. Emoji insertion ──────────────────────────────────────

_BENIGN_EMOJIS = [
    "😊", "👍", "❤️", "🙂", "✨", "🌟", "💯", "🔥",
    "😎", "🎉", "👏", "💪", "🙏", "😇", "🌈", "☀️",
]

def perturb_emoji(text: str, cfg: dict = None) -> str:
    """Insert benign emojis at random positions."""
    count = (cfg or {}).get("perturbation", {}).get("emoji_insert_count", 1)
    words = text.split()
    for _ in range(min(count, len(words))):
        pos = random.randint(0, len(words))
        em = random.choice(_BENIGN_EMOJIS)
        words.insert(pos, em)
    return " ".join(words)


# ── Dispatch ────────────────────────────────────────────────

PERTURBATION_FNS = [
    perturb_noop,
    perturb_typo,
    perturb_leetspeak,
    perturb_whitespace,
    perturb_homoglyph,
    perturb_synonym,
    perturb_emoji,
]

def apply_perturbation(text: str, action: int, cfg: dict = None) -> str:
    """Apply perturbation by action index."""
    return PERTURBATION_FNS[action](text, cfg)


# ── Text feature extraction (for Evader state) ─────────────

def extract_text_features(text: str, guardian_prob: float = 0.5) -> np.ndarray:
    """
    Extract a simple feature vector from text for the Evader's state.
    Features:
      0: normalized char length
      1: normalized word count
      2: fraction of uppercase chars
      3: fraction of punctuation chars
      4: has emoji (0/1)
      5: has URL (0/1)
      6: avg word length (normalized)
      7: guardian predicted hate probability
    """
    import emoji as emoji_lib

    char_len = min(len(text) / 500.0, 1.0)
    words = text.split()
    word_count = min(len(words) / 100.0, 1.0)
    upper_frac = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    punct_frac = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    has_emoji = 1.0 if emoji_lib.emoji_count(text) > 0 else 0.0
    has_url = 1.0 if re.search(r"https?://", text) else 0.0
    avg_word_len = np.mean([len(w) for w in words]) / 15.0 if words else 0.0
    avg_word_len = min(avg_word_len, 1.0)

    return np.array([
        char_len, word_count, upper_frac, punct_frac,
        has_emoji, has_url, avg_word_len, guardian_prob,
    ], dtype=np.float32)


# ── NLTK bootstrap ──────────────────────────────────────────

def ensure_nltk_data():
    """Download required NLTK data if not present."""
    import nltk
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
