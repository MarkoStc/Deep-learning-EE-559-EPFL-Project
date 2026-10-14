"""Masking of slurs in anything printed or logged.

The patterns cover common spellings and leetspeak variants of a short list of
slurs. They protect log readers; they are not a content filter and are not used
by the classifier.
"""

from __future__ import annotations

import re

_SLUR_PATTERNS = (
    r"\bn[i1!|]gg[e3a@][r5sz]?\b",
    r"\bf[a@4]gg?[o0][t7]\b",
    r"\bk[i1!]ke\b",
    r"\bch[i1!]nk\b",
    r"\bsp[i1!]c\b",
    r"\btr[a@4]nn[yi1!e3]\b",
    r"\bwh[o0]re\b",
    r"\bb[i1!]tch\b",
)
_COMPILED = tuple(re.compile(p, re.IGNORECASE) for p in _SLUR_PATTERNS)

MAX_DISPLAY_CHARS = 120


def sanitize_text(text: str, mask_char: str = "█") -> str:
    """Replace every match of a known slur with ``mask_char`` of the same length."""
    for pattern in _COMPILED:
        text = pattern.sub(lambda m: mask_char * len(m.group()), text)
    return text


def format_example(
    text: str, label: int, action: str = "", sanitize: bool = True, mask_char: str = "█"
) -> str:
    """One-line display of an example: ``[HATE] (leetspeak) text...``."""
    display = sanitize_text(text, mask_char) if sanitize else text
    if len(display) > MAX_DISPLAY_CHARS:
        display = display[: MAX_DISPLAY_CHARS - 3] + "..."
    parts = ["[HATE]" if label == 1 else "[SAFE]"]
    if action:
        parts.append(f"({action})")
    parts.append(display)
    return " ".join(parts)
