from __future__ import annotations

from red_queen.text_safety import MAX_DISPLAY_CHARS, format_example, sanitize_text


def test_masks_leetspeak_variant_with_same_length() -> None:
    masked = sanitize_text("you b1tch", mask_char="#")
    assert masked == "you #####"


def test_leaves_clean_text_alone() -> None:
    assert sanitize_text("a perfectly normal sentence") == "a perfectly normal sentence"


def test_format_example_truncates_and_labels() -> None:
    line = format_example("x" * 500, label=1, action="typo")
    assert line.startswith("[HATE] (typo) ")
    assert line.endswith("...")
    assert len(line) == len("[HATE] (typo) ") + MAX_DISPLAY_CHARS


def test_format_example_can_skip_sanitising() -> None:
    assert format_example("whore", label=0, sanitize=False) == "[SAFE] whore"
