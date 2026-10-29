"""Explicit, reviewable English text expansion for the speech editor."""

import re
import threading


_NORMALIZER_LOCK = threading.Lock()
_PHONE_NUMBER = re.compile(
    r"(?<![\w.])(?:\+?\d{1,3}[ .-])?"
    r"(?:(?:\(\d{3}\)[ .-]?|\d{3}[ .-]))?\d{3}[ .-]\d{4}(?!\d)"
)


def expand_english_text(text, *, max_length):
    """Return a proposal only; callers decide whether to apply the result."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Enter the text you want to expand.")
    if len(text) > max_length:
        raise ValueError(f"Keep the text under {max_length} characters before expanding it.")
    if _PHONE_NUMBER.search(stripped):
        # The upstream normalizer can treat a phone hyphen as subtraction.
        raise ValueError("Write phone numbers as you want them spoken, then review the text before generating.")
    from dots_tts.utils.text import normalize_english_text

    # The first call builds the cached normalizer. Keep concurrent first calls
    # from compiling the same files, and never invoke language detection.
    with _NORMALIZER_LOCK:
        normalized = normalize_english_text(stripped)
    if not normalized:
        raise ValueError("The expanded text is empty. Keep or edit your original text.")
    if len(normalized) > max_length:
        raise ValueError(f"Expanded text has {len(normalized)} characters; shorten the original to fit the {max_length}-character limit.")
    return normalized
