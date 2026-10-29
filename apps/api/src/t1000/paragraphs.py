"""Plan long requests without changing their words or model context limits.

Offsets address ``text.strip()``. They partition that source exactly, including
the whitespace between chunks; ``TextSegment.text`` removes only the surrounding
whitespace before inference. A single request within the model's existing limit
is returned unchanged so its seeded output keeps the same conditioning.
"""

import re
from dataclasses import dataclass


REQUEST_TEXT_LIMIT = 5000
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
    "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m.", "no.", "fig.",
    "inc.", "ltd.", "co.", "dept.", "approx.", "est.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
    "sep.", "sept.", "oct.", "nov.", "dec.",
}
_CLOSING = '\"\u201d\u2019\u00bb\u203a\')]} '
_PAUSES = {"start": 0.0, "sentence": 0.18, "paragraph": 0.30,
           "clause": 0.08, "word": 0.08}


def _is_abbreviation(token):
    ending = token.rstrip(_CLOSING)
    return (ending.lower() in _ABBREVIATIONS
            or re.fullmatch(r"(?:[A-Za-z]\.)+", ending) is not None)


@dataclass(frozen=True)
class TextSegment:
    text: str
    start: int
    end: int
    pause_before: float = 0.0
    boundary: str = "start"

    @property
    def pause_seconds(self):
        """Compatibility spelling: the pause precedes this segment."""
        return self.pause_before


def _boundary(token, gap):
    if "\n" in gap or "\r" in gap:
        return "paragraph"
    ending = token.rstrip(_CLOSING)
    if ending.endswith(("!", "?")):
        return "sentence"
    if ending.endswith("."):
        # Do not mistake initials or dotted abbreviations for sentence ends.
        # Decimal/version dots occur inside a token and never enter this path
        # unless a separate final full stop is present.
        if not _is_abbreviation(ending):
            return "sentence"
    if ending.endswith((",", ";", ":", "\u2014", "\u2013")):
        return "clause"
    return "word"


def plan_text(text, max_chars):
    """Prefer paragraphs and sentences, then clauses and complete words.

    Each inference text fits ``max_chars``. Adjacent source slices reconstruct
    the entire trimmed input without gaps, duplication or normalization. The
    total request is bounded separately from the per-inference context budget.
    A word longer than that budget is rejected explicitly rather than split.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter the text you want to hear.")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("The speech segment limit must be a positive integer.")
    source = text.strip()
    if len(source) > REQUEST_TEXT_LIMIT:
        raise ValueError(f"Keep each request under {REQUEST_TEXT_LIMIT} characters.")
    if len(source) <= max_chars:
        return [TextSegment(source, 0, len(source))]

    words = list(re.finditer(r"\S+", source))
    if any(word.end() - word.start() > max_chars for word in words):
        raise ValueError(
            f"A word or unbroken token exceeds the {max_chars}-character speech "
            "segment limit. Shorten it or add spaces where it should be spoken."
        )
    kinds = []
    for index, word in enumerate(words):
        following = words[index + 1].start() if index + 1 < len(words) else len(source)
        kinds.append(_boundary(word.group(), source[word.end():following]))

    segments = []
    first = 0
    preceding = "start"
    while first < len(words):
        start = words[first].start()
        last = first
        while last + 1 < len(words) and words[last + 1].end() - start <= max_chars:
            last += 1
        # Retain explicit paragraph pauses. Otherwise fill with whole sentences
        # before resorting to a clause or whitespace inside a long sentence.
        paragraph = next((i for i in range(first, last + 1)
                          if kinds[i] == "paragraph"), None)
        if paragraph is not None:
            cut = paragraph
        elif last == len(words) - 1:
            cut = last
        else:
            sentences = [i for i in range(first, last + 1) if kinds[i] == "sentence"]
            clauses = [i for i in range(first, last + 1) if kinds[i] == "clause"]
            cut = sentences[-1] if sentences else clauses[-1] if clauses else last
            # If a whitespace fallback happens to land on "Dr." or "J.",
            # leave that title/initial with the following word when it fits
            # in the next segment. Do not force a zero-length segment.
            while cut > first and kinds[cut] == "word" and _is_abbreviation(words[cut].group()):
                cut -= 1
        end = words[cut + 1].start() if cut + 1 < len(words) else len(source)
        segments.append(TextSegment(source[start:end].strip(), start, end,
                                    _PAUSES[preceding], preceding))
        preceding = kinds[cut]
        first = cut + 1
    return segments
