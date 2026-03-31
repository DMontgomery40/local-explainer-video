"""Extract cue points from Whisper word timestamps.

Dumb text matching. Finds when specific numbers and phrases are spoken in
the audio. No scene type detection. No routing. No creativity.

Factored from projects/09-05-1954-0__codex_handrolled_artifacts_20260324_013736/assemble.py
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, asdict
from typing import Any

from .whisper_timestamps import WhisperResult, WordTimestamp


@dataclass(frozen=True)
class CuePoint:
    word: str           # The spoken word/phrase that triggered this cue
    time: float         # Start time in seconds
    cue_type: str       # "numeric_reveal", "session_marker", "keyword"
    value: str          # The data value (e.g., "54", "+52%")


# --- Spoken number utilities (factored from project scripts) ---

_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen",
}

_TENS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}


def int_to_words(n: int) -> str:
    """Convert an integer to English words (0-9999)."""
    if n < 0:
        return "negative " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("-" + _ONES[ones] if ones else "")
    if n < 1000:
        hundreds, remainder = divmod(n, 100)
        parts = [_ONES[hundreds], "hundred"]
        if remainder:
            parts.append(int_to_words(remainder))
        return " ".join(parts)
    if n < 10000:
        thousands, remainder = divmod(n, 1000)
        parts = [int_to_words(thousands), "thousand"]
        if remainder:
            parts.append(int_to_words(remainder))
        return " ".join(parts)
    return str(n)


def spoken_variants(value: float | int | str) -> list[str]:
    """Generate plausible spoken forms of a numeric value.

    Examples:
        54 -> ["fifty-four", "fifty four", "54"]
        3.2 -> ["three point two", "3.2"]
        52% -> ["fifty-two percent", "52 percent", "52%"]
    """
    variants: list[str] = []
    raw = str(value).strip()
    variants.append(raw)

    # Strip % for processing
    is_pct = raw.endswith("%")
    numeric_str = raw.rstrip("%").strip()

    try:
        num = float(numeric_str)
    except ValueError:
        return variants

    # Integer form
    if num == int(num) and abs(num) < 10000:
        word_form = int_to_words(int(abs(num)))
        if num < 0:
            word_form = "negative " + word_form
        variants.append(word_form)
        variants.append(word_form.replace("-", " "))
        if is_pct:
            variants.append(word_form + " percent")
            variants.append(word_form.replace("-", " ") + " percent")
    else:
        # Decimal: "three point two"
        parts = numeric_str.split(".")
        if len(parts) == 2:
            try:
                int_part = int(parts[0])
                dec_str = parts[1]
                int_word = int_to_words(abs(int_part))
                # Each decimal digit as a word
                dec_words = " ".join(int_to_words(int(d)) for d in dec_str)
                spoken = f"{int_word} point {dec_words}"
                if int_part < 0:
                    spoken = "negative " + spoken
                variants.append(spoken)
                variants.append(spoken.replace("-", " "))
                if is_pct:
                    variants.append(spoken + " percent")
            except ValueError:
                pass

    if is_pct:
        variants.append(numeric_str + " percent")

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in variants:
        lower = v.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(v)
    return unique


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _find_phrase_in_words(
    words: list[WordTimestamp],
    phrase: str,
) -> float | None:
    """Find a phrase in the word list and return its start time.

    Uses sliding window over consecutive words.
    """
    phrase_norm = normalize_text(phrase)
    phrase_tokens = phrase_norm.split()
    n = len(phrase_tokens)
    if n == 0:
        return None

    for i in range(len(words) - n + 1):
        window = " ".join(normalize_text(words[j].word) for j in range(i, i + n))
        if phrase_norm in window or window in phrase_norm:
            return words[i].start

    # Single-word fallback
    if n == 1:
        for w in words:
            if phrase_norm in normalize_text(w.word):
                return w.start

    return None


def extract_cue_points(
    whisper_result: WhisperResult,
    scene: dict[str, Any],
    data_pack: dict[str, Any] | None = None,
) -> list[CuePoint]:
    """Match Whisper word timestamps against narration to identify trigger points.

    Finds:
    1. Numeric values in the narration and their spoken timestamps
    2. Session markers ("Session one", "Session two", etc.)

    No scene type detection. No routing. Just text matching.
    """
    words = whisper_result.words
    if not words:
        return []

    cues: list[CuePoint] = []
    narration = str(scene.get("narration", ""))

    # Strategy 1: Extract digit-form numbers from narration text
    number_pattern = re.compile(r"\b(\d[\d,]*(?:\.\d+)?%?)\b")
    for match in number_pattern.finditer(narration):
        raw_number = match.group(1)
        variants = spoken_variants(raw_number)
        for variant in variants:
            time = _find_phrase_in_words(words, variant)
            if time is not None:
                cues.append(CuePoint(
                    word=variant,
                    time=time,
                    cue_type="numeric_reveal",
                    value=raw_number,
                ))
                break

    # Strategy 2: Search Whisper transcript for digit-form numbers
    # (Whisper often transcribes "ninety-four" as "94")
    whisper_digits = number_pattern.findall(whisper_result.text)
    for digit_str in whisper_digits:
        # Skip if we already found this value
        if any(c.value == digit_str for c in cues):
            continue
        time = _find_phrase_in_words(words, digit_str)
        if time is not None:
            cues.append(CuePoint(
                word=digit_str,
                time=time,
                cue_type="numeric_reveal",
                value=digit_str,
            ))

    # Strategy 3: Search for spelled-out numbers in the narration
    # (narrations spell out numbers per TTS rule: "ninety-four" not "94")
    spelled_number_pattern = re.compile(
        r"\b((?:negative\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
        r"(?:[\s-]+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|point|and))*"
        r"(?:\s+percent)?)\b",
        re.IGNORECASE,
    )
    for match in spelled_number_pattern.finditer(narration):
        phrase = match.group(1).strip()
        # Try to find this phrase in whisper words
        time = _find_phrase_in_words(words, phrase)
        if time is not None:
            # Skip very common/ambiguous words
            if phrase.lower() in ("one", "two", "three", "four", "five"):
                continue
            cues.append(CuePoint(
                word=phrase,
                time=time,
                cue_type="numeric_reveal",
                value=phrase,
            ))

    # Session markers
    for session_num in [1, 2, 3]:
        for phrase in [f"session {int_to_words(session_num)}", f"session {session_num}"]:
            time = _find_phrase_in_words(words, phrase)
            if time is not None:
                cues.append(CuePoint(
                    word=phrase,
                    time=time,
                    cue_type="session_marker",
                    value=str(session_num),
                ))
                break

    # Sort by time
    cues.sort(key=lambda c: c.time)

    # Deduplicate: if same value appears multiple times, keep earliest
    seen_values: set[str] = set()
    unique_cues: list[CuePoint] = []
    for cue in cues:
        key = f"{cue.cue_type}:{cue.value}"
        if key not in seen_values:
            seen_values.add(key)
            unique_cues.append(cue)

    return unique_cues


def cue_points_to_json(cues: list[CuePoint]) -> str:
    """Serialize cue points to JSON for URL query param."""
    return json.dumps([asdict(c) for c in cues], separators=(",", ":"))


def cue_points_to_url_param(cues: list[CuePoint]) -> str:
    """Serialize cue points to a URL-safe query parameter value."""
    return urllib.parse.quote(cue_points_to_json(cues))
