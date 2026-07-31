"""Parse YouTube subtitle tracks into a flat, de-duplicated word list.

The hard part is YouTube's *rolling* auto-captions. To animate text appearing
line by line, the auto-caption track re-emits words it has already shown:

    cue 1:  مرحبا بكم
    cue 2:  مرحبا بكم في
    cue 3:  مرحبا بكم في هذا

Parsed naively that triples the transcript, which would wreck forced alignment
and every downstream text metric. Two defences are applied, in order:

1. When word-level timings exist (json3 ``tOffsetMs``, or VTT's inline
   ``<00:00:01.500><c>`` tags), the repeats carry the *same absolute
   timestamp* as the original, so exact de-duplication on (time, word) removes
   them and leaves genuine repeated words -- "لا لا لا" -- intact, because
   those occur at different times.
2. When only cue-level timings exist, any leading run of words in a cue that
   repeats the tail of the previous cue is dropped.

Note that the timings here barely matter downstream: the pipeline re-derives
word timings with CTC forced alignment. What must be right is the *text
sequence*.
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..arabic import normalize, strip_caption_annotations

log = logging.getLogger(__name__)

# Same absolute time to within this tolerance counts as the same word instance.
_TIME_EPS = 0.06
# Longest tail/head overlap considered when de-duplicating cue-level captions.
_MAX_OVERLAP_WORDS = 40


@dataclass
class SubWord:
    text: str
    start: float
    end: float


@dataclass
class Subtitles:
    words: list[SubWord] = field(default_factory=list)
    kind: str = ""  # "yt_manual" | "yt_auto"
    lang: str = ""
    has_word_timings: bool = False

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def __bool__(self) -> bool:
        return bool(self.words)


def parse(path: Path, kind: str = "", lang: str = "") -> Subtitles:
    """Parse a subtitle file by extension, returning de-duplicated words."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in (".json3", ".srv3", ".json"):
            words, word_timed = _parse_json3(path)
        elif suffix in (".vtt", ".srt"):
            words, word_timed = _parse_vtt(path)
        else:
            log.warning("unsupported subtitle format: %s", path.name)
            return Subtitles(kind=kind, lang=lang)
    except Exception as exc:  # noqa: BLE001 - never let a bad sub file kill a video
        log.error("failed to parse %s: %s", path.name, exc)
        return Subtitles(kind=kind, lang=lang)

    # Only auto-captions roll. Applying block-collapse to a human-written track
    # could only ever delete a legitimately repeated phrase, so don't.
    words = _dedupe(words, word_timed, allow_block_collapse=kind != "yt_manual")
    return Subtitles(words=words, kind=kind, lang=lang, has_word_timings=word_timed)


# --------------------------------------------------------------------------
# json3
# --------------------------------------------------------------------------
def _parse_json3(path: Path) -> tuple[list[SubWord], bool]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events") or []

    raw: list[SubWord] = []
    word_timed = False
    for event in events:
        start_ms = event.get("tStartMs")
        if start_ms is None:
            continue
        duration_ms = event.get("dDurationMs") or 0
        segs = event.get("segs") or []
        for seg in segs:
            text = seg.get("utf8", "")
            if not text or not text.strip():
                continue
            offset = seg.get("tOffsetMs")
            if offset:
                word_timed = True
            start = (start_ms + (offset or 0)) / 1000.0
            for token in text.split():
                raw.append(SubWord(text=token, start=start, end=start))
        # Give the last word of the event the event's end time.
        if raw and duration_ms:
            raw[-1].end = max(raw[-1].start, (start_ms + duration_ms) / 1000.0)

    _fill_end_times(raw)
    return raw, word_timed


# --------------------------------------------------------------------------
# WebVTT / SRT
# --------------------------------------------------------------------------
_VTT_CUE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3})[^\n]*\n"
    r"(?P<text>(?:.*\n?)*?)(?=\n\s*\n|\n\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\Z)"
)
# Inline word timing that auto-captions carry: <00:00:01.500><c> word</c>
_INLINE = re.compile(r"<(\d{1,2}:\d{2}:\d{2}[.,]\d{3})>")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def _timestamp(value: str) -> float:
    value = value.replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_vtt(path: Path) -> tuple[list[SubWord], bool]:
    content = path.read_text(encoding="utf-8", errors="replace")
    raw: list[SubWord] = []
    word_timed = False

    for match in _VTT_CUE.finditer(content):
        cue_start = _timestamp(match.group("start"))
        cue_end = _timestamp(match.group("end"))
        body = html.unescape(match.group("text"))
        if not body.strip():
            continue

        if _INLINE.search(body):
            word_timed = True
            raw.extend(_parse_inline_cue(body, cue_start, cue_end))
        else:
            text = _TAG.sub("", body)
            tokens = text.split()
            if not tokens:
                continue
            step = (cue_end - cue_start) / max(len(tokens), 1)
            for i, token in enumerate(tokens):
                start = cue_start + i * step
                raw.append(SubWord(text=token, start=start, end=start + step))

    _fill_end_times(raw)
    return raw, word_timed


def _parse_inline_cue(body: str, cue_start: float, cue_end: float) -> list[SubWord]:
    """Split a cue on its inline ``<timestamp>`` markers."""
    words: list[SubWord] = []
    pieces = _INLINE.split(body)
    # split() yields [text, ts, text, ts, text, ...]
    current = cue_start
    for index, piece in enumerate(pieces):
        if index % 2 == 1:
            current = _timestamp(piece)
            continue
        text = _TAG.sub("", piece)
        for token in text.split():
            words.append(SubWord(text=token, start=current, end=current))
    for word in words:
        word.end = max(word.end, word.start)
    if words:
        words[-1].end = max(words[-1].start, cue_end)
    return words


# --------------------------------------------------------------------------
# de-duplication
# --------------------------------------------------------------------------
def _fill_end_times(words: list[SubWord]) -> None:
    """Give every word an end time: the next word's start, or its own start."""
    for i, word in enumerate(words):
        if word.end > word.start:
            continue
        if i + 1 < len(words):
            word.end = max(word.start, words[i + 1].start)
        else:
            word.end = word.start


def _dedupe(words: list[SubWord], word_timed: bool, allow_block_collapse: bool = True) -> list[SubWord]:
    cleaned: list[SubWord] = []
    for word in words:
        text = strip_caption_annotations(word.text).strip()
        if text:
            cleaned.append(SubWord(text=text, start=word.start, end=word.end))

    if not cleaned:
        return []

    if word_timed:
        # Exact-timestamp de-duplication is safe for any track: two real
        # utterances of a word never share a timestamp.
        return _dedupe_by_time(cleaned)
    if allow_block_collapse:
        return _dedupe_rolling_cues(cleaned)
    return cleaned


def _dedupe_by_time(words: list[SubWord]) -> list[SubWord]:
    """Drop words repeated at the same absolute timestamp.

    Genuine repetition survives because two utterances of the same word never
    share a timestamp.
    """
    words = sorted(words, key=lambda w: (w.start, w.end))
    kept: list[SubWord] = []
    seen: dict[str, float] = {}
    for word in words:
        key = normalize(word.text)
        if not key:
            continue
        previous = seen.get(key)
        if previous is not None and abs(word.start - previous) <= _TIME_EPS:
            continue
        seen[key] = word.start
        kept.append(word)
    return kept


def _dedupe_rolling_cues(words: list[SubWord]) -> list[SubWord]:
    """Remove leading words that repeat the tail of what was already emitted.

    Applied greedily as words arrive, so a cue that restates the previous
    cue's ending contributes only its new words.
    """
    kept: list[SubWord] = []
    for word in words:
        kept.append(word)

    # Work over the flat sequence: find and collapse runs where a block of
    # words immediately repeats the block before it.
    result: list[SubWord] = []
    index = 0
    while index < len(kept):
        collapsed = False
        # Try the longest plausible repeat first so nested repeats collapse fully.
        max_span = min(_MAX_OVERLAP_WORDS, len(result), len(kept) - index)
        for span in range(max_span, 1, -1):
            tail = [normalize(w.text) for w in result[-span:]]
            head = [normalize(w.text) for w in kept[index : index + span]]
            if tail == head:
                index += span
                collapsed = True
                break
        if not collapsed:
            result.append(kept[index])
            index += 1
    return result
