"""Turn speech regions plus speaker turns into final chunk boundaries.

Rules, in the order they are applied:

1. Overlapped speech is cut out entirely. A clip containing two voices teaches
   a voice-cloning model to blend them, so it is worth losing the audio.
2. Speech regions are split at speaker turns -- a chunk never spans two
   speakers.
3. Same-speaker fragments separated by less than ``merge_gap`` are merged, so
   a natural sentence is not chopped at every breath.
4. Anything longer than ``max_duration`` is split recursively at its deepest
   internal pause.
5. Chunks are padded by ``pad`` seconds on each side, then anything still
   outside [``min_duration``, ``max_duration``] is dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from .diarize import Diarization
from .vad import Region, subtract

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    """One candidate dataset clip."""

    index: int
    start: float
    end: float
    speaker: str
    # Populated by later stages.
    scores: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    text_yt: str = ""
    text_cohere: str = ""
    text_source: str = ""
    reject_reason: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def reject(self, reason: str) -> "Chunk":
        # First reason wins, so the report shows the earliest cause rather than
        # whichever stage happened to run last.
        if self.reject_reason is None:
            self.reject_reason = reason
        return self


def build(
    regions: list[Region],
    diarization: Diarization,
    cfg: Config,
    duration: float,
) -> list[Chunk]:
    """Produce ordered, speaker-labelled chunks."""
    if not regions:
        return []

    speech = list(regions)
    if cfg.diarize.drop_overlap and diarization.overlaps:
        before = sum(r.duration for r in speech)
        speech = subtract(speech, diarization.overlaps)
        after = sum(r.duration for r in speech)
        log.info("overlap removal dropped %.1fs of speech", before - after)

    labelled = _assign_speakers(speech, diarization)
    merged = _merge(labelled, cfg.segment.merge_gap)

    chunks: list[Chunk] = []
    for start, end, speaker in merged:
        for lo, hi in _split_long(start, end, speech, cfg):
            chunks.append(Chunk(index=0, start=lo, end=hi, speaker=speaker))

    padded = _pad_and_filter(chunks, cfg, duration)
    for i, chunk in enumerate(padded):
        chunk.index = i
    return padded


def _assign_speakers(regions: list[Region], diarization: Diarization) -> list[tuple[float, float, str]]:
    """Split each speech region wherever the active speaker changes.

    A region is cut at every diarization boundary that falls inside it; each
    resulting piece takes the speaker whose turn covers the most of it.
    """
    if not diarization.turns:
        return [(r.start, r.end, "SPEAKER_00") for r in regions]

    turns = sorted(diarization.turns, key=lambda t: t.start)
    out: list[tuple[float, float, str]] = []

    for region in regions:
        boundaries = {region.start, region.end}
        for turn in turns:
            for point in (turn.start, turn.end):
                if region.start < point < region.end:
                    boundaries.add(point)
        points = sorted(boundaries)

        for lo, hi in zip(points, points[1:]):
            if hi - lo <= 1e-6:
                continue
            speaker = _dominant_speaker(lo, hi, turns)
            if speaker is not None:
                out.append((lo, hi, speaker))
    return out


def _dominant_speaker(start: float, end: float, turns: list) -> str | None:
    """Speaker covering the most of ``[start, end)``, or None if none does."""
    best: str | None = None
    best_overlap = 0.0
    for turn in turns:
        if turn.end <= start:
            continue
        if turn.start >= end:
            break
        overlap = min(turn.end, end) - max(turn.start, start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = turn.speaker
    # Require the speaker to actually cover half the slice; otherwise this is
    # speech the diarizer did not attribute and we should not guess.
    if best is None or best_overlap < (end - start) * 0.5:
        return None
    return best


def _merge(pieces: list[tuple[float, float, str]], max_gap: float) -> list[tuple[float, float, str]]:
    """Join adjacent same-speaker pieces separated by a short gap."""
    if not pieces:
        return []
    ordered = sorted(pieces, key=lambda p: p[0])
    merged = [list(ordered[0])]
    for start, end, speaker in ordered[1:]:
        prev = merged[-1]
        if speaker == prev[2] and start - prev[1] <= max_gap:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end, speaker])
    return [(m[0], m[1], m[2]) for m in merged]


def _split_long(start: float, end: float, speech: list[Region], cfg: Config) -> list[tuple[float, float]]:
    """Recursively split a span at its deepest internal pause until it fits."""
    if end - start <= cfg.segment.max_duration:
        return [(start, end)]

    pause = _deepest_pause(start, end, speech, cfg.segment.min_split_pause)
    if pause is None:
        # No usable pause: cut on a grid so the material is not lost entirely.
        return _hard_split(start, end, cfg.segment.max_duration)

    mid_lo, mid_hi = pause
    left = _split_long(start, mid_lo, speech, cfg)
    right = _split_long(mid_hi, end, speech, cfg)
    return left + right


def _deepest_pause(start: float, end: float, speech: list[Region], min_pause: float) -> tuple[float, float] | None:
    """Longest silence strictly inside ``[start, end)``, preferring the middle.

    Silences are the gaps between VAD regions. Ties, and near-ties, are broken
    toward the centre so splitting produces balanced halves rather than a
    two-second sliver.
    """
    inside = sorted((r for r in speech if r.end > start and r.start < end), key=lambda r: r.start)
    if len(inside) < 2:
        return None

    centre = (start + end) / 2.0
    best: tuple[float, float] | None = None
    best_key = (0.0, 0.0)
    for left, right in zip(inside, inside[1:]):
        gap_lo, gap_hi = left.end, right.start
        gap = gap_hi - gap_lo
        if gap < min_pause or gap_lo <= start or gap_hi >= end:
            continue
        # Prefer long pauses, then central ones.
        key = (round(gap, 2), -abs((gap_lo + gap_hi) / 2.0 - centre))
        if key > best_key:
            best_key = key
            best = (gap_lo, gap_hi)
    return best


def _hard_split(start: float, end: float, max_duration: float) -> list[tuple[float, float]]:
    """Even split into the fewest pieces that all fit under the limit."""
    import math

    pieces = max(1, math.ceil((end - start) / max_duration))
    step = (end - start) / pieces
    return [(start + i * step, start + (i + 1) * step) for i in range(pieces)]


def _pad_and_filter(chunks: list[Chunk], cfg: Config, duration: float) -> list[Chunk]:
    """Add silence padding, clamp to the file, and drop out-of-range chunks."""
    out: list[Chunk] = []
    for chunk in chunks:
        start = max(0.0, chunk.start - cfg.segment.pad)
        end = min(duration, chunk.end + cfg.segment.pad) if duration > 0 else chunk.end + cfg.segment.pad
        if end - start < cfg.segment.min_duration:
            continue
        if end - start > cfg.segment.max_duration + 2 * cfg.segment.pad:
            continue
        chunk.start = start
        chunk.end = end
        out.append(chunk)
    return out
