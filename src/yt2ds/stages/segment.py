"""Turn speech regions plus speaker turns into final chunk boundaries.

The one invariant everything here serves: **a clip starts and ends in a real
pause.** A clip cut mid-word teaches a TTS model to clip its own words, and a
clip whose edge holds a sliver of the neighbouring sentence teaches it to
hallucinate one -- and once a transcript is involved, the text of that sliver
lands in the wrong clip.

Rules, in the order they are applied:

1. Every VAD speech region is cut at each speaker-turn boundary and, with
   ``diarize.drop_overlap``, at each overlapped-speech boundary. Each piece
   takes the speaker covering most of it. A short piece no turn covers (VAD
   and the diarizer never agree to the millisecond) goes to the speaker on
   both sides of it; overlapped or genuinely unattributed pieces belong to
   nobody and are dropped.
2. Same-speaker pieces separated by at most ``merge_gap`` of silence -- with no
   other voice inside the gap -- are merged into one candidate.
3. A candidate edge is only allowed where it sits at a VAD region boundary
   with at least ``min_cut_pause`` of silence to the nearest speech of *any*
   speaker. An edge that fails (a turn change or overlap mid-phrase) is pulled
   back to the nearest pause that passes, discarding the partial phrase rather
   than cutting it.
4. A candidate whose speech runs past ``max_duration`` is split recursively at
   its longest internal pause, preferring central ones. Only pauses that are
   themselves valid cut points qualify; a stretch with none is dropped rather
   than chopped on a grid.
5. Each clip gets up to ``pad`` seconds of the surrounding silence on each
   side, but never more than half the gap to the neighbouring speech -- so no
   two clips share audio and no clip's padding reaches into someone's words.
6. Clips outside [``min_duration``, ``max_duration + 2 * pad``] are dropped.
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from .diarize import Diarization
from .vad import Region

log = logging.getLogger(__name__)

_EPS = 1e-6


@dataclass
class Chunk:
    """One candidate dataset clip.

    ``start``/``end`` are the clip's edges in source seconds, padding included;
    ``speech_start``/``speech_end`` bound the speech inside it.
    """

    index: int
    start: float
    end: float
    speaker: str
    speech_start: float = 0.0
    speech_end: float = 0.0
    # Populated by later stages.
    scores: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    text_yt: str = ""
    text_asr: str = ""
    text_verify: str = ""
    text_source: str = ""
    reject_reason: str | None = None

    def __post_init__(self) -> None:
        if self.speech_end <= self.speech_start:
            self.speech_start, self.speech_end = self.start, self.end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_duration(self) -> float:
        return self.speech_end - self.speech_start

    def reject(self, reason: str) -> "Chunk":
        # First reason wins, so the report shows the earliest cause rather than
        # whichever stage happened to run last.
        if self.reject_reason is None:
            self.reject_reason = reason
        return self


@dataclass
class _Piece:
    """A slice of one VAD region holding a single speaker (or nobody)."""

    start: float
    end: float
    speaker: str | None
    # Whether each edge coincides with its VAD region's edge, i.e. borders
    # silence rather than more speech.
    open_left: bool
    open_right: bool


class _Silence:
    """Answers "how much silence surrounds this point" against all speech.

    Speech is everything VAD heard plus diarized overlaps and other speakers'
    turns, so a pause that VAD missed a voice in still counts as occupied.
    """

    def __init__(self, spans: list[tuple[float, float]], duration: float) -> None:
        merged: list[list[float]] = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1] + _EPS:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.starts = [s for s, _ in merged]
        self.ends = [e for _, e in merged]
        self.duration = duration

    def _containing(self, t: float) -> int:
        """Index of the last interval starting at or before ``t``, or -1."""
        return bisect.bisect_right(self.starts, t + _EPS) - 1

    def before(self, t: float) -> float:
        """Seconds of silence immediately before ``t``; infinite if no speech precedes it."""
        i = self._containing(t)
        if i >= 0 and self.ends[i] >= t - _EPS:
            if self.starts[i] < t - _EPS:
                return 0.0  # speech runs straight into t
            i -= 1
        return max(0.0, t - self.ends[i]) if i >= 0 else math.inf

    def after(self, t: float) -> float:
        """Seconds of silence immediately after ``t``; infinite if no speech follows it."""
        i = self._containing(t)
        if i >= 0 and self.ends[i] > t + _EPS:
            return 0.0  # speech continues past t
        return max(0.0, self.starts[i + 1] - t) if i + 1 < len(self.starts) else math.inf


def build(
    regions: list[Region],
    diarization: Diarization,
    cfg: Config,
    duration: float,
) -> list[Chunk]:
    """Produce ordered, speaker-labelled chunks whose edges all sit in silence."""
    if not regions:
        return []
    seg = cfg.segment
    duration = duration if duration > 0 else max(r.end for r in regions)

    holes = list(diarization.overlaps) if cfg.diarize.drop_overlap else []
    pieces = _pieces(sorted(regions, key=lambda r: r.start), diarization, holes, seg.merge_gap)

    # Silence is judged per speaker: VAD speech and overlaps count against
    # everyone, but a speaker's own diarized turn -- which usually starts a
    # little before VAD hears them -- does not count against themselves.
    base = [(r.start, r.end) for r in regions] + [(o.start, o.end) for o in diarization.overlaps]
    silences: dict[str, _Silence] = {}

    def silence_for(speaker: str) -> _Silence:
        if speaker not in silences:
            others = [(t.start, t.end) for t in diarization.turns if t.speaker != speaker]
            silences[speaker] = _Silence(base + others, duration)
        return silences[speaker]

    stats = {"trimmed": 0.0, "unsplittable": 0.0, "length": 0.0}
    chunks: list[Chunk] = []
    for run in _runs(pieces, diarization, seg.merge_gap):
        silence = silence_for(run[0].speaker or "")
        before = sum(p.end - p.start for p in run)
        run = _trim_to_pauses(run, silence, seg.min_cut_pause)
        stats["trimmed"] += before - sum(p.end - p.start for p in run)
        for part in _split(run, seg, stats):
            chunk = _pad(part, silence, seg.pad, duration)
            if not seg.min_duration <= chunk.duration <= seg.max_duration + 2 * seg.pad + _EPS:
                stats["length"] += chunk.speech_duration
                continue
            chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        chunk.index = i
    log.info(
        "segmentation: %d clips; dropped %.1fs mid-phrase at turn/overlap edges, "
        "%.1fs with no usable pause, %.1fs out of duration bounds",
        len(chunks),
        stats["trimmed"],
        stats["unsplittable"],
        stats["length"],
    )
    return chunks


# ---------------------------------------------------------------- pieces
def _pieces(
    regions: list[Region],
    diarization: Diarization,
    holes: list[Region],
    max_sliver: float,
) -> list[_Piece]:
    """Cut each region at turn and hole boundaries and label every slice."""
    turns = sorted(diarization.turns, key=lambda t: t.start)
    single = not turns
    out: list[_Piece] = []
    for region in regions:
        points = {region.start, region.end}
        for boundary in [p for t in turns for p in (t.start, t.end)] + [p for h in holes for p in (h.start, h.end)]:
            if region.start < boundary < region.end:
                points.add(boundary)
        ordered = sorted(points)

        slices: list[_Piece] = []
        for lo, hi in zip(ordered, ordered[1:]):
            if hi - lo <= _EPS:
                continue
            if any(h.start <= (lo + hi) / 2 < h.end for h in holes):
                speaker = None
            else:
                # "" marks speech no turn covers; _attribute_slivers decides.
                speaker = "SPEAKER_00" if single else (_dominant_speaker(lo, hi, turns) or "")
            slices.append(_Piece(lo, hi, speaker, lo == region.start, hi == region.end))
        _attribute_slivers(slices, max_sliver)

        # A third speaker's turn boundary can split a slice without changing
        # who dominates it; those halves are one piece of speech.
        for piece in slices:
            if out and out[-1].speaker == piece.speaker and abs(out[-1].end - piece.start) <= _EPS:
                out[-1].end = piece.end
                out[-1].open_right = piece.open_right
            else:
                out.append(piece)
    return out


def _attribute_slivers(slices: list[_Piece], max_sliver: float) -> None:
    """Give short unattributed speech to the speaker on both sides of it.

    Diarizer turns and VAD regions never line up exactly: VAD hears a phrase
    start a little before the turn does, and a diarizer leaves small gaps
    between one speaker's turns. Those slivers are that speaker's speech. A
    longer stretch, or one between two different speakers, stays nobody's.
    """
    for i, piece in enumerate(slices):
        if piece.speaker != "" or piece.end - piece.start > max_sliver:
            continue
        neighbours = {
            s.speaker
            for s in (slices[i - 1] if i > 0 else None, slices[i + 1] if i + 1 < len(slices) else None)
            if s is not None and s.speaker
        }
        if len(neighbours) == 1:
            piece.speaker = neighbours.pop()
    for piece in slices:
        if piece.speaker == "":
            piece.speaker = None


def _dominant_speaker(start: float, end: float, turns: list) -> str | None:
    """Speaker covering the most of ``[start, end)``, or None if none covers half."""
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
    # Speech the diarizer did not attribute is not guessed onto anyone.
    if best is None or best_overlap < (end - start) * 0.5:
        return None
    return best


def _runs(pieces: list[_Piece], diarization: Diarization, merge_gap: float) -> list[list[_Piece]]:
    """Group consecutive same-speaker pieces separated only by short silence."""
    runs: list[list[_Piece]] = []
    current: list[_Piece] = []
    for piece in pieces:
        if piece.speaker is None:
            if current:
                runs.append(current)
            current = []
            continue
        if current:
            last = current[-1]
            gap = piece.start - last.end
            joinable = (
                piece.speaker == last.speaker
                and gap <= merge_gap
                and last.open_right
                and piece.open_left
                and not _foreign_voice(last.end, piece.start, piece.speaker, diarization)
            )
            if not joinable:
                runs.append(current)
                current = []
        current.append(piece)
    if current:
        runs.append(current)
    return runs


def _foreign_voice(start: float, end: float, speaker: str, diarization: Diarization) -> bool:
    """True when overlapped speech or another speaker's turn falls in ``(start, end)``."""
    if any(o.start < end and o.end > start for o in diarization.overlaps):
        return True
    return any(t.start < end and t.end > start and t.speaker != speaker for t in diarization.turns)


# ------------------------------------------------------------------ edges
def _left_ok(piece: _Piece, silence: _Silence, min_pause: float) -> bool:
    return piece.open_left and silence.before(piece.start) >= min_pause


def _right_ok(piece: _Piece, silence: _Silence, min_pause: float) -> bool:
    return piece.open_right and silence.after(piece.end) >= min_pause


def _trim_to_pauses(run: list[_Piece], silence: _Silence, min_pause: float) -> list[_Piece]:
    """Pull each edge back to the nearest valid pause, dropping partial phrases."""
    lo, hi = 0, len(run) - 1
    while lo <= hi and not _left_ok(run[lo], silence, min_pause):
        lo += 1
    while hi >= lo and not _right_ok(run[hi], silence, min_pause):
        hi -= 1
    return run[lo : hi + 1]


# -------------------------------------------------------------- splitting
def _split(run: list[_Piece], seg, stats: dict[str, float]) -> list[list[_Piece]]:
    """Split a run at its longest valid internal pause until each part fits."""
    if not run:
        return []
    span = run[-1].end - run[0].start
    if span <= seg.max_duration:
        return [run]

    centre = (run[0].start + run[-1].end) / 2.0
    best: int | None = None
    best_key: tuple = ()
    for i in range(len(run) - 1):
        gap = run[i + 1].start - run[i].end
        if gap < seg.min_cut_pause:
            continue
        left = run[i].end - run[0].start
        right = run[-1].end - run[i + 1].start
        # Prefer a split leaving both halves long enough to keep, then the
        # longest pause (sentence ends pause longest), then the most central.
        key = (
            left >= seg.min_duration and right >= seg.min_duration,
            round(gap, 2),
            -abs((run[i].end + run[i + 1].start) / 2.0 - centre),
        )
        if best is None or key > best_key:
            best, best_key = i, key

    if best is None:
        stats["unsplittable"] += sum(p.end - p.start for p in run)
        return []
    return _split(run[: best + 1], seg, stats) + _split(run[best + 1 :], seg, stats)


# ---------------------------------------------------------------- padding
def _pad(run: list[_Piece], silence: _Silence, pad: float, duration: float) -> Chunk:
    """Extend into the surrounding silence, at most half-way to the next voice."""
    speech_start, speech_end = run[0].start, run[-1].end
    left = min(pad, silence.before(speech_start) / 2.0, speech_start)
    right = min(pad, silence.after(speech_end) / 2.0, max(0.0, duration - speech_end))
    chunk = Chunk(
        index=0,
        start=speech_start - left,
        end=speech_end + right,
        speaker=run[0].speaker or "SPEAKER_00",
        speech_start=speech_start,
        speech_end=speech_end,
    )
    chunk.scores["voiced_seconds"] = round(sum(p.end - p.start for p in run), 3)
    chunk.scores["max_internal_pause"] = round(
        max((b.start - a.end for a, b in zip(run, run[1:])), default=0.0), 3
    )
    return chunk
