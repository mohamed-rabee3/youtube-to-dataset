"""Decide each chunk's canonical text, then gate on text quality.

This is where the user's core requirement lives: YouTube's subtitles are the
canonical text, but they are *verified* against the ASR backend's transcription
of the same chunk and dropped when the two disagree. For a TTS dataset that
check matters more than it would for ASR -- a wrong transcript does not merely
add noise, it teaches the model to pronounce a word as something else entirely.

Text source resolution:

* Both YouTube subtitles and ASR text present -> YouTube text wins, but only
  if the CER between them is under the gate. Both are kept in the metadata so
  the threshold can be re-tuned later without recomputing anything.
* Only ASR text (video had no Arabic subtitles) -> the ASR text is canonical
  and the CER gate does not apply. ``text_source`` records which backend
  produced it, ``google`` or ``cohere``.
* Neither -> the chunk is dropped.

The ``text_cohere`` / ``cer_yt_vs_cohere`` column names predate the Google
backend and are kept as they are so datasets built before it stay readable
against the same schema; they hold whichever backend's transcript ran.
"""

from __future__ import annotations

import logging

from ..arabic import arabic_ratio, cer, clean_for_output, script_ratio, strip_tashkeel
from ..config import Config
from .segment import Chunk

log = logging.getLogger(__name__)

#: ``text_source`` values meaning "no subtitles, the ASR transcript is canonical".
ASR_SOURCES = frozenset({"google", "cohere", "google_batch"})


def resolve_text(chunks: list[Chunk], subtitle_kind: str | None, cfg: Config) -> None:
    """Pick canonical text per chunk and apply the text-quality gates."""
    backend = (cfg.asr.backend or "google").lower()
    for chunk in chunks:
        yt = clean_for_output(chunk.text_yt)
        asr = clean_for_output(chunk.text_cohere)
        chunk.text_yt = yt
        chunk.text_cohere = asr

        if yt and asr:
            disagreement = cer(yt, asr)
            chunk.scores["cer_yt_vs_cohere"] = round(disagreement, 5)
            chunk.text = yt
            chunk.text_source = subtitle_kind or "yt"
            if disagreement > cfg.filters.max_cer_yt_vs_cohere:
                chunk.reject(f"text:cer={disagreement:.3f}")
        elif asr:
            chunk.scores["cer_yt_vs_cohere"] = None
            chunk.text = asr
            chunk.text_source = backend
        elif yt:
            # The ASR returned nothing for audio the subtitles claim has
            # speech. That disagreement is itself a signal, so do not trust
            # the text.
            chunk.scores["cer_yt_vs_cohere"] = None
            chunk.text = yt
            chunk.text_source = subtitle_kind or "yt"
            chunk.reject("text:asr_empty")
        else:
            chunk.text = ""
            chunk.text_source = "none"
            chunk.reject("text:empty")

    _apply_text_gates(chunks, cfg)


def _apply_text_gates(chunks: list[Chunk], cfg: Config) -> None:
    f = cfg.filters
    for chunk in chunks:
        if chunk.reject_reason is not None:
            continue

        text = chunk.text
        if not text:
            chunk.reject("text:empty")
            continue

        words = text.split()
        if len(words) < f.min_words:
            chunk.reject(f"text:too_short={len(words)}w")
            continue

        # Check the text against the script of the language it was decoded in,
        # so English chunks are not rejected for failing to be Arabic. Both
        # ratios are recorded, which is what makes code-switching measurable
        # after the fact.
        language = chunk.scores.get("asr_language") or "ar"
        ratio = script_ratio(text, language)
        chunk.scores["arabic_ratio"] = round(arabic_ratio(text), 4)
        chunk.scores["script_ratio"] = round(ratio, 4)
        if ratio < f.min_script_ratio:
            chunk.reject(f"text:script={language}={ratio:.2f}")
            continue

        confidence = chunk.scores.get("asr_confidence")
        asr_only = chunk.text_source in ASR_SOURCES
        # The backends score on different scales -- Google reports 0-1, Cohere
        # a mean per-token log-probability -- so the floor follows the source
        # of the text rather than being one shared number.
        floor = f.min_asr_confidence_google if chunk.text_source == "google" else f.min_asr_confidence
        if asr_only and confidence is not None and confidence < floor:
            # No subtitles to cross-check against, so the model's own
            # confidence is the only thing standing between the dataset and a
            # hallucinated transcript.
            chunk.reject(f"text:asr_confidence={confidence:.2f}")
            continue

        # Characters per second catches both halves of a timing failure: a
        # transcript far too long for its audio (words bled in from a
        # neighbouring chunk) and far too short (speech the transcript missed).
        duration = max(chunk.duration, 1e-6)
        cps = len(text.replace(" ", "")) / duration
        chunk.scores["chars_per_sec"] = round(cps, 3)
        if cps < f.min_chars_per_sec:
            chunk.reject(f"text:cps_low={cps:.1f}")
            continue
        if cps > f.max_chars_per_sec:
            chunk.reject(f"text:cps_high={cps:.1f}")
            continue

        if f.drop_boundary_clipped_words and chunk.scores.get("boundary_clipped"):
            chunk.reject("text:boundary_clipped")
            continue

        align_score = chunk.scores.get("align_score")
        if (
            not asr_only
            and align_score is not None
            and align_score < cfg.align.min_align_score
        ):
            chunk.reject(f"align:score={align_score:.2f}")


def gate_rows(rows: list[dict], cfg: Config) -> tuple[list[dict], list[dict]]:
    """Apply the text gates to metadata rows, splitting them kept / rejected.

    The chunk-level gates above run inside a video's processing, while the
    chunks are still objects in memory. The batch backend has no such moment:
    its text arrives long after the run, against rows already on disk. The
    rules are the same ones, expressed over the row schema.

    Rows whose episode has not been transcribed yet are left pending and kept,
    so a partial transcription pass never deletes audio it simply has no text
    for.
    """
    f = cfg.filters
    kept: list[dict] = []
    rejected: list[dict] = []

    for row in rows:
        # Still "pending" means this row's episode has not come back from the
        # transcription pass. Untouched, so a partial pass is resumable.
        if row.get("text_source") == "pending":
            kept.append(row)
            continue

        text = clean_for_output(row.get("text_cohere") or "")
        row["text_cohere"] = text
        row["text"] = text
        row["cer_yt_vs_cohere"] = None

        reason = _row_reason(row, text, cfg, f)
        if reason:
            row["reject_reason"] = reason
            rejected.append(row)
        else:
            row.pop("reject_reason", None)
            kept.append(row)

    return kept, rejected


def _row_reason(row: dict, text: str, cfg: Config, f) -> str | None:
    """The first text gate this row fails, or None if it passes them all."""
    if not text:
        return "text:empty"

    words = text.split()
    if len(words) < f.min_words:
        return f"text:too_short={len(words)}w"

    language = row.get("asr_language") or "ar"
    ratio = script_ratio(text, language)
    row["arabic_ratio"] = round(arabic_ratio(text), 4)
    row["script_ratio"] = round(ratio, 4)
    if ratio < f.min_script_ratio:
        return f"text:script={language}={ratio:.2f}"

    # Google reports a 0-1 confidence, so it gets Google's floor. With no
    # subtitles to cross-check, this is the only guard against a hallucination.
    confidence = row.get("asr_confidence")
    if confidence is not None and confidence < f.min_asr_confidence_google:
        return f"text:asr_confidence={confidence:.2f}"

    duration = max(float(row.get("duration") or 0.0), 1e-6)
    # Counted on the undiacritized skeleton. The thresholds were calibrated on
    # unmarked Arabic (5.4-12.8 chars/sec), and tashkeel roughly halves again
    # the seconds per character without changing how long the words take to
    # say -- so counting marks would push ordinary diacritized speech straight
    # through the upper bound.
    cps = len(strip_tashkeel(text).replace(" ", "")) / duration
    row["chars_per_sec"] = round(cps, 3)
    if cps < f.min_chars_per_sec:
        return f"text:cps_low={cps:.1f}"
    if cps > f.max_chars_per_sec:
        return f"text:cps_high={cps:.1f}"

    return None


def summarize(chunks: list[Chunk]) -> dict[str, int]:
    """Histogram of rejection reasons, grouped by their category prefix."""
    counts: dict[str, int] = {}
    for chunk in chunks:
        if chunk.reject_reason is None:
            counts["kept"] = counts.get("kept", 0) + 1
        else:
            category = chunk.reject_reason.split(":", 1)[0]
            counts[category] = counts.get(category, 0) + 1
    return counts
