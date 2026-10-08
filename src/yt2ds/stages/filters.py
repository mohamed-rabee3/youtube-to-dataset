"""Decide each chunk's canonical text, then gate on text quality.

For a TTS dataset a wrong transcript does not merely add noise: it teaches the
model to say a word as something else. So no transcript is trusted on its own.

Text source resolution:

* YouTube subtitles and ASR text both present -> kept only if their CER is
  under ``filters.max_cer_yt_vs_asr``. The ASR text becomes canonical
  (``filters.prefer_asr_text``, on by default: it is diacritized and properly
  spelled, subtitles are neither); ``text_source`` reads e.g. ``cohere+auto``.
* Only ASR text -> the ASR text is canonical. It is checked against the verify
  model's independent transcription of the same clip (``max_cer_verify``) and
  against the model's own confidence. ``text_source`` names the backend.
* Neither -> the chunk is dropped.

Every rate here is measured on the undiacritized skeleton and over the speech
inside the clip, not its padding: diacritics roughly double the character
count without changing how long the words take to say.
"""

from __future__ import annotations

import logging
import re

from ..arabic import arabic_ratio, cer, clean_for_output, script_ratio, strip_tashkeel
from ..config import Config, FiltersConfig
from .segment import Chunk

log = logging.getLogger(__name__)

#: ``text_source`` values meaning "no subtitles, the ASR transcript is canonical".
ASR_SOURCES = frozenset({"google", "cohere", "google_batch"})

# The two Cohere models write English and numbers differently -- one in Latin
# script and digits, the other transliterated and spelled out -- so a CER
# between them says nothing about whether either is right.
_UNCOMPARABLE = re.compile(r"[A-Za-z0-9٠-٩۰-۹]")


def resolve_text(chunks: list[Chunk], subtitle_kind: str | None, cfg: Config) -> None:
    """Pick canonical text per chunk and apply the text-quality gates."""
    backend = (cfg.asr.backend or "cohere").lower()
    for chunk in chunks:
        yt = clean_for_output(chunk.text_yt)
        asr = clean_for_output(chunk.text_asr)
        chunk.text_yt = yt
        chunk.text_asr = asr
        chunk.text_verify = clean_for_output(chunk.text_verify)

        if yt and asr:
            disagreement = cer(yt, asr)
            chunk.scores["cer_yt_vs_asr"] = round(disagreement, 5)
            # Subtitles are the check either way; which text is kept is a
            # choice. The ASR text carries diacritics and real orthography,
            # which subtitles never do.
            if cfg.filters.prefer_asr_text:
                chunk.text, chunk.text_source = asr, f"{backend}+{subtitle_kind or 'yt'}"
            else:
                chunk.text, chunk.text_source = yt, subtitle_kind or "yt"
            if disagreement > cfg.filters.max_cer_yt_vs_asr:
                chunk.reject(f"text:cer={disagreement:.3f}")
        elif asr:
            chunk.scores["cer_yt_vs_asr"] = None
            chunk.text = asr
            chunk.text_source = backend
        elif yt:
            # The ASR returned nothing for audio the subtitles claim has
            # speech. That disagreement is itself a signal, so do not trust
            # the text.
            chunk.scores["cer_yt_vs_asr"] = None
            chunk.text = yt
            chunk.text_source = subtitle_kind or "yt"
            chunk.reject("text:asr_empty")
        else:
            chunk.text = ""
            chunk.text_source = "none"
            chunk.reject("text:empty")

    _apply_text_gates(chunks, cfg)


def verify_cer(text: str, verify: str) -> float | None:
    """CER between the canonical text and the verify transcript, or None.

    None means the pair cannot be compared fairly: no verify transcript, or
    English or digits that the two models are known to write differently.
    """
    if not verify or _UNCOMPARABLE.search(verify) or _UNCOMPARABLE.search(text):
        return None
    return cer(verify, text)


def speech_rate(text: str, seconds: float) -> float:
    """Undiacritized characters per second of speech."""
    return len(strip_tashkeel(text).replace(" ", "")) / max(seconds, 1e-6)


def _apply_text_gates(chunks: list[Chunk], cfg: Config) -> None:
    f = cfg.filters
    for chunk in chunks:
        if chunk.reject_reason is not None:
            continue
        asr_only = chunk.text_source in ASR_SOURCES
        floor = f.min_asr_confidence_google if chunk.text_source == "google" else f.min_asr_confidence

        reason = text_gate(
            chunk.text,
            language=chunk.scores.get("asr_language") or "ar",
            seconds=chunk.speech_duration,
            confidence=chunk.scores.get("asr_confidence") if asr_only else None,
            confidence_floor=floor,
            verify=chunk.text_verify if asr_only else "",
            scores=chunk.scores,
            f=f,
        )
        if reason:
            chunk.reject(reason)
            continue

        if f.drop_boundary_clipped_words and chunk.scores.get("boundary_clipped"):
            chunk.reject("text:boundary_clipped")
            continue

        align_score = chunk.scores.get("align_score")
        if not asr_only and align_score is not None and align_score < cfg.align.min_align_score:
            chunk.reject(f"align:score={align_score:.2f}")


def text_gate(
    text: str,
    *,
    language: str,
    seconds: float,
    confidence: float | None,
    confidence_floor: float,
    verify: str,
    scores: dict,
    f: FiltersConfig,
) -> str | None:
    """The first text gate ``text`` fails, or None. Records measurements in ``scores``.

    Shared by the in-run chunk path and the row path that later
    transcription passes use, so the two can never drift apart.
    """
    if not text:
        return "text:empty"

    words = text.split()
    if len(words) < f.min_words:
        return f"text:too_short={len(words)}w"

    # Checked against the script of the language it was decoded in, so English
    # chunks are not rejected for failing to be Arabic.
    ratio = script_ratio(text, language)
    scores["arabic_ratio"] = round(arabic_ratio(text), 4)
    scores["script_ratio"] = round(ratio, 4)
    if ratio < f.min_script_ratio:
        return f"text:script={language}={ratio:.2f}"

    # With no subtitles, the model's own confidence and a second model's
    # opinion are what stand between the dataset and a hallucination.
    if confidence is not None and confidence < confidence_floor:
        return f"text:asr_confidence={confidence:.2f}"

    disagreement = verify_cer(text, verify)
    scores["cer_verify"] = round(disagreement, 5) if disagreement is not None else None
    if disagreement is not None and disagreement > f.max_cer_verify:
        return f"text:verify_cer={disagreement:.3f}"

    # Characters per second catches both halves of a timing failure: a
    # transcript far too long for its audio and one far too short.
    cps = speech_rate(text, seconds)
    scores["chars_per_sec"] = round(cps, 3)
    if cps < f.min_chars_per_sec:
        return f"text:cps_low={cps:.1f}"
    if cps > f.max_chars_per_sec:
        return f"text:cps_high={cps:.1f}"
    return None


def gate_rows(rows: list[dict], cfg: Config) -> tuple[list[dict], list[dict]]:
    """Apply the text gates to metadata rows, splitting them kept / rejected.

    The chunk-level gates above run inside a video's processing. Transcription
    passes that run later -- the batch backend, the Gemini scripts -- have no
    such moment: their text arrives against rows already on disk. The rules
    are the same ones, through the same :func:`text_gate`.

    Rows still marked ``pending`` have no text yet and are kept untouched, so a
    partial transcription pass never deletes audio it simply has no text for.
    """
    f = cfg.filters
    kept: list[dict] = []
    rejected: list[dict] = []

    for row in rows:
        if row.get("text_source") == "pending":
            kept.append(row)
            continue

        text = clean_for_output(row.get("text") or "")
        row["text"] = text
        source = row.get("text_source")
        # Each backend reports confidence on its own scale; Gemini reports none.
        floor = f.min_asr_confidence_google if source in ("google", "google_batch") else f.min_asr_confidence
        confidence = row.get("asr_confidence") if source in ASR_SOURCES else None
        seconds = float(row.get("speech_end") or 0) - float(row.get("speech_start") or 0)
        if seconds <= 0:
            seconds = float(row.get("duration") or 0.0)

        reason = text_gate(
            text,
            language=row.get("asr_language") or "ar",
            seconds=seconds,
            confidence=confidence,
            confidence_floor=floor,
            verify=row.get("text_verify") or "",
            scores=row,
            f=f,
        )
        if reason:
            row["reject_reason"] = reason
            rejected.append(row)
        else:
            row.pop("reject_reason", None)
            kept.append(row)

    return kept, rejected


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
