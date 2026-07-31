"""Decide each chunk's canonical text, then gate on text quality.

This is where the user's core requirement lives: YouTube's subtitles are the
canonical text, but they are *verified* against Cohere's transcription of the
same chunk and dropped when the two disagree. For a TTS dataset that check
matters more than it would for ASR -- a wrong transcript does not merely add
noise, it teaches the model to pronounce a word as something else entirely.

Text source resolution:

* Both YouTube subtitles and Cohere text present -> YouTube text wins, but only
  if the CER between them is under the gate. Both are kept in the metadata so
  the threshold can be re-tuned later without recomputing anything.
* Only Cohere text (video had no Arabic subtitles) -> Cohere text is canonical
  and the CER gate does not apply. ``text_source`` records this.
* Neither -> the chunk is dropped.
"""

from __future__ import annotations

import logging

from ..arabic import arabic_ratio, cer, clean_for_output, script_ratio
from ..config import Config
from .segment import Chunk

log = logging.getLogger(__name__)


def resolve_text(chunks: list[Chunk], subtitle_kind: str | None, cfg: Config) -> None:
    """Pick canonical text per chunk and apply the text-quality gates."""
    for chunk in chunks:
        yt = clean_for_output(chunk.text_yt)
        cohere = clean_for_output(chunk.text_cohere)
        chunk.text_yt = yt
        chunk.text_cohere = cohere

        if yt and cohere:
            disagreement = cer(yt, cohere)
            chunk.scores["cer_yt_vs_cohere"] = round(disagreement, 5)
            chunk.text = yt
            chunk.text_source = subtitle_kind or "yt"
            if disagreement > cfg.filters.max_cer_yt_vs_cohere:
                chunk.reject(f"text:cer={disagreement:.3f}")
        elif cohere:
            chunk.scores["cer_yt_vs_cohere"] = None
            chunk.text = cohere
            chunk.text_source = "cohere"
        elif yt:
            # Cohere returned nothing for audio the subtitles claim has speech.
            # That disagreement is itself a signal, so do not trust the text.
            chunk.scores["cer_yt_vs_cohere"] = None
            chunk.text = yt
            chunk.text_source = subtitle_kind or "yt"
            chunk.reject("text:cohere_empty")
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
        if (
            chunk.text_source == "cohere"
            and confidence is not None
            and confidence < f.min_asr_confidence
        ):
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
            chunk.text_source != "cohere"
            and align_score is not None
            and align_score < cfg.align.min_align_score
        ):
            chunk.reject(f"align:score={align_score:.2f}")


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
