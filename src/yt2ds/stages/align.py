"""CTC forced alignment of the YouTube transcript against the audio.

YouTube's own subtitle timings are unreliable -- auto-captions in particular
drift, and their word timings describe when text *appeared on screen*, not when
it was spoken. So the timings are thrown away and recomputed: the full subtitle
text is aligned against the full audio with MMS-300M, an emission-per-frame CTC
model covering 1130+ languages.

Two details make this work for dialectal Arabic:

* The MMS aligner tokenizes *romanized* text (via uroman), so it does not care
  whether a word is Modern Standard, Najdi or Egyptian -- it aligns on sound,
  not on a fixed Arabic lexicon.
* Alignment runs over windows of the audio rather than the whole file, because
  the CTC trellis is O(frames x tokens) and a one-hour video would not fit.

The result is per-word timestamps, which are then assigned to chunks by
overlap. A word whose span straddles a chunk boundary marks that chunk as
boundary-clipped, since its transcript would be missing part of a word.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..models import ModelRegistry
from .segment import Chunk
from .subtitles import Subtitles

log = logging.getLogger(__name__)

_ALIGN_SR = 16000
# The MMS aligner's CTC blank is "<blank>" at id 0. It also has a "<pad>" at
# id 1, which is *not* the blank -- using that instead silently produces
# garbage alignments rather than an error, so resolve by name and fall back
# to 0 rather than guessing.
_BLANK_TOKENS = ("<blank>", "<pad>")


@dataclass
class AlignedWord:
    text: str
    start: float
    end: float
    score: float


_UROMAN = None


def _uroman():
    """Uroman loads sizeable data tables, so build it once per process."""
    global _UROMAN
    if _UROMAN is None:
        import uroman as ur

        _UROMAN = ur.Uroman()
    return _UROMAN


def _romanize(words: list[str]) -> list[str]:
    """Romanize Arabic words for the MMS aligner's Latin token vocabulary."""
    try:
        roman = _uroman()
        out = []
        for word in words:
            try:
                out.append(roman.romanize_string(word))
            except Exception:  # noqa: BLE001 - fall back per word
                out.append(word)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("uroman unavailable (%s); falling back to NFKD stripping", exc)
        return [unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode() or w for w in words]


def _tokenize(words: list[str], vocab: dict[str, int]) -> tuple[list[list[int]], list[int]]:
    """Map romanized words to vocabulary ids, reporting which words survived."""
    star = vocab.get("<star>")
    unk = vocab.get("<unk>", star)

    tokens: list[list[int]] = []
    kept: list[int] = []
    for index, word in enumerate(words):
        ids = [vocab[c] for c in word.lower() if c in vocab]
        if not ids:
            # Nothing in this word is representable (emoji, pure digits after
            # romanization). Skip it rather than derailing the alignment.
            if unk is None:
                continue
            ids = [unk]
        tokens.append(ids)
        kept.append(index)
    return tokens, kept


@torch.inference_mode()
def align(
    work_path: Path,
    subtitles: Subtitles,
    cfg: Config,
    registry: ModelRegistry,
) -> list[AlignedWord]:
    """Align ``subtitles`` against the 16 kHz working audio."""
    import soundfile as sf

    words = [w.text for w in subtitles.words]
    if not words:
        return []

    processor, model, vocab = registry.aligner
    romanized = _romanize(words)
    token_ids, kept = _tokenize(romanized, vocab)
    if not token_ids:
        log.warning("no alignable tokens in subtitle text")
        return []

    data, file_sr = sf.read(str(work_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != _ALIGN_SR:
        raise ValueError(f"aligner expects {_ALIGN_SR} Hz, got {file_sr}")

    total_seconds = len(data) / _ALIGN_SR
    window = cfg.align.window_seconds
    if total_seconds <= window:
        spans = [(0.0, total_seconds)]
    else:
        spans = _plan_windows(subtitles, kept, total_seconds, window)

    return _align_windows(data, spans, token_ids, kept, words, processor, model, vocab, registry)


def _plan_windows(
    subtitles: Subtitles,
    kept: list[int],
    total_seconds: float,
    window: float,
) -> list[tuple[float, float, int, int]]:
    """Split the job into (audio_start, audio_end, first_word, last_word) slices.

    The subtitle timings are too coarse to trust for alignment, but they are
    plenty good enough to decide *which* words belong in which ten-minute
    window, which is all that is needed to bound memory.
    """
    spans: list[tuple[float, float, int, int]] = []
    start = 0.0
    while start < total_seconds:
        end = min(total_seconds, start + window)
        # Words whose rough subtitle time lands in this window.
        indices = [
            position
            for position, word_index in enumerate(kept)
            if start <= subtitles.words[word_index].start < end
        ]
        if indices:
            spans.append((start, end, indices[0], indices[-1] + 1))
        start = end
    if not spans:
        spans = [(0.0, total_seconds, 0, len(kept))]
    return spans


def _align_windows(
    data: np.ndarray,
    spans,
    token_ids: list[list[int]],
    kept: list[int],
    words: list[str],
    processor,
    model,
    vocab: dict[str, int],
    registry: ModelRegistry,
) -> list[AlignedWord]:
    import warnings

    import torchaudio.functional as AF

    # torchaudio entered maintenance and marks forced_align deprecated, but it
    # is present and working in the pinned 2.9.1. The warning fires once per
    # window and would bury the pipeline's own output.
    warnings.filterwarnings("once", message=".*forced_align has been deprecated.*")

    blank_id = next((vocab[t] for t in _BLANK_TOKENS if t in vocab), 0)
    aligned: list[AlignedWord] = []

    normalized_spans = []
    for span in spans:
        if len(span) == 2:
            normalized_spans.append((span[0], span[1], 0, len(token_ids)))
        else:
            normalized_spans.append(span)

    for audio_start, audio_end, first, last in normalized_spans:
        if last <= first:
            continue
        begin = int(audio_start * _ALIGN_SR)
        stop = min(len(data), int(audio_end * _ALIGN_SR))
        segment = data[begin:stop]
        if segment.size < _ALIGN_SR // 10:
            continue

        wav = torch.from_numpy(np.ascontiguousarray(segment)).unsqueeze(0).to(registry.device)
        logits = model(wav).logits  # (1, frames, vocab)
        emissions = torch.log_softmax(logits.float(), dim=-1)
        frames = emissions.shape[1]
        seconds_per_frame = (stop - begin) / _ALIGN_SR / max(frames, 1)

        window_tokens = token_ids[first:last]
        flat = [t for word in window_tokens for t in word]
        if not flat:
            continue
        targets = torch.tensor([flat], dtype=torch.int32, device=registry.device)

        try:
            paths, scores = AF.forced_align(emissions, targets, blank=blank_id)
        except Exception as exc:  # noqa: BLE001 - a window that will not align
            log.warning("forced alignment failed on window %.0f-%.0fs: %s", audio_start, audio_end, exc)
            continue

        spans_per_token = _token_spans(paths[0].cpu().numpy(), scores[0].float().cpu().numpy(), blank_id)
        cursor = 0
        for offset, word_tokens in enumerate(window_tokens):
            count = len(word_tokens)
            token_span = spans_per_token[cursor : cursor + count]
            cursor += count
            if not token_span:
                continue
            frame_lo = token_span[0][0]
            frame_hi = token_span[-1][1]
            score = float(np.mean([s for _, _, s in token_span]))
            word_index = kept[first + offset]
            aligned.append(
                AlignedWord(
                    text=words[word_index],
                    start=audio_start + frame_lo * seconds_per_frame,
                    end=audio_start + frame_hi * seconds_per_frame,
                    score=score,
                )
            )

    aligned.sort(key=lambda w: w.start)
    return aligned


def _token_spans(path: np.ndarray, scores: np.ndarray, blank: int) -> list[tuple[int, int, float]]:
    """Collapse a CTC alignment path into one (start, end, score) per token."""
    spans: list[tuple[int, int, float]] = []
    frame = 0
    while frame < len(path):
        if path[frame] == blank:
            frame += 1
            continue
        start = frame
        label = path[frame]
        while frame < len(path) and path[frame] == label:
            frame += 1
        spans.append((start, frame, float(np.mean(scores[start:frame]))))
    return spans


def assign_to_chunks(
    chunks: list[Chunk],
    aligned: list[AlignedWord],
    cfg: Config,
) -> None:
    """Attach aligned words to chunks by time overlap.

    A word counts as belonging to a chunk when the majority of its span falls
    inside.

    A chunk is flagged as boundary-clipped only when a word it *owns* extends
    past its audio -- the transcript would then name a word the clip does not
    fully contain. Words that merely poke into a neighbour do not count:
    chunks are padded on both sides, so adjacent chunks deliberately overlap
    and edge words routinely touch two of them.
    """
    edge = 0.02  # tolerance for alignment jitter at frame resolution
    if not chunks:
        return
    ordered = sorted(chunks, key=lambda c: c.start)

    for chunk in ordered:
        chunk.scores.setdefault("boundary_clipped", False)

    for word in aligned:
        span = max(word.end - word.start, 1e-6)
        for chunk in ordered:
            if word.end <= chunk.start or word.start >= chunk.end:
                continue
            overlap = min(word.end, chunk.end) - max(word.start, chunk.start)
            if overlap >= span * 0.5:
                chunk.scores.setdefault("_words", []).append(word)
                if word.start < chunk.start - edge or word.end > chunk.end + edge:
                    chunk.scores["boundary_clipped"] = True

    for chunk in ordered:
        collected: list[AlignedWord] = chunk.scores.pop("_words", [])
        chunk.text_yt = " ".join(w.text for w in collected)
        if collected:
            chunk.scores["align_score"] = round(float(np.mean([w.score for w in collected])), 5)
            chunk.scores["aligned_words"] = len(collected)
        else:
            chunk.scores["align_score"] = None
            chunk.scores["aligned_words"] = 0
