"""Transcribe chunks with a Cohere ASR model (``asr.backend: cohere``).

The default backend. Two models of the same architecture run here:

* ``asr.model`` -- by default NAMAA-Space/Cohere-Speech-Tashkeel-2B, which
  writes fully diacritized Arabic straight from the audio, so the dataset's
  vowels come from what was said rather than from a text diacritizer guessing.
* ``asr.verify_model`` -- by default Cohere Transcribe Arabic, an independent
  second transcription. ``filters`` compares the two skeletons and drops clips
  where they disagree, which is the guard a subtitle-less video otherwise
  lacks.

No API key, no per-minute billing and no network, at the cost of 2B models
resident on the card.

The model emits no timestamps, which is precisely why the pipeline segments
first and transcribes second: each chunk is already a 2-12 second clip, so the
transcript it returns *is* that chunk's text. There is no alignment step to get
wrong, and no risk of a word drifting across a boundary.

It is a 2B-parameter Conformer encoder-decoder (Apache-2.0) built for Arabic
dialect variety and Arabic-English code-switching, which is what makes it a
better fit here than a general multilingual model that flattens Saudi and
Egyptian speech toward MSA.

**Language selection.** The processor takes a required ``language`` prompt and
has no auto-detect. Forcing every chunk through the Arabic prompt makes the
model invent Arabic for speech that is actually English. So each chunk is
decoded once per configured language and the higher-confidence result wins,
where confidence is the mean per-token log-probability of the generated
sequence. Arabic-English code-switching inside a single chunk is handled by the
model itself under whichever prompt scores better.

That same confidence number is the only quality signal available when a video
has no subtitles to cross-check against, so it is recorded either way.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..arabic import clean_for_output
from ..config import Config
from ..models import ModelRegistry
from .audio import resample
from .segment import Chunk

log = logging.getLogger(__name__)

_ASR_SR = 16000


@torch.inference_mode()
def transcribe_batch(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    cfg: Config,
    registry: ModelRegistry,
) -> list[str]:
    """Transcribe a batch, returning one string per chunk.

    Writes ``asr_confidence`` and ``asr_language`` into each chunk's scores.
    """
    if not items:
        return []

    processor, model = registry.asr
    audio = [resample(samples, sr, _ASR_SR) for _, samples in items]
    languages = cfg.asr.languages or ["ar"]

    # candidates[language] -> (texts, confidences) aligned with `audio`
    candidates = {
        language: _decode_all(audio, language, processor, model, cfg, registry) for language in languages
    }

    results: list[str] = []
    for index, (chunk, _) in enumerate(items):
        best_language, best_text, best_confidence = _pick(candidates, index, languages)
        chunk.scores["asr_confidence"] = best_confidence
        chunk.scores["asr_language"] = best_language
        if len(languages) > 1:
            # Keep the runner-up's confidence so the margin is inspectable.
            others = {
                language: candidates[language][1][index]
                for language in languages
                if language != best_language
            }
            chunk.scores["asr_confidence_alt"] = others
        results.append(best_text)
    return results


@torch.inference_mode()
def verify_batch(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    cfg: Config,
    registry: ModelRegistry,
) -> list[str]:
    """Second, independent transcription with ``asr.verify_model``.

    Decoded under the language the primary transcript was chosen in, so the
    two are comparable. Writes ``verify_confidence`` into each chunk's scores.
    """
    if not items:
        return []
    processor, model = registry.asr_verify
    audio = [resample(samples, sr, _ASR_SR) for _, samples in items]
    by_language: dict[str, list[int]] = {}
    for index, (chunk, _) in enumerate(items):
        language = chunk.scores.get("asr_language") or (cfg.asr.languages or ["ar"])[0]
        by_language.setdefault(language, []).append(index)

    texts = [""] * len(items)
    for language, indices in by_language.items():
        decoded, scores = _decode_all([audio[i] for i in indices], language, processor, model, cfg, registry)
        for i, text, score in zip(indices, decoded, scores):
            texts[i] = text
            items[i][0].scores["verify_confidence"] = score
    return texts


def _decode_all(
    audio: list[np.ndarray],
    language: str,
    processor,
    model,
    cfg: Config,
    registry: ModelRegistry,
) -> tuple[list[str], list[float | None]]:
    """Decode every clip under one language prompt, ``asr.batch_size`` at a time."""
    texts: list[str] = []
    confidences: list[float | None] = []
    step = max(1, cfg.asr.batch_size)
    for start in range(0, len(audio), step):
        window_texts, window_scores = _transcribe(audio[start : start + step], language, processor, model, cfg, registry)
        texts.extend(window_texts)
        confidences.extend(window_scores)
    return texts, confidences


def _pick(
    candidates: dict[str, tuple[list[str], list[float | None]]],
    index: int,
    languages: list[str],
) -> tuple[str, str, float | None]:
    """Choose the language whose decoding the model was most confident in.

    Empty output never wins over non-empty output, regardless of confidence:
    a confidently-produced empty string is not a transcription.
    """
    best_language = languages[0]
    best_text = ""
    best_confidence: float | None = None

    for language in languages:
        texts, confidences = candidates[language]
        text = texts[index]
        confidence = confidences[index]
        if not text:
            continue
        if not best_text:
            best_language, best_text, best_confidence = language, text, confidence
            continue
        if confidence is not None and (best_confidence is None or confidence > best_confidence):
            best_language, best_text, best_confidence = language, text, confidence

    if not best_text:
        # Nothing decoded in any language; report the primary language's score.
        best_confidence = candidates[languages[0]][1][index]
    return best_language, best_text, best_confidence


def _transcribe(
    audio: list[np.ndarray],
    language: str,
    processor,
    model,
    cfg: Config,
    registry: ModelRegistry,
) -> tuple[list[str], list[float | None]]:
    usable = [i for i, a in enumerate(audio) if a.size >= _ASR_SR // 10]
    if not usable:
        return [""] * len(audio), [None] * len(audio)

    inputs = processor(
        [audio[i] for i in usable],
        sampling_rate=_ASR_SR,
        return_tensors="pt",
        language=language,
    )
    # Audio longer than the model's window (35 s) comes back as several rows.
    # This maps every row to the clip it came from; without it, one long clip
    # would shift every later transcript onto the wrong clip.
    row_owner = _row_owners(inputs.pop("audio_chunk_index", None), len(usable))
    # The feature extractor always returns float32 log-mels. The model runs in
    # bf16, and the encoder's first conv rejects a dtype mismatch outright, so
    # cast the float tensors while leaving the integer ones (decoder_input_ids,
    # attention_mask) alone.
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: _to_device(v, registry.device, model_dtype) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.asr.max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
    except torch.cuda.OutOfMemoryError:
        # Fall back to one-at-a-time rather than losing the whole batch.
        torch.cuda.empty_cache()
        log.warning("ASR OOM at batch size %d; retrying individually", len(usable))
        texts: list[str] = [""] * len(audio)
        scores: list[float | None] = [None] * len(audio)
        for i in usable:
            single_text, single_score = _transcribe([audio[i]], language, processor, model, cfg, registry)
            texts[i] = single_text[0]
            scores[i] = single_score[0]
        return texts, scores

    decoded = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    confidences = _confidence(model, outputs, processor)

    # Rows arrive in clip order, and within a clip in time order.
    pieces: dict[int, list[str]] = {}
    row_scores: dict[int, list[float]] = {}
    for row, owner in enumerate(row_owner):
        pieces.setdefault(owner, []).append(decoded[row].strip())
        if confidences[row] is not None:
            row_scores.setdefault(owner, []).append(confidences[row])

    texts = [""] * len(audio)
    scores = [None] * len(audio)
    for position, i in enumerate(usable):
        texts[i] = clean_for_output(" ".join(p for p in pieces.get(position, []) if p))
        kept = row_scores.get(position)
        scores[i] = round(sum(kept) / len(kept), 5) if kept else None
    return texts, scores


def _row_owners(chunk_index, count: int) -> list[int]:
    """Which input clip each feature row belongs to."""
    if chunk_index is None:
        return list(range(count))
    return [int(entry[0]) if isinstance(entry, (tuple, list)) else int(entry) for entry in chunk_index]


def _confidence(model, outputs, processor) -> list[float | None]:
    """Mean per-token log-probability of each generated sequence.

    Padding and post-EOS positions are excluded, so a short confident
    transcription is not penalised against a long one.
    """
    try:
        transition = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
    except Exception as exc:  # noqa: BLE001 - scoring must never break a run
        log.debug("could not compute ASR confidence: %s", exc)
        return [None] * len(outputs.sequences)

    generated = outputs.sequences[:, outputs.sequences.shape[1] - transition.shape[1] :]
    tokenizer = getattr(processor, "tokenizer", processor)
    ignore = {
        token_id
        for token_id in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(model.generation_config, "pad_token_id", None),
            getattr(model.generation_config, "eos_token_id", None),
        )
        if isinstance(token_id, int)
    }

    mask = torch.ones_like(generated, dtype=torch.bool)
    for token_id in ignore:
        mask &= generated != token_id
    mask &= torch.isfinite(transition)

    out: list[float | None] = []
    for row_scores, row_mask in zip(transition, mask):
        kept = row_scores[row_mask]
        out.append(round(float(kept.mean()), 5) if kept.numel() else None)
    return out


def _to_device(tensor, device, dtype):
    tensor = tensor.to(device)
    return tensor.to(dtype) if tensor.dtype.is_floating_point else tensor
