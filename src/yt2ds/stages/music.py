"""Reject chunks containing music or non-speech noise.

Two independent detectors, because each misses what the other catches:

* **Demucs** separates the chunk into vocals / drums / bass / other and reports
  how much energy sits in the accompaniment. This is what catches *speech over
  background music* -- a talking-head intro with a music bed reads as ordinary
  speech to a classifier, but Demucs sees the instrumental energy plainly.
* **AST** (AudioSet) classifies the chunk against 527 event labels. This is what
  catches applause, crowd noise, and pure-music passages that Demucs might
  attribute to "vocals" when someone is singing.

A chunk is dropped if either detector fires. Scores are recorded either way, so
thresholds can be re-tuned against ``rejected.jsonl`` without recomputing.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..config import Config
from ..models import ModelRegistry
from .audio import resample
from .segment import Chunk

log = logging.getLogger(__name__)

_AST_SR = 16000


def score_batch(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    cfg: Config,
    registry: ModelRegistry,
) -> None:
    """Score a batch of chunks in place, writing into ``chunk.scores``."""
    if not items:
        return
    _score_demucs(items, sr, cfg, registry)
    _score_ast(items, sr, cfg, registry)

    for chunk, _ in items:
        ratio = chunk.scores.get("accompaniment_ratio")
        music = chunk.scores.get("music_score")
        if ratio is not None and ratio > cfg.music.max_accompaniment_ratio:
            chunk.reject(f"music:accompaniment={ratio:.3f}")
        elif music is not None and music > cfg.music.max_music_score:
            label = chunk.scores.get("music_label", "?")
            chunk.reject(f"music:ast={label}={music:.3f}")


# ---------------------------------------------------------------- Demucs
@torch.inference_mode()
def _score_demucs(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    cfg: Config,
    registry: ModelRegistry,
) -> None:
    from demucs.apply import apply_model

    model = registry.demucs
    target_sr = cfg.audio.demucs_sample_rate
    sources = list(model.sources)
    try:
        vocals_index = sources.index("vocals")
    except ValueError:
        log.warning("demucs model has no 'vocals' source; skipping accompaniment scoring")
        return

    for chunk, samples in items:
        if samples.size == 0:
            chunk.scores["accompaniment_ratio"] = 1.0
            continue

        audio = resample(samples, sr, target_sr)
        # Demucs expects (channels, samples); the model is stereo, so duplicate.
        wav = torch.from_numpy(audio).to(registry.device)
        wav = wav.unsqueeze(0).repeat(model.audio_channels, 1)

        # Demucs is trained on standardized input.
        ref = wav.mean(0)
        std = ref.std()
        if std < 1e-8:
            chunk.scores["accompaniment_ratio"] = 1.0
            continue
        wav = (wav - ref.mean()) / std

        estimates = apply_model(
            model,
            wav[None],
            device=registry.device,
            split=True,
            overlap=0.10,
            progress=False,
        )[0]

        energies = (estimates.float() ** 2).mean(dim=(1, 2))
        vocal_energy = float(energies[vocals_index])
        accompaniment = float(energies.sum() - energies[vocals_index])
        total = vocal_energy + accompaniment
        # Fraction of separated energy that is not voice.
        ratio = accompaniment / total if total > 1e-12 else 1.0

        chunk.scores["accompaniment_ratio"] = round(ratio, 5)
        chunk.scores["vocal_ratio"] = round(1.0 - ratio, 5)


# ------------------------------------------------------------------- AST
@torch.inference_mode()
def _score_ast(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    cfg: Config,
    registry: ModelRegistry,
) -> None:
    extractor, model = registry.ast
    id2label = model.config.id2label
    wanted = {label.lower() for label in cfg.music.music_labels}
    music_ids = [i for i, label in id2label.items() if label.lower() in wanted]
    speech_ids = [i for i, label in id2label.items() if label.lower() in {"speech", "male speech, man speaking", "female speech, woman speaking", "conversation", "narration, monologue"}]
    if not music_ids:
        log.warning("no configured music labels matched the AST label set")
        return

    audio = [resample(samples, sr, _AST_SR) for _, samples in items]
    audio = [a if a.size else np.zeros(_AST_SR, dtype=np.float32) for a in audio]

    inputs = extractor(audio, sampling_rate=_AST_SR, return_tensors="pt")
    inputs = {k: v.to(registry.device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    # AudioSet is multi-label, so each class gets an independent sigmoid.
    probs = torch.sigmoid(logits).float().cpu().numpy()

    for (chunk, _), row in zip(items, probs):
        music_scores = {id2label[i]: float(row[i]) for i in music_ids}
        top_label, top_score = max(music_scores.items(), key=lambda kv: kv[1])
        chunk.scores["music_score"] = round(top_score, 5)
        chunk.scores["music_label"] = top_label
        if speech_ids:
            chunk.scores["speech_score"] = round(float(max(row[i] for i in speech_ids)), 5)
