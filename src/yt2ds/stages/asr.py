"""Transcription: pick a backend and hand it the batch.

Two backends, same contract. Each takes a batch of (chunk, samples) and returns
one string per chunk, having written ``asr_confidence`` and ``asr_language``
into every chunk's scores:

* ``google`` (default) -- Cloud Speech-to-Text, model ``latest_long``. One
  small request per chunk, sent concurrently. Needs credentials and is billed
  per audio minute; needs no GPU.
* ``cohere`` -- Cohere Transcribe Arabic, run locally on the GPU. No API key,
  no network, no per-minute cost; a 2B model resident on the card instead.

The two report confidence on different scales, so the floors that gate on it
are separate config keys. See ``filters.min_asr_confidence`` /
``filters.min_asr_confidence_google``.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..models import ModelRegistry
from .segment import Chunk

BACKENDS = ("google", "cohere")


def transcribe_batch(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    cfg: Config,
    registry: ModelRegistry,
) -> list[str]:
    """Transcribe a batch with the configured backend, one string per chunk."""
    backend = (cfg.asr.backend or "google").lower()
    if backend == "google":
        from . import asr_google

        return asr_google.transcribe_batch(items, sr, cfg, registry)
    if backend == "cohere":
        from . import asr_cohere

        return asr_cohere.transcribe_batch(items, sr, cfg, registry)
    raise ValueError(f"unknown asr.backend: {cfg.asr.backend!r} (expected one of {', '.join(BACKENDS)})")
