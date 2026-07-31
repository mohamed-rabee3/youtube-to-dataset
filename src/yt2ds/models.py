"""Lazy, cached model registry.

Every GPU model is loaded on first use and then kept resident. The full set
(Cohere 2B + Demucs + AST + ECAPA + MMS + SQUIM) fits comfortably in well under
half of a 48 GB card, so paying the load cost once per run rather than once per
video is the right trade.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .config import Config

log = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ModelRegistry:
    """Holds one instance of each model, loaded on demand."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._cache: dict[str, Any] = {}
        requested = cfg.runtime.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to CPU (this will be slow)")
            requested = "cpu"
        self.device = torch.device(requested)

    @property
    def device_string(self) -> str:
        """Device with an explicit index, for libraries that parse it as text."""
        if self.device.type == "cuda":
            return f"cuda:{self.device.index if self.device.index is not None else 0}"
        return str(self.device)

    # -- helpers ---------------------------------------------------------
    def _get(self, key: str, loader) -> Any:
        if key not in self._cache:
            log.info("loading model: %s", key)
            self._cache[key] = loader()
        return self._cache[key]

    def release(self, key: str) -> None:
        if self._cache.pop(key, None) is not None:
            torch.cuda.empty_cache()

    def release_all(self) -> None:
        self._cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- models ----------------------------------------------------------
    @property
    def vad(self) -> Any:
        """Silero VAD model plus its helper functions."""

        def load():
            from silero_vad import get_speech_timestamps, load_silero_vad

            return load_silero_vad(), get_speech_timestamps

        return self._get("vad", load)

    @property
    def ast(self) -> tuple[Any, Any]:
        """AudioSet event classifier (feature extractor, model)."""

        def load():
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            name = self.cfg.music.ast_model
            extractor = AutoFeatureExtractor.from_pretrained(name)
            model = AutoModelForAudioClassification.from_pretrained(name)
            model.to(self.device).eval()
            return extractor, model

        return self._get("ast", load)

    @property
    def demucs(self) -> Any:
        """Demucs source separator, used for the vocals/accompaniment ratio."""

        def load():
            from demucs.pretrained import get_model

            model = get_model("htdemucs")
            model.to(self.device).eval()
            return model

        return self._get("demucs", load)

    @property
    def speaker_encoder(self) -> Any:
        """ECAPA-TDNN speaker embeddings for chunk purity and cross-video linking."""

        def load():
            from speechbrain.inference.speaker import EncoderClassifier

            return EncoderClassifier.from_hparams(
                source=self.cfg.speakers.embed_model,
                savedir=f".cache/speechbrain/{self.cfg.speakers.embed_model.replace('/', '_')}",
                # SpeechBrain parses the device string itself and cannot handle
                # a bare "cuda", so hand it an explicit index.
                run_opts={"device": self.device_string},
            )

        return self._get("speaker_encoder", load)

    @property
    def asr(self) -> tuple[Any, Any]:
        """Cohere Transcribe Arabic (processor, model).

        Emits no timestamps, which is why the pipeline segments the audio first
        and runs this per chunk.
        """

        def load():
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration

            name = self.cfg.asr.model
            dtype = _DTYPES[self.cfg.asr.dtype]
            processor = AutoProcessor.from_pretrained(name)
            model = CohereAsrForConditionalGeneration.from_pretrained(name, dtype=dtype)
            model.to(self.device).eval()
            return processor, model

        return self._get("asr", load)

    @property
    def aligner(self) -> tuple[Any, Any, Any]:
        """MMS-300M CTC forced aligner (processor, model, vocabulary)."""

        def load():
            from transformers import AutoProcessor, Wav2Vec2ForCTC

            name = self.cfg.align.model
            processor = AutoProcessor.from_pretrained(name)
            model = Wav2Vec2ForCTC.from_pretrained(name)
            model.to(self.device).eval()
            tokenizer = getattr(processor, "tokenizer", processor)
            vocab = tokenizer.get_vocab()
            return processor, model, vocab

        return self._get("aligner", load)

    @property
    def squim_objective(self) -> Any:
        """SQUIM objective metrics: estimated STOI / PESQ / SI-SDR, no reference."""

        def load():
            from torchaudio.pipelines import SQUIM_OBJECTIVE

            model = SQUIM_OBJECTIVE.get_model()
            model.to(self.device).eval()
            return model

        return self._get("squim_objective", load)

    @property
    def squim_subjective(self) -> Any:
        """SQUIM subjective MOS. Needs a non-matching clean reference clip."""

        def load():
            from torchaudio.pipelines import SQUIM_SUBJECTIVE

            model = SQUIM_SUBJECTIVE.get_model()
            model.to(self.device).eval()
            return model

        return self._get("squim_subjective", load)
