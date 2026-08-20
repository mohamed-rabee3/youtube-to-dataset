"""Lazy, cached model registry.

Every GPU model is loaded on first use and then kept resident. The full set
(Cohere 2B + Demucs + AST + ECAPA + MMS + SQUIM) fits comfortably in well under
half of a 48 GB card, so paying the load cost once per run rather than once per
video is the right trade. On the default ASR backend the Cohere 2B is never
loaded at all, which takes the resident set well below that again.

The Google Speech-to-Text client lives here too. It is not a model, but it is
built once and reused for the same reason.
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

    def free_cached_memory(self) -> None:
        """Return unused allocator blocks to the driver, keeping models loaded.

        Torch's caching allocator never gives memory back on its own, so after
        one big video this process can be sitting on tens of gigabytes it is
        not using. That is invisible within the process but fatal to the
        out-of-process diarizer, which sees only what the driver reports free.
        """
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
    def separator(self) -> Any:
        """Mel-Band RoFormer vocal isolator, loaded through audio-separator.

        This is the model that *removes* music; `demucs` above only measures
        how much of it is there. They are deliberately different models: the
        detector's job is a cheap ratio over four stems, the isolator's is the
        highest-fidelity voice it can reconstruct from under the music bed.
        """

        def load():
            from audio_separator.separator import Separator

            sep_cfg = self.cfg.separate
            if self.device.type == "cuda" and (self.device.index or 0) != 0:
                # audio-separator picks torch.device("cuda") itself.
                log.warning(
                    "separation runs on cuda:0; runtime.device=%s applies to every other model",
                    self.cfg.runtime.device,
                )
            separator = Separator(
                log_level=logging.WARNING,
                model_file_dir=sep_cfg.model_dir,
                # Overwritten per video by the separate stage.
                output_dir=sep_cfg.model_dir,
                output_format="WAV",
                output_single_stem=sep_cfg.stem,
                use_autocast=self.device.type == "cuda",
                mdxc_params={
                    "segment_size": sep_cfg.segment_size,
                    "override_model_segment_size": False,
                    "batch_size": sep_cfg.batch_size,
                    "overlap": sep_cfg.overlap,
                    "pitch_shift": 0,
                },
            )
            separator.load_model(model_filename=sep_cfg.model)
            return separator

        return self._get("separator", load)

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
    def speech_client(self) -> Any:
        """Google Cloud Speech-to-Text client, the default ASR backend.

        Not a GPU model, but it belongs here for the same reason the others do:
        built once and reused, rather than reconstructed per batch. The client
        is thread-safe, which is what lets a batch's chunks go out in parallel.
        """

        def load():
            from .stages.asr_google import GoogleAsrUnavailable, _speech_module

            speech = _speech_module()
            path = self.cfg.asr.google.credentials_file
            try:
                if path:
                    return speech.SpeechClient.from_service_account_file(path)
                return speech.SpeechClient()
            except Exception as exc:  # noqa: BLE001 - surfaced as a clear setup error
                raise GoogleAsrUnavailable(
                    f"could not authenticate to Google Cloud Speech-to-Text ({exc}). Set "
                    "asr.google.credentials_file or GOOGLE_APPLICATION_CREDENTIALS to a "
                    "service-account JSON, or run with --asr-backend cohere."
                ) from exc

        return self._get("speech_client", load)

    @property
    def asr(self) -> tuple[Any, Any]:
        """Cohere Transcribe Arabic (processor, model), the ``cohere`` backend.

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
