"""Transcribe chunks with Google Cloud Speech-to-Text (``asr.backend: google``).

The default backend. Like the Cohere path it transcribes *chunks*, not whole
videos: the pipeline has already cut the audio into 2-12 second clips, so the
transcript the API returns for a clip *is* that clip's text. Nothing has to be
aligned back onto a timeline afterwards, and no word can drift across a chunk
boundary. That is worth one request per clip, which is why the requests go out
concurrently rather than one at a time.

**Language selection.** Unlike Cohere, the API detects the language itself:
the first entry of ``asr.languages`` is sent as ``language_code`` and the rest
as ``alternative_language_codes``, so a chunk costs one request no matter how
many languages are configured. The response says which language it decided on,
and that is what the script-ratio gate downstream is checked against.

**Confidence.** Google reports its own 0-1 confidence per result, on a
different scale from Cohere's mean per-token log-probability. It is written to
the same ``asr_confidence`` score, and ``filters.min_asr_confidence_google``
is the floor that applies to it.

Credentials come from ``asr.google.credentials_file``, else
``GOOGLE_APPLICATION_CREDENTIALS``, else application default credentials.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..arabic import clean_for_output
from ..config import Config
from ..models import ModelRegistry
from .audio import resample
from .segment import Chunk

log = logging.getLogger(__name__)

_ASR_SR = 16000
# Below this the API has nothing to work with and reliably returns no results.
_MIN_SAMPLES = _ASR_SR // 10

# Set once, per process, if the API rejects alternative_language_codes for the
# configured model -- not every model supports them, and the error is the only
# way to find out. Retrying every chunk to rediscover that would double the
# request count for the whole run.
_alternatives_rejected = False


class GoogleAsrUnavailable(RuntimeError):
    """The Speech-to-Text client could not be built (missing SDK or credentials)."""


def transcribe_batch(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    cfg: Config,
    registry: ModelRegistry,
) -> list[str]:
    """Transcribe a batch, returning one string per chunk.

    Writes ``asr_confidence`` and ``asr_language`` into each chunk's scores,
    matching the Cohere backend's contract.
    """
    if not items:
        return []

    client = registry.speech_client
    speech = _speech_module()
    languages = _locales(cfg)
    recognition_config = _recognition_config(speech, cfg, languages)

    payloads = [_pcm16(resample(samples, sr, _ASR_SR)) for _, samples in items]
    workers = max(1, min(cfg.asr.google.max_workers, len(payloads)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(
            pool.map(
                lambda content: _recognize(client, speech, recognition_config, content, cfg),
                payloads,
            )
        )

    results: list[str] = []
    for (chunk, _), (text, confidence, language) in zip(items, outcomes):
        chunk.scores["asr_confidence"] = confidence
        chunk.scores["asr_language"] = language or _short(languages[0])
        results.append(text)
    return results


def _recognize(client, speech, recognition_config, content: bytes, cfg: Config) -> tuple[str, float | None, str | None]:
    """One chunk. Returns (text, confidence, short language code)."""
    global _alternatives_rejected

    if len(content) < _MIN_SAMPLES * 2:  # 2 bytes per sample
        return "", None, None

    audio = speech.RecognitionAudio(content=content)
    google = cfg.asr.google
    from google.api_core import exceptions as gexc

    delay = google.retry_backoff
    for attempt in range(1, google.max_retries + 1):
        config = recognition_config
        if _alternatives_rejected and config.alternative_language_codes:
            config = _without_alternatives(speech, config)
        try:
            response = client.recognize(config=config, audio=audio, timeout=google.timeout)
            return _collect(response)
        except gexc.InvalidArgument as exc:
            # Not every model accepts alternative_language_codes. Drop them and
            # retry once, then remember for the rest of the run.
            if config.alternative_language_codes and "language" in str(exc).lower():
                log.warning(
                    "%s does not accept alternative languages; falling back to %s only",
                    google.model,
                    config.language_code,
                )
                _alternatives_rejected = True
                continue
            log.error("Speech-to-Text rejected a chunk: %s", exc)
            return "", None, None
        except (
            gexc.ServiceUnavailable,
            gexc.DeadlineExceeded,
            gexc.ResourceExhausted,
            gexc.InternalServerError,
            gexc.TooManyRequests,
        ) as exc:
            if attempt == google.max_retries:
                log.error("Speech-to-Text failed after %d attempts: %s", attempt, exc)
                return "", None, None
            log.warning("Speech-to-Text %s; retry %d in %.1fs", type(exc).__name__, attempt, delay)
            time.sleep(delay)
            delay *= 2
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the run
            log.error("Speech-to-Text error on a chunk: %s", exc)
            return "", None, None
    return "", None, None


def _collect(response) -> tuple[str, float | None, str | None]:
    """Flatten the API's per-result list into one chunk-level transcript.

    A short clip usually comes back as a single result, but the API is free to
    split it. Confidence is averaged over the results that reported one, and
    the language is taken from the first result that named one.
    """
    parts: list[str] = []
    confidences: list[float] = []
    language: str | None = None

    for result in response.results:
        if not result.alternatives:
            continue
        best = result.alternatives[0]
        transcript = (best.transcript or "").strip()
        if not transcript:
            continue
        parts.append(transcript)
        # The field is unset rather than absent when the model does not score
        # a result, and unset reads back as 0.0.
        if best.confidence:
            confidences.append(float(best.confidence))
        if language is None and getattr(result, "language_code", ""):
            language = _short(result.language_code)

    text = clean_for_output(" ".join(parts))
    confidence = round(sum(confidences) / len(confidences), 5) if confidences else None
    return text, confidence, language


def _recognition_config(speech, cfg: Config, languages: list[str]):
    google = cfg.asr.google
    alternatives = languages[1:] if google.use_alternative_languages else []
    return speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=_ASR_SR,
        audio_channel_count=1,
        language_code=languages[0],
        # The API caps this at three alternatives.
        alternative_language_codes=alternatives[:3],
        model=google.model,
        enable_automatic_punctuation=google.enable_automatic_punctuation,
        profanity_filter=google.profanity_filter,
    )


def _without_alternatives(speech, config):
    stripped = speech.RecognitionConfig()
    speech.RecognitionConfig.copy_from(stripped, config)
    del stripped.alternative_language_codes[:]
    return stripped


def _locales(cfg: Config) -> list[str]:
    """Map ``asr.languages`` short codes onto the BCP-47 locales the API wants."""
    table = cfg.asr.google.locales or {}
    codes = [table.get(language, language) for language in (cfg.asr.languages or ["ar"])]
    # Duplicates would be sent as pointless alternatives; order is significant.
    seen: dict[str, None] = {}
    for code in codes:
        seen.setdefault(code, None)
    return list(seen) or ["ar-SA"]


def _short(locale: str) -> str:
    """``ar-SA`` -> ``ar``, which is what the script-ratio gate expects."""
    return locale.split("-", 1)[0].lower()


def _pcm16(samples: np.ndarray) -> bytes:
    """Mono float32 in [-1, 1] -> 16-bit little-endian PCM, the LINEAR16 wire format."""
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _speech_module():
    try:
        from google.cloud import speech
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise GoogleAsrUnavailable(
            "google-cloud-speech is not installed. Install it (pip install google-cloud-speech) "
            "or run with --asr-backend cohere."
        ) from exc
    return speech
