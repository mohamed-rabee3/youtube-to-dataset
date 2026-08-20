"""Whole-episode transcription via Speech-to-Text V2 BatchRecognize.

This backend inverts the usual order. The per-chunk backends transcribe each
clip on its own, which is simple but pays Google's 15-second-per-request
rounding on clips that average under five seconds -- roughly three times the
audio actually present. Here the whole episode goes up once, billed for its
real length, and the word-level timestamps that come back are cut into the
chunks the diarizer already chose. The forced aligner goes too: its only job
was to place subtitle words in time, and these words arrive with times.

The flow is two-phase by necessity. Batch recognition only reads from Cloud
Storage and answers as a long-running operation, so the GPU pass runs to
completion first and writes its chunk boundaries, then this module uploads,
submits, waits, and fills the text in afterwards. That split is also what makes
the discounted ``DYNAMIC_BATCHING`` tier usable: nothing is blocked on the
transcript, so a latency allowance of up to 24 hours costs nothing.

One episode per request. The API bounds how much it will return inline, and a
request per file keeps the mapping from transcript back to episode trivial;
concurrency comes from having many requests in flight, not many files in one.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..config import Config

log = logging.getLogger(__name__)

# Google wants >= 16 kHz; the working copy is already there and mono FLAC is
# lossless, so the upload is a third the size of the equivalent WAV.
EXPORT_SR = 16000


class GoogleBatchUnavailable(RuntimeError):
    """Raised when the backend cannot run at all, rather than for one episode."""


@dataclass
class BatchWord:
    text: str
    start: float
    end: float
    confidence: float


def export_audio(source: Path, target: Path) -> Path:
    """Write the mono 16 kHz FLAC that gets uploaded for one episode.

    ``source`` is the working copy, which is already mono at 16 kHz, so this is
    a container change rather than a resample. FLAC because it is lossless and
    about a third the size of the WAV -- across 209 episodes that is the
    difference between a 36 GB upload and a 10 GB one.
    """
    import subprocess

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", str(source),
            "-ac", "1", "-ar", str(EXPORT_SR), "-c:a", "flac",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


# -- cloud ---------------------------------------------------------------
def _clients(cfg: Config):
    from google.api_core.client_options import ClientOptions
    from google.cloud import storage
    from google.cloud.speech_v2 import SpeechClient

    batch = cfg.asr.google_batch
    if not batch.bucket:
        raise GoogleBatchUnavailable(
            "asr.google_batch.bucket is unset; batch recognition can only read from "
            "Cloud Storage. Set it in the config or pass --bucket."
        )
    endpoint = "speech.googleapis.com" if batch.location == "global" else f"{batch.location}-speech.googleapis.com"
    speech = SpeechClient(client_options=ClientOptions(api_endpoint=endpoint))
    return speech, storage.Client()


def _project(cfg: Config) -> str:
    import google.auth

    _, project = google.auth.default()
    if not project:
        raise GoogleBatchUnavailable("no GCP project found in the active credentials")
    return project


def upload(paths: list[Path], cfg: Config) -> dict[str, str]:
    """Upload each episode's FLAC, returning ``{video_id: gs:// uri}``.

    A blob already present is left alone: uploads are the slow part of this
    backend, and a re-run after a failed transcription should not repeat them.
    """
    from google.cloud import storage

    batch = cfg.asr.google_batch
    client = storage.Client()
    bucket = client.bucket(batch.bucket)

    def one(path: Path) -> tuple[str, str]:
        name = f"{batch.prefix}/audio/{path.name}"
        blob = bucket.blob(name)
        if not blob.exists():
            blob.upload_from_filename(str(path))
        return path.stem, f"gs://{batch.bucket}/{name}"

    uris: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, batch.upload_workers)) as pool:
        for i, (video_id, uri) in enumerate(pool.map(one, paths), 1):
            uris[video_id] = uri
            if i % 20 == 0 or i == len(paths):
                log.info("uploaded %d/%d", i, len(paths))
    return uris


def _request(uri: str, cfg: Config, project: str):
    from google.cloud.speech_v2.types import cloud_speech

    batch = cfg.asr.google_batch
    strategy = (
        cloud_speech.BatchRecognizeRequest.ProcessingStrategy.DYNAMIC_BATCHING
        if batch.dynamic_batching
        else cloud_speech.BatchRecognizeRequest.ProcessingStrategy.PROCESSING_STRATEGY_UNSPECIFIED
    )
    return cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{project}/locations/{batch.location}/recognizers/_",
        config=cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=list(batch.language_codes),
            model=batch.model,
            features=cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_word_confidence=batch.enable_word_confidence,
                enable_automatic_punctuation=batch.enable_automatic_punctuation,
            ),
        ),
        files=[cloud_speech.BatchRecognizeFileMetadata(uri=uri)],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
        processing_strategy=strategy,
    )


def transcribe(uris: dict[str, str], cfg: Config) -> dict[str, list[BatchWord]]:
    """Submit every episode and collect the words, keyed by video id.

    Each episode is submitted and awaited on its own thread. One episode that
    fails is logged and skipped rather than losing the batch -- the caller can
    re-run, and the uploads it would need are already in place.
    """
    speech, _ = _clients(cfg)
    project = _project(cfg)
    batch = cfg.asr.google_batch

    def one(item: tuple[str, str]) -> tuple[str, list[BatchWord]]:
        video_id, uri = item
        try:
            operation = speech.batch_recognize(request=_request(uri, cfg, project))
            response = operation.result(timeout=batch.timeout)
        except Exception as exc:  # noqa: BLE001 - one episode must not sink the run
            log.error("batch transcription failed for %s: %s", video_id, exc)
            return video_id, []
        return video_id, _parse(response)

    results: dict[str, list[BatchWord]] = {}
    items = sorted(uris.items())
    with ThreadPoolExecutor(max_workers=max(1, batch.max_workers)) as pool:
        for i, (video_id, words) in enumerate(pool.map(one, items), 1):
            results[video_id] = words
            log.info("[%d/%d] %s: %d words", i, len(items), video_id, len(words))
    return results


def _parse(response) -> list[BatchWord]:
    """Flatten a BatchRecognize response into a flat, time-ordered word list."""
    words: list[BatchWord] = []
    for file_result in response.results.values():
        for result in file_result.transcript.results:
            if not result.alternatives:
                continue
            for word in result.alternatives[0].words:
                words.append(
                    BatchWord(
                        text=word.word,
                        start=word.start_offset.total_seconds(),
                        end=word.end_offset.total_seconds(),
                        confidence=float(word.confidence),
                    )
                )
    words.sort(key=lambda w: w.start)
    return words


# -- persistence ---------------------------------------------------------
def save_words(path: Path, words: list[BatchWord]) -> None:
    """Cache one episode's words so a re-run never re-pays for the audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"t": w.text, "s": round(w.start, 3), "e": round(w.end, 3), "c": round(w.confidence, 4)} for w in words]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_words(path: Path) -> list[BatchWord]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [BatchWord(text=w["t"], start=w["s"], end=w["e"], confidence=w.get("c", 0.0)) for w in raw]


def assign_to_rows(rows: list[dict], words: list[BatchWord]) -> int:
    """Cut one episode's words into its chunk rows by time overlap.

    A word belongs to the chunk holding the majority of its span -- the same
    rule the subtitle path uses, so a chunk never inherits a word that mostly
    belongs to its neighbour. Rows are matched on ``start``/``end``, which the
    GPU pass already wrote.
    """
    if not rows:
        return 0

    ordered = sorted(rows, key=lambda r: float(r.get("start") or 0.0))
    buckets: dict[int, list[BatchWord]] = {}
    for word in words:
        span = max(word.end - word.start, 1e-6)
        for i, row in enumerate(ordered):
            start, end = float(row.get("start") or 0.0), float(row.get("end") or 0.0)
            if word.end <= start or word.start >= end:
                continue
            if min(word.end, end) - max(word.start, start) >= span * 0.5:
                buckets.setdefault(i, []).append(word)

    filled = 0
    for i, row in enumerate(ordered):
        collected = buckets.get(i, [])
        row["text_cohere"] = " ".join(w.text for w in collected)
        # Every row of a transcribed episode leaves "pending", including one no
        # word covered. Otherwise a silent chunk is indistinguishable from an
        # episode that was never sent, and would wait forever instead of being
        # dropped as empty.
        row["text_source"] = "google_batch"
        if collected:
            row["asr_confidence"] = round(sum(w.confidence for w in collected) / len(collected), 5)
            row["aligned_words"] = len(collected)
            filled += 1
        else:
            row["asr_confidence"] = None
            row["aligned_words"] = 0
    return filled
