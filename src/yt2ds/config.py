"""Configuration: YAML profile plus CLI overrides.

Every stage reads its thresholds from here rather than hard-coding them, so a
run can be re-tuned without touching code.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class AudioConfig:
    out_sample_rate: int = 24000
    work_sample_rate: int = 16000
    demucs_sample_rate: int = 44100
    target_lufs: float = -23.0
    # Off by default: nothing downstream reads the MP3, and at ~27 MB per
    # video an archive of a long run costs more disk than the dataset it
    # accompanies. Turn it on to keep a full-length copy of each source.
    keep_mp3: bool = False
    chunk_from_mp3: bool = False


@dataclass
class DownloadConfig:
    sub_langs: list[str] = field(default_factory=lambda: ["ar", "ar-SA", "ar-EG", "ar.*"])
    prefer_manual_subs: bool = True
    sub_format: str = "json3/vtt/best"
    retries: int = 5
    sleep_interval: int = 1
    cookies_from_browser: str | None = None
    # Netscape-format cookies.txt. The single most effective fix for
    # "Sign in to confirm you're not a bot" on a datacentre IP.
    cookies_file: str | None = None
    # YouTube answers each of its own clients differently: the one that gets a
    # 403 on fragments or a bot check is often not the one that succeeds. Each
    # entry is a whole fresh attempt at the video; "default" is yt-dlp's own
    # client choice. Order matters -- cheapest and most reliable first.
    player_clients: list[str] = field(
        default_factory=lambda: ["default", "tv_simply", "web_safari", "android_vr", "ios", "mweb"]
    )
    # Seconds to wait after a failed attempt, multiplied by the attempt number.
    attempt_backoff: float = 4.0
    socket_timeout: int = 30
    # Live and upcoming streams have no stable audio to cut a dataset from.
    skip_live: bool = True
    # A `watch?v=...&list=...` URL expands to the whole playlist, as yt-dlp
    # itself does. Set false (`--no-playlist`) to take only the named video.
    follow_playlist: bool = True


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 200
    speech_pad_ms: int = 100


@dataclass
class DiarizeConfig:
    model: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    drop_overlap: bool = True
    collapse_single_speaker_ratio: float = 0.9
    # Segmentation/embedding batch size for the worker. 0 keeps the model
    # config's own (32), which is halved on each retry after an OOM.
    batch_size: int = 0


@dataclass
class SegmentConfig:
    min_duration: float = 2.0
    max_duration: float = 12.0
    merge_gap: float = 0.3
    pad: float = 0.2
    min_split_pause: float = 0.25


@dataclass
class SeparateConfig:
    """Vocal isolation: take the music out of the speech, keep the speech."""

    enabled: bool = True
    # audio-separator model filename. Mel-Band RoFormer (Kim, FT by unwa) sits
    # within 0.1 dB of the best vocal SDR on MUSDB (11.53 vs 11.63 for
    # BS-Roformer-1296) and separates about twice as fast; both are ~2.3 dB
    # ahead of the htdemucs the music *detector* uses.
    model: str = "mel_band_roformer_kim_ft_unwa.ckpt"
    # Where model weights are cached. ~640 MB for the default model.
    model_dir: str = ".cache/audio-separator"
    # The stem to keep. "Vocals" is the model's name for the voice.
    stem: str = "Vocals"
    # These models run at 44.1 kHz; the isolated voice is resampled back to
    # the pipeline's 48 kHz master and 16 kHz working rates.
    segment_size: int = 256
    overlap: int = 8
    batch_size: int = 1
    # Keep the untouched master beside the isolated voice. Off by default:
    # nothing downstream reads it once separation has run.
    keep_original: bool = False


@dataclass
class MusicConfig:
    # Whether a musical chunk is dropped. Off by default, because separation
    # removes the music rather than the chunk; the scores are still recorded.
    reject: bool = False
    max_accompaniment_ratio: float = 0.10
    ast_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    max_music_score: float = 0.20
    music_labels: list[str] = field(
        default_factory=lambda: [
            "Music",
            "Singing",
            "Musical instrument",
            "Applause",
            "Cheering",
            "Crowd",
            "Theme music",
            "Background music",
            "Drum",
            "Guitar",
            "Piano",
        ]
    )


@dataclass
class SpeakersConfig:
    embed_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    max_centroid_distance: float = 0.45
    link_threshold: float = 0.35
    min_speaker_seconds: float = 30.0


@dataclass
class GoogleAsrConfig:
    """Google Cloud Speech-to-Text, the default transcription backend."""

    # "latest_long" is Google's general long-form model. "latest_short",
    # "chirp_2" and the rest are also accepted; whatever the v1 API takes.
    model: str = "latest_long"
    # Short language code (as used in asr.languages) -> BCP-47 locale, which
    # is what the API actually wants. Anything not listed is sent through
    # unchanged, so asr.languages can also name a locale directly.
    locales: dict[str, str] = field(default_factory=lambda: {"ar": "ar-SA", "en": "en-US"})
    # Unlike Cohere, the API detects among languages itself: the first entry
    # of asr.languages is the primary and the rest go in as alternatives, so
    # each chunk costs one request rather than one per language.
    use_alternative_languages: bool = True
    enable_automatic_punctuation: bool = True
    profanity_filter: bool = False
    # Chunks are 2-12 s, so this is one small request each. Concurrency is
    # what makes that tolerable; raise it if the quota allows.
    max_workers: int = 8
    max_retries: int = 4
    # Seconds before the first retry, doubled each attempt.
    retry_backoff: float = 2.0
    timeout: float = 60.0
    # Service-account JSON. Unset falls back to GOOGLE_APPLICATION_CREDENTIALS
    # and then to application default credentials.
    credentials_file: str | None = None


@dataclass
class GoogleBatchConfig:
    """Speech-to-Text **V2** BatchRecognize, whole-episode and off the GPU.

    The per-chunk backends pay twice over: Google bills each request rounded up
    to 15 seconds, and chunks here average under 5, so a corpus transcribed
    clip by clip is billed for roughly three times the audio it contains.
    Sending the whole episode instead is billed for its true length, and the
    word-level timestamps that come back are enough to cut the transcript into
    the chunks the diarizer already decided on -- so the per-chunk requests, and
    the forced-alignment pass that would otherwise place the words, both go.

    ``DYNAMIC_BATCHING`` is the discounted tier: about a quarter of the standard
    per-minute price, in exchange for a latency SLA of up to 24 hours. In
    practice it has returned in seconds.
    """

    # ar-SA is served by "long" only in the "global" and "us" locations -- not
    # us-central1, and no Chirp model offers it at all. Verified against the
    # live API; see README. "long" is the V2 name for v1's "latest_long".
    location: str = "global"
    model: str = "long"
    language_codes: list[str] = field(default_factory=lambda: ["ar-SA"])
    # Batch recognition only reads from Cloud Storage, so audio is uploaded
    # first. Unset means the backend cannot run.
    bucket: str | None = None
    # Prefix inside the bucket for uploaded audio and returned transcripts.
    prefix: str = "yt2ds"
    # Keep the uploaded audio after a successful run, so re-transcribing under
    # a different model costs no second upload.
    keep_uploads: bool = True
    # Trade the discounted tier for the standard one by setting this false.
    dynamic_batching: bool = True
    enable_automatic_punctuation: bool = True
    enable_word_confidence: bool = True
    # One episode per request, many requests in flight: the API caps how much
    # it will return inline, and one file per request keeps the mapping from
    # transcript back to episode trivial.
    max_workers: int = 16
    upload_workers: int = 8
    # A 90-minute episode is well inside this; it is the guard against a
    # request that never settles.
    timeout: float = 14400.0


@dataclass
class AsrConfig:
    # "google" (per-chunk Cloud Speech-to-Text v1), "google_batch" (whole-episode
    # V2 BatchRecognize, filled in by `yt2ds transcribe`), or "cohere" (local 2B
    # GPU model).
    backend: str = "google"
    # Decoded once per language; highest-confidence result wins. See
    # configs/default.yaml for why this is not auto-detected.
    languages: list[str] = field(default_factory=lambda: ["ar", "en"])
    google: GoogleAsrConfig = field(default_factory=GoogleAsrConfig)
    google_batch: GoogleBatchConfig = field(default_factory=GoogleBatchConfig)
    # -- cohere backend only --
    model: str = "CohereLabs/cohere-transcribe-arabic-07-2026"
    batch_size: int = 8
    max_new_tokens: int = 256
    dtype: str = "bfloat16"


@dataclass
class AlignConfig:
    model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    batch_size: int = 4
    window_seconds: float = 900.0
    # Measured Arabic speech centres near -3.7; catch only outright failure.
    min_align_score: float = -5.5


@dataclass
class QualityConfig:
    min_mos: float = 3.0
    min_stoi: float = 0.75
    min_pesq: float = 1.8
    min_snr_db: float = 12.0
    max_clipping_ratio: float = 0.001
    min_peak_dbfs: float = -35.0
    max_peak_dbfs: float = -0.5


@dataclass
class FiltersConfig:
    max_cer_yt_vs_cohere: float = 0.35
    # Arabic script is compact; measured speech runs ~5-13 chars/sec.
    min_chars_per_sec: float = 4.0
    max_chars_per_sec: float = 18.0
    min_words: int = 2
    # Fraction of characters that must be in the script of the decoded
    # language (Arabic for "ar", Latin for "en").
    min_script_ratio: float = 0.6
    # Confidence floor for the transcript, applied only on the ASR-only path
    # where there is no second transcript to compare against. The two backends
    # report on different scales and so get separate floors: Cohere's is a mean
    # per-token log-probability (<= 0), Google's is its own 0-1 confidence.
    min_asr_confidence: float = -1.0
    min_asr_confidence_google: float = 0.5
    drop_boundary_clipped_words: bool = True


@dataclass
class RuntimeConfig:
    download_workers: int = 4
    device: str = "cuda"
    keep_models_loaded: bool = True
    # Delete each video's raw download and working WAVs once it is finished.
    # A long run otherwise accumulates roughly 0.5 GB per hour of source audio
    # in .work/ on top of the dataset itself.
    keep_intermediates: bool = False


@dataclass
class Config:
    profile: str = "tts"
    audio: AudioConfig = field(default_factory=AudioConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    separate: SeparateConfig = field(default_factory=SeparateConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    speakers: SpeakersConfig = field(default_factory=SpeakersConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        """Build a Config from a YAML file, then apply dotted-key overrides."""
        path = Path(path) if path else DEFAULT_CONFIG
        data: dict[str, Any] = {}
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        cfg = _from_dict(cls, data)
        for key, value in (overrides or {}).items():
            if value is not None:
                cfg.set_dotted(key, value)
        return cfg

    def set_dotted(self, dotted: str, value: Any) -> None:
        """Set ``music.max_music_score`` style keys, used by CLI overrides."""
        parts = dotted.split(".")
        target: Any = self
        for part in parts[:-1]:
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"unknown config key: {dotted}")
        setattr(target, leaf, value)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass. Unknown keys raise rather than pass silently.

    ``from __future__ import annotations`` turns field types into strings, so
    resolve them with ``get_type_hints`` before checking for nested dataclasses.
    """
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown key(s) in config for {cls.__name__}: {sorted(unknown)}")

    hints = get_type_hints(cls)
    for name in known:
        if name not in data:
            continue
        value = data[name]
        hint = hints.get(name)
        if isinstance(value, dict) and is_dataclass(hint):
            kwargs[name] = _from_dict(hint, value)
        else:
            kwargs[name] = copy.deepcopy(value)
    return cls(**kwargs)


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj
