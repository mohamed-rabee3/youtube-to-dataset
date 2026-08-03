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


@dataclass
class SegmentConfig:
    min_duration: float = 2.0
    max_duration: float = 12.0
    merge_gap: float = 0.3
    pad: float = 0.2
    min_split_pause: float = 0.25


@dataclass
class MusicConfig:
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
class AsrConfig:
    model: str = "CohereLabs/cohere-transcribe-arabic-07-2026"
    # Decoded once per language; highest-confidence result wins. See
    # configs/default.yaml for why this is not auto-detected.
    languages: list[str] = field(default_factory=lambda: ["ar", "en"])
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
    # Mean per-token log-probability floor, applied only on the Cohere-only
    # path where there is no second transcript to compare against.
    min_asr_confidence: float = -1.0
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
