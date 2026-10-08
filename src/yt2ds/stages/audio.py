"""Decode downloaded media into the working audio copies.

Two working copies are produced in one ffmpeg pass:

* ``<id>.48k.wav`` -- the master. Chunks are cut from this, and 48 kHz
  decimates cleanly to the 24 kHz output rate.
* ``<id>.16k.wav`` -- what VAD, diarization, the aligner and Cohere all
  consume.

The archival MP3 the user asked for is written alongside, but nothing
downstream reads it: encoding to MP3 and then cutting TTS clips out of the
decoded result would bake compression artefacts into training data. Set
``audio.chunk_from_mp3`` if that trade is wanted anyway.

Loudness is *measured* here and applied when clips are written, by
:func:`render_clip`: per clip by default, so the whole dataset sits at one
level, or per video (``audio.loudness: video``) to keep a speaker's loud and
quiet passages in proportion. Either way the gain is capped so no written
sample exceeds ``audio.peak_ceiling_dbfs``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AudioConfig, Config
from ..io import Workspace
from .download import VideoAssets

log = logging.getLogger(__name__)


@dataclass
class PreparedAudio:
    video_id: str
    master_path: Path
    work_path: Path
    mp3_path: Path | None
    duration: float
    master_sr: int
    work_sr: int
    # Measured integrated loudness of the source, and the gain that moves it to
    # the configured target. Applied when chunks are written.
    lufs: float
    gain_db: float
    # Set once vocal isolation has run: `master_path` and `work_path` then
    # point at the isolated voice, and these at the untouched decode.
    separated: bool = False
    source_master_path: Path | None = None
    source_work_path: Path | None = None


class FfmpegError(RuntimeError):
    pass


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FfmpegError(f"{cmd[0]} failed: {proc.stderr.strip()[-2000:]}")
    return proc.stdout


def probe_duration(path: Path) -> float:
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        return float(json.loads(out)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def has_encoder(name: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return name in out


def prepare(assets: VideoAssets, ws: Workspace, cfg: Config) -> PreparedAudio:
    """Decode ``assets.audio_path`` into the working copies."""
    if assets.audio_path is None:
        raise FfmpegError(f"no audio to prepare for {assets.video_id}")

    master_sr = cfg.audio.out_sample_rate * 2  # 48 kHz for a 24 kHz deliverable
    work_sr = cfg.audio.work_sample_rate
    master = ws.audio / f"{assets.video_id}.{master_sr // 1000}k.wav"
    work = ws.audio / f"{assets.video_id}.{work_sr // 1000}k.wav"
    # An MP3 is written when it is wanted as an archive, or when chunks are to
    # be cut from it. In the second case the pipeline deletes it afterwards.
    want_mp3 = cfg.audio.keep_mp3 or cfg.audio.chunk_from_mp3
    mp3 = ws.mp3 / f"{assets.video_id}.mp3" if want_mp3 else None

    needed = [p for p in (master, work) if not p.exists()]
    if mp3 is not None and not mp3.exists():
        needed.append(mp3)

    if needed:
        # One decode, three outputs. `highpass=f=20` strips DC offset without
        # touching anything audible.
        cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(assets.audio_path), "-vn"]
        for path, rate in ((master, master_sr), (work, work_sr)):
            cmd += ["-map", "0:a:0", "-ac", "1", "-af", "highpass=f=20", "-ar", str(rate), "-c:a", "pcm_s16le", str(path)]
        if mp3 is not None:
            cmd += ["-map", "0:a:0", "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-q:a", "2", str(mp3)]
        _run(cmd)

    duration = probe_duration(master)
    lufs, gain_db = measure_loudness(work, cfg.audio.target_lufs)

    return PreparedAudio(
        video_id=assets.video_id,
        master_path=master,
        work_path=work,
        mp3_path=mp3,
        duration=duration,
        master_sr=master_sr,
        work_sr=work_sr,
        lufs=lufs,
        gain_db=gain_db,
    )


def measure_loudness(path: Path, target_lufs: float) -> tuple[float, float]:
    """Integrated loudness (ITU-R BS.1770) and the gain to reach the target.

    Re-measured after vocal isolation: pulling the music bed out drops the
    integrated loudness of a scene by several LU, and the gain applied at emit
    time has to be the one that suits the voice that is actually kept.
    """
    import pyloudnorm
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size < file_sr:
        # Under a second of audio: BS.1770 gating has nothing to work with.
        return float("nan"), 0.0

    meter = pyloudnorm.Meter(file_sr)
    try:
        lufs = float(meter.integrated_loudness(data))
    except Exception as exc:  # noqa: BLE001 - pyloudnorm raises bare exceptions
        log.warning("loudness measurement failed for %s: %s", path.name, exc)
        return float("nan"), 0.0

    if not np.isfinite(lufs):
        return float("nan"), 0.0
    return lufs, float(target_lufs - lufs)


def read_segment(path: Path, sr: int, start: float, end: float) -> np.ndarray:
    """Read ``[start, end)`` seconds as mono float32, clamped to the file."""
    import soundfile as sf

    with sf.SoundFile(str(path)) as fh:
        total = len(fh)
        begin = max(0, int(round(start * sr)))
        stop = min(total, int(round(end * sr)))
        if stop <= begin:
            return np.zeros(0, dtype=np.float32)
        fh.seek(begin)
        data = fh.read(stop - begin, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


@dataclass
class RenderedClip:
    samples: np.ndarray
    # Gain applied, and how much of the wanted gain the peak ceiling withheld.
    gain_db: float
    gain_limited_db: float
    # Loudness and peak of the clip as written.
    lufs: float | None
    peak_dbfs: float | None


def render_clip(
    samples: np.ndarray,
    source_sr: int,
    audio_cfg: AudioConfig,
    video_gain_db: float,
) -> RenderedClip:
    """Turn a cut of the master into the clip that is written.

    Order matters. Resampling comes first, so the peak ceiling is enforced on
    the samples that are actually written -- a resampler can overshoot the
    source's peak, and a ceiling checked before it would not hold after. The
    gain is then the configured loudness target, backed off only as far as the
    ceiling demands, so no sample is ever hard-clipped by the 16-bit writer.
    """
    out_sr = audio_cfg.out_sample_rate
    samples = resample(samples, source_sr, out_sr)
    if samples.size == 0:
        return RenderedClip(samples, 0.0, 0.0, None, None)

    mode = audio_cfg.loudness.lower()
    if mode == "clip":
        measured = integrated_loudness(samples, out_sr)
        wanted = audio_cfg.target_lufs - measured if measured is not None else video_gain_db
    elif mode == "video":
        wanted = video_gain_db
    else:
        raise ValueError(f"unknown audio.loudness: {audio_cfg.loudness!r} (expected 'clip' or 'video')")
    wanted = wanted if np.isfinite(wanted) else 0.0

    peak = float(np.max(np.abs(samples)))
    allowed = audio_cfg.peak_ceiling_dbfs - 20 * np.log10(peak) if peak > 0 else wanted
    gain = min(wanted, allowed)
    out = (samples * 10.0 ** (gain / 20.0)).astype(np.float32)
    apply_fades(out, out_sr, audio_cfg.fade_ms)

    out_peak = float(np.max(np.abs(out)))
    return RenderedClip(
        samples=out,
        gain_db=round(float(gain), 3),
        gain_limited_db=round(float(max(0.0, wanted - gain)), 3),
        lufs=_round(integrated_loudness(out, out_sr)),
        peak_dbfs=round(20 * np.log10(out_peak), 3) if out_peak > 0 else None,
    )


def apply_fades(samples: np.ndarray, sr: int, fade_ms: float) -> None:
    """Raised-cosine fade-in and fade-out, in place."""
    n = min(int(round(fade_ms * sr / 1000.0)), samples.size // 4)
    if n <= 1:
        return
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))).astype(np.float32)
    samples[:n] *= ramp
    samples[-n:] *= ramp[::-1]


def integrated_loudness(samples: np.ndarray, sr: int) -> float | None:
    """BS.1770 integrated loudness, or None where it is undefined.

    Undefined means under one 400 ms gating block, or nothing above the
    absolute gate -- silence has no loudness to normalize to.
    """
    import pyloudnorm

    if samples.size < int(0.4 * sr):
        return None
    try:
        value = float(pyloudnorm.Meter(sr).integrated_loudness(samples.astype(np.float64)))
    except Exception:  # noqa: BLE001 - pyloudnorm raises bare exceptions
        return None
    return value if np.isfinite(value) else None


def _round(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono float32 audio. Uses soxr via librosa for quality."""
    if src_sr == dst_sr or samples.size == 0:
        return samples
    import librosa

    return librosa.resample(samples, orig_sr=src_sr, target_sr=dst_sr, res_type="soxr_hq").astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    """Write 16-bit mono PCM."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    sf.write(str(path), clipped, sr, subtype="PCM_16")
