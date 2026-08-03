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

Loudness is *measured* here and applied at emit time. Normalizing per video
rather than per chunk keeps the relative dynamics between a speaker's loud and
quiet passages intact, which a voice-cloning model needs.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
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
    lufs, gain_db = _measure_loudness(work, work_sr, cfg.audio.target_lufs)

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


def _measure_loudness(path: Path, sr: int, target_lufs: float) -> tuple[float, float]:
    """Integrated loudness (ITU-R BS.1770) and the gain to reach the target."""
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


def apply_gain(samples: np.ndarray, gain_db: float, peak_ceiling_dbfs: float = -1.0) -> np.ndarray:
    """Apply loudness gain, backing it off if it would push the peak into clipping."""
    if not np.isfinite(gain_db) or gain_db == 0.0 or samples.size == 0:
        return samples
    gain = 10.0 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)
        gain = min(gain, ceiling / peak)
    return (samples * gain).astype(np.float32)


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
