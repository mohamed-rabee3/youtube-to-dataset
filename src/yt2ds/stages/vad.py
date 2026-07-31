"""Voice activity detection with Silero VAD.

Produces coarse speech regions over the whole file. These are intersected with
speaker turns in :mod:`yt2ds.stages.segment`; nothing here decides final chunk
boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..models import ModelRegistry

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect(work_path: Path, cfg: Config, registry: ModelRegistry) -> list[Region]:
    """Return speech regions in seconds, ordered by start time."""
    import soundfile as sf

    model, get_speech_timestamps = registry.vad
    sr = cfg.audio.work_sample_rate

    data, file_sr = sf.read(str(work_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        raise ValueError(f"expected {sr} Hz work audio, got {file_sr} from {work_path}")
    if data.size == 0:
        return []

    wav = torch.from_numpy(np.ascontiguousarray(data))
    stamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        threshold=cfg.vad.threshold,
        min_speech_duration_ms=cfg.vad.min_speech_duration_ms,
        min_silence_duration_ms=cfg.vad.min_silence_duration_ms,
        speech_pad_ms=cfg.vad.speech_pad_ms,
        return_seconds=True,
    )

    regions = [Region(float(s["start"]), float(s["end"])) for s in stamps]
    total = sum(r.duration for r in regions)
    log.info(
        "%s: %d speech regions, %.1fs speech of %.1fs (%.0f%%)",
        work_path.stem,
        len(regions),
        total,
        len(data) / sr,
        100 * total / max(len(data) / sr, 1e-9),
    )
    return regions


def intersect(regions: list[Region], start: float, end: float) -> list[Region]:
    """Clip a region list to a window, dropping anything that falls outside."""
    out: list[Region] = []
    for region in regions:
        lo = max(region.start, start)
        hi = min(region.end, end)
        if hi > lo:
            out.append(Region(lo, hi))
    return out


def subtract(regions: list[Region], holes: list[Region]) -> list[Region]:
    """Remove ``holes`` from ``regions``. Used to cut out overlapped speech."""
    if not holes:
        return list(regions)

    ordered = sorted(holes, key=lambda r: r.start)
    out: list[Region] = []
    for region in regions:
        cursor = region.start
        for hole in ordered:
            if hole.end <= cursor or hole.start >= region.end:
                continue
            if hole.start > cursor:
                out.append(Region(cursor, min(hole.start, region.end)))
            cursor = max(cursor, hole.end)
            if cursor >= region.end:
                break
        if cursor < region.end:
            out.append(Region(cursor, region.end))
    return [r for r in out if r.duration > 0]
