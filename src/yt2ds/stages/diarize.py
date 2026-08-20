"""Speaker diarization, executed out-of-process.

DiariZen is the most accurate open-source diarizer available (DER 9.1 on
VoxConverse, 14.5 on DIHARD 3 -- better than pyannote community-1 across the
board), but it vendors pyannote-audio against an older torch than the
transformers 5.x stack Cohere needs. Rather than fight that, it lives in its
own virtualenv and is driven through ``scripts/diarize_worker.py``.

Licence note: DiariZen's *weights* are CC BY-NC 4.0 (non-commercial). The code
here talks to a generic worker contract, so pointing ``diarize.model`` at a
differently-licensed diarizer is a configuration change, not a rewrite.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..io import Workspace
from ..models import ModelRegistry
from .vad import Region

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKER = REPO_ROOT / "scripts" / "diarize_worker.py"
DIARIZE_VENV_PYTHON = REPO_ROOT / ".venv-diarize" / "bin" / "python"


@dataclass
class Turn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Diarization:
    turns: list[Turn] = field(default_factory=list)
    overlaps: list[Region] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    model: str = ""
    available: bool = True

    def speaker_seconds(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for turn in self.turns:
            totals[turn.speaker] = totals.get(turn.speaker, 0.0) + turn.duration
        return totals


class DiarizerUnavailable(RuntimeError):
    pass


def worker_python() -> Path:
    """Interpreter that has DiariZen installed.

    ``YT2DS_DIARIZE_PYTHON`` overrides, which is what the tests and any
    non-default venv layout use.
    """
    override = os.environ.get("YT2DS_DIARIZE_PYTHON")
    if override:
        return Path(override)
    return DIARIZE_VENV_PYTHON


def is_available() -> bool:
    return worker_python().exists() and WORKER.exists()


def run(
    work_path: Path,
    ws: Workspace,
    cfg: Config,
    video_id: str,
    registry: ModelRegistry | None = None,
) -> Diarization:
    """Diarize the 16 kHz working copy, caching the result on disk."""
    cache = ws.work / "diarization" / f"{video_id}.json"
    if cache.exists():
        try:
            return _parse(json.loads(cache.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            log.warning("discarding unreadable diarization cache for %s", video_id)
            cache.unlink(missing_ok=True)

    python = worker_python()
    if not python.exists():
        raise DiarizerUnavailable(
            f"DiariZen interpreter not found at {python}. Run scripts/setup.sh, "
            "or set YT2DS_DIARIZE_PYTHON to an interpreter that has diarizen installed."
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    log.info("diarizing %s", video_id)

    # The worker is a separate process, so it only gets the GPU memory this one
    # is not holding. Hand back the allocator's unused blocks first, or a long
    # video -- whose scoring batches grew the cache to tens of gigabytes --
    # starves the diarizer that follows it.
    if registry is not None:
        registry.free_cached_memory()

    ladder = _batch_sizes(cfg)
    last_error = ""
    for attempt, batch_size in enumerate(ladder, start=1):
        cmd = [
            str(python),
            str(WORKER),
            str(work_path),
            str(cache),
            "--model",
            cfg.diarize.model,
            "--device",
            cfg.runtime.device,
        ]
        if batch_size > 0:
            cmd += ["--batch-size", str(batch_size)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            if proc.stderr.strip():
                log.info("diarizer: %s", proc.stderr.strip().splitlines()[-1])
            return _parse(json.loads(cache.read_text(encoding="utf-8")))

        last_error = proc.stderr.strip()[-2000:]
        if not _is_out_of_memory(last_error) or attempt >= len(ladder):
            break
        next_batch = ladder[attempt]
        log.warning(
            "%s: diarizer ran out of GPU memory at batch size %s; retrying at %d",
            video_id,
            batch_size or "default",
            next_batch,
        )

    raise RuntimeError(f"diarization worker failed for {video_id}: {last_error}")


def _batch_sizes(cfg: Config) -> list[int]:
    """Batch sizes to try, in order. 0 means the model config's own default."""
    first = getattr(cfg.diarize, "batch_size", 0)
    ladder = [first]
    size = first if first > 0 else 32
    while size > 2:
        size //= 2
        ladder.append(size)
    return ladder


def _is_out_of_memory(stderr: str) -> bool:
    """Whether a worker failure was memory pressure rather than a real fault.

    pyannote catches the CUDA OOM itself and re-raises it as a MemoryError
    advising a smaller batch size, so both spellings have to be recognised.
    """
    lowered = stderr.lower()
    return "memoryerror" in lowered or "out of memory" in lowered or "batch_size" in lowered


def _parse(data: dict) -> Diarization:
    return Diarization(
        turns=[Turn(float(t["start"]), float(t["end"]), str(t["speaker"])) for t in data.get("turns", [])],
        overlaps=[Region(float(o["start"]), float(o["end"])) for o in data.get("overlaps", [])],
        speakers=list(data.get("speakers", [])),
        model=data.get("model", ""),
    )


def single_speaker(duration: float, label: str = "SPEAKER_00") -> Diarization:
    """Fallback annotation covering the whole file with one speaker."""
    return Diarization(
        turns=[Turn(0.0, duration, label)],
        overlaps=[],
        speakers=[label],
        model="none",
        available=False,
    )


def collapse_if_dominant(diarization: Diarization, cfg: Config) -> Diarization:
    """Merge to a single speaker when one cluster owns nearly all the speech.

    Guards against a diarizer that split one voice across several clusters on a
    monologue, which would otherwise fragment that voice in the dataset.
    """
    totals = diarization.speaker_seconds()
    if len(totals) < 2:
        return diarization
    total = sum(totals.values())
    if total <= 0:
        return diarization

    dominant, dominant_seconds = max(totals.items(), key=lambda kv: kv[1])
    if dominant_seconds / total < cfg.diarize.collapse_single_speaker_ratio:
        return diarization

    log.info(
        "collapsing %d speakers to %s (%.0f%% of speech)",
        len(totals),
        dominant,
        100 * dominant_seconds / total,
    )
    return Diarization(
        turns=[Turn(t.start, t.end, dominant) for t in diarization.turns],
        overlaps=diarization.overlaps,
        speakers=[dominant],
        model=diarization.model,
        available=diarization.available,
    )
