"""Take the music out of the speech, rather than the chunk out of the dataset.

The music detector in ``music.py`` answers "is there a music bed here?" and the
pipeline used to act on that by dropping the chunk. For drama, interviews and
anything else scored under a soundtrack that throws away most of the usable
speech: the voice is fine, it simply has an orchestra behind it.

This stage runs a Mel-Band RoFormer vocal isolator over the *whole* video
before segmentation and replaces the working audio with the isolated voice.
Everything downstream -- VAD, diarization, alignment, ASR, and the clips that
are finally written -- then sees speech with the music bed removed.

Why the whole video and not per chunk: these models reconstruct the voice from
spectral context, and context that reaches across a chunk boundary is context
they get to use. Running once per video is also cheaper than running once per
chunk, at roughly 40x realtime on an Ada-class card.

Why a second separator when ``music.py`` already loads Demucs: they are asked
different questions. Demucs answers a cheap four-stem energy ratio -- good
enough to *score* a chunk. Mel-Band RoFormer is ~2.3 dB better on vocal SDR,
which is what matters when the separated voice is the thing being kept.

The music scores are still computed afterwards, against the isolated audio, so
``music_score`` and ``accompaniment_ratio`` in the metadata read as *residual*
music: what the isolator failed to remove.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from ..config import Config
from ..io import Workspace
from ..models import ModelRegistry
from .audio import FfmpegError, PreparedAudio, _run, measure_loudness, probe_duration

log = logging.getLogger(__name__)

# The isolators are trained and run at 44.1 kHz.
MODEL_SR = 44100

# Beyond this, the isolated file no longer lines up with the timestamps every
# other stage produced against the original decode.
MAX_DRIFT_SECONDS = 0.25


class SeparationError(RuntimeError):
    pass


def apply(
    prepared: PreparedAudio,
    ws: Workspace,
    cfg: Config,
    registry: ModelRegistry,
) -> PreparedAudio:
    """Replace the working audio with its isolated voice.

    Returns ``prepared`` unchanged when separation is disabled, and raises
    ``SeparationError`` when it was asked for but could not be done -- a
    silently un-separated video would land in the dataset with its music bed
    intact and nothing to show for it.
    """
    if not cfg.separate.enabled:
        return prepared

    video_id = prepared.video_id
    master = ws.audio / f"{video_id}.{prepared.master_sr // 1000}k.vocals.wav"
    work = ws.audio / f"{video_id}.{prepared.work_sr // 1000}k.vocals.wav"

    if not (master.exists() and work.exists()):
        isolated = _isolate(prepared.master_path, ws, cfg, registry)
        try:
            _decode(isolated, [(master, prepared.master_sr), (work, prepared.work_sr)])
        finally:
            isolated.unlink(missing_ok=True)

    duration = probe_duration(master)
    drift = abs(duration - prepared.duration)
    if drift > MAX_DRIFT_SECONDS:
        raise SeparationError(
            f"{video_id}: isolated audio is {duration:.2f}s against {prepared.duration:.2f}s of source "
            f"({drift:.2f}s drift); timestamps would not line up"
        )

    lufs, gain_db = measure_loudness(work, cfg.audio.target_lufs)
    log.info(
        "%s: voice isolated with %s (%.1f -> %.1f LUFS)",
        video_id,
        cfg.separate.model,
        prepared.lufs,
        lufs,
    )

    if not cfg.separate.keep_original and not cfg.runtime.keep_intermediates:
        # The original decode has no reader left once the isolated pair exists,
        # and it is the same half-gigabyte-per-hour as the copy replacing it.
        for path in (prepared.master_path, prepared.work_path):
            path.unlink(missing_ok=True)
        source_master: Path | None = None
        source_work: Path | None = None
    else:
        source_master = prepared.master_path
        source_work = prepared.work_path

    return replace(
        prepared,
        master_path=master,
        work_path=work,
        lufs=lufs,
        gain_db=gain_db,
        separated=True,
        source_master_path=source_master,
        source_work_path=source_work,
    )


def _isolate(source: Path, ws: Workspace, cfg: Config, registry: ModelRegistry) -> Path:
    """Run the isolator over ``source``, returning the 44.1 kHz voice stem."""
    separator = registry.separator
    name = f"{source.stem}.vocals.raw"

    # The output directory is fixed at construction, but the pipeline writes
    # one workspace per run, so it is repointed here rather than reloading the
    # model. audio-separator does the same thing internally for ensembles.
    separator.output_dir = str(ws.audio)
    if separator.model_instance is not None:
        separator.model_instance.output_dir = str(ws.audio)

    # `separate` catches its own exceptions and returns an empty list, so the
    # result is what has to be checked.
    outputs = separator.separate(str(source), custom_output_names={cfg.separate.stem: name})
    if not outputs:
        raise SeparationError(f"{cfg.separate.model} produced no output for {source.name}")

    produced = ws.audio / outputs[0] if not Path(outputs[0]).is_absolute() else Path(outputs[0])
    if not produced.exists():
        raise SeparationError(f"{cfg.separate.model} reported {produced} but it is not on disk")
    return produced


def _decode(source: Path, targets: list[tuple[Path, int]]) -> None:
    """Write mono copies of ``source`` at each requested rate, in one pass."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(source), "-vn"]
    for path, rate in targets:
        cmd += ["-map", "0:a:0", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(path)]
    try:
        _run(cmd)
    except FfmpegError as exc:
        raise SeparationError(f"could not decode the isolated voice: {exc}") from exc
