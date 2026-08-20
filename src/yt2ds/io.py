"""Dataset layout, atomic writes, and per-video resume state."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Workspace:
    """Directory layout for one dataset run.

    ``root`` holds the deliverable (``wavs/`` + ``metadata.jsonl``); ``work``
    holds everything intermediate and is safe to delete once a run finishes.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    # -- deliverable -----------------------------------------------------
    @property
    def wavs(self) -> Path:
        return self.root / "wavs"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.jsonl"

    @property
    def rejected(self) -> Path:
        return self.root / "rejected.jsonl"

    @property
    def speakers(self) -> Path:
        return self.root / "speakers.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def failed(self) -> Path:
        """URLs that failed, in `--urls-file` format so they can be re-fed."""
        return self.root / "failed.txt"

    # -- intermediates ---------------------------------------------------
    @property
    def work(self) -> Path:
        return self.root / ".work"

    @property
    def raw(self) -> Path:
        return self.work / "raw"

    @property
    def mp3(self) -> Path:
        return self.root / "mp3"

    @property
    def audio(self) -> Path:
        return self.work / "audio"

    @property
    def subs(self) -> Path:
        return self.work / "subs"

    @property
    def state(self) -> Path:
        return self.work / "state"

    @property
    def embeddings(self) -> Path:
        return self.work / "embeddings"

    @property
    def asr_audio(self) -> Path:
        """Per-episode FLAC awaiting upload for batch transcription."""
        return self.work / "asr-audio"

    @property
    def asr_words(self) -> Path:
        """Word timings returned by batch transcription, cached per episode."""
        return self.work / "asr-words"

    def create(self) -> "Workspace":
        for d in (
            self.root,
            self.wavs,
            self.work,
            self.raw,
            self.mp3,
            self.audio,
            self.subs,
            self.state,
            self.embeddings,
            self.asr_audio,
            self.asr_words,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # -- resume ----------------------------------------------------------
    def state_file(self, video_id: str) -> Path:
        return self.state / f"{video_id}.json"

    def load_state(self, video_id: str) -> dict[str, Any]:
        path = self.state_file(video_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A run killed mid-write leaves a truncated file; treat it as absent
            # so the video is simply reprocessed.
            return {}

    def save_state(self, video_id: str, state: dict[str, Any]) -> None:
        write_json(self.state_file(video_id), state)

    def is_complete(self, video_id: str) -> bool:
        return bool(self.load_state(video_id).get("complete"))


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically so an interrupted run never leaves a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlWriter:
    """Append-only JSONL writer that flushes each record.

    Records are flushed and fsynced per batch so that a crash mid-run leaves a
    valid file and ``--resume`` can trust what is already on disk.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self) -> "JsonlWriter":
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def write(self, record: dict[str, Any]) -> None:
        assert self._fh is not None, "JsonlWriter used outside its context manager"
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_all(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.write(record)
        self.flush()

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self.flush()
            self._fh.close()
            self._fh = None


def drop_video_records(path: Path, video_id: str) -> int:
    """Remove every record for ``video_id`` from a JSONL file.

    Used when reprocessing a video that a previous run left half-written, so
    ``--resume`` cannot produce duplicate rows.
    """
    path = Path(path)
    if not path.exists():
        return 0
    kept: list[str] = []
    removed = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                if json.loads(stripped).get("video_id") == video_id:
                    removed += 1
                    continue
            except json.JSONDecodeError:
                removed += 1
                continue
            kept.append(stripped)
    if removed:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(kept) + ("\n" if kept else ""))
        os.replace(tmp, path)
    return removed
