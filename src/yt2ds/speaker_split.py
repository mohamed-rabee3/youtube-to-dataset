"""One folder per voice: link speakers across videos and file every clip.

Diarization names speakers per video, so a podcast host across a hundred
episodes arrives as a hundred unrelated ``<video>_SPKn`` labels. This module
pools them into global identities and writes the result as

    <dataset>/speakers/
        report.json                 ranking: hours, clips, episodes, MOS
        GLOBAL_SPEAKER_00/          the voice with the most usable audio
            metadata.jsonl          its rows, audio_file relative to wavs/
            speaker.json
            wavs/*.wav              hardlinks into the dataset's wavs/
        GLOBAL_SPEAKER_01/ ...
        _unassigned/                clips no voice could vouch for

Each speaker folder is a valid dataset root on its own (``wavs/`` plus
``metadata.jsonl``), so it can be transcribed, exported or uploaded alone.
Clips are hardlinked, not copied: the split costs inodes, not a second copy of
the audio, and deleting a speaker folder never touches the dataset.

Three checks decide where a clip goes:

1. **Linking.** Per-video speaker centroids are clustered (average linkage on
   cosine distance, ``speakers.link_threshold``).
2. **Debris.** A global identity pooling under ``speakers.min_speaker_seconds``
   is a diarization scrap -- an intro voice-over, crosstalk -- not a voice.
3. **Per-clip verification.** Every clip's own embedding is compared with the
   centroid of the voice it was linked to; one further than
   ``speakers.max_clip_distance`` was mislabelled by the diarizer and is not
   filed under that voice. A centroid can be right while one of its clips is
   wrong, and one wrong clip is what poisons a cloned voice.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .io import Workspace, read_jsonl, write_json
from .stages.speakers import robust_centroid, link_across_videos

log = logging.getLogger(__name__)

UNASSIGNED = "_unassigned"
_PREFIX = "GLOBAL_SPEAKER_"


@dataclass
class SplitResult:
    """Where every clip went, and the per-voice summary."""

    assignment: dict[str, str] = field(default_factory=dict)  # audio_file -> folder
    summary: dict[str, dict] = field(default_factory=dict)  # folder -> stats


def load_embeddings(ws: Workspace) -> tuple[dict[str, np.ndarray], dict[tuple[str, int], np.ndarray]]:
    """Per-video speaker centroids and per-clip embeddings saved by the run."""
    centroids: dict[str, np.ndarray] = {}
    clips: dict[tuple[str, int], np.ndarray] = {}
    for path in sorted(ws.embeddings.glob("*.npz")):
        with np.load(path) as data:
            if path.name.endswith(".clips.npz"):
                video_id = path.name[: -len(".clips.npz")]
                for index, vector in zip(data["index"], data["embedding"]):
                    clips[(video_id, int(index))] = vector
            else:
                for key in data.files:
                    centroids[key] = data[key]
    return centroids, clips


def split(
    ws: Workspace,
    cfg: Config,
    out_root: Path | None = None,
    only: set[str] | None = None,
    write: bool = True,
) -> SplitResult:
    """Link voices, verify every clip, and (unless ``write`` is false) build the folders."""
    rows = [r for r in read_jsonl(ws.metadata) if r.get("audio_file")]
    centroids, clip_embeddings = load_embeddings(ws)
    if not centroids:
        raise SystemExit(f"no speaker embeddings under {ws.embeddings}; run the pipeline first")

    seconds: dict[str, float] = defaultdict(float)
    for row in rows:
        seconds[str(row.get("speaker"))] += float(row.get("duration") or 0.0)

    sp = cfg.speakers
    linked = _drop_thin(link_across_videos(centroids, sp.link_threshold, weights=seconds), seconds, sp.min_speaker_seconds)
    voice_centroids = _voice_centroids(rows, linked, clip_embeddings, centroids)

    result = SplitResult()
    for row in rows:
        voice = linked.get(str(row.get("speaker")))
        distance = None
        vector = clip_embeddings.get(_clip_key(row))
        if voice is not None and vector is not None and voice in voice_centroids:
            distance = float(1.0 - vector @ voice_centroids[voice])
            if distance > sp.max_clip_distance:
                voice = None
        row["global_speaker"] = voice
        row["global_speaker_distance"] = round(distance, 5) if distance is not None else None
        result.assignment[row["audio_file"]] = voice or UNASSIGNED

    result.summary = _summarize(rows, linked)
    if write:
        root = out_root or ws.root / "speakers"
        _write_folders(ws, root, rows, result, only)
        _rewrite_metadata(ws, rows)
    return result


def _clip_key(row: dict) -> tuple[str, int]:
    index = row.get("chunk_index")
    return str(row.get("video_id")), int(index) if index is not None else -1


def _drop_thin(linked: dict[str, str], seconds: dict[str, float], min_seconds: float) -> dict[str, str]:
    """Unassign identities pooling too little audio, and renumber the rest densely.

    The floor is on the pooled total: a genuine voice accumulates across the
    corpus, while debris stays debris no matter how many videos it shows up in.
    """
    pooled: dict[str, float] = defaultdict(float)
    for key, label in linked.items():
        pooled[label] += seconds.get(key, 0.0)
    keep = sorted((label for label in pooled if pooled[label] >= min_seconds), key=lambda label: -pooled[label])
    renumbered = {label: f"{_PREFIX}{i:02d}" for i, label in enumerate(keep)}
    dropped = len(pooled) - len(keep)
    if dropped:
        log.info("%d identit(ies) under %.0fs treated as debris", dropped, min_seconds)
    return {key: renumbered[label] for key, label in linked.items() if label in renumbered}


def _voice_centroids(
    rows: list[dict],
    linked: dict[str, str],
    clip_embeddings: dict[tuple[str, int], np.ndarray],
    centroids: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """A robust centroid per voice, from its clips (or its per-video centroids)."""
    members: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        voice = linked.get(str(row.get("speaker")))
        vector = clip_embeddings.get(_clip_key(row))
        if voice is not None and vector is not None:
            members[voice].append(vector)
    for key, voice in linked.items():
        if voice not in members:
            members[voice].append(centroids[key])
    return {voice: robust_centroid(np.stack(vectors)) for voice, vectors in members.items()}


def _summarize(rows: list[dict], linked: dict[str, str]) -> dict[str, dict]:
    acc: dict[str, dict] = defaultdict(
        lambda: {"seconds": 0.0, "clips": 0, "episodes": set(), "mos": [], "members": set(), "samples": []}
    )
    for row in rows:
        folder = row.get("global_speaker") or UNASSIGNED
        entry = acc[folder]
        entry["seconds"] += float(row.get("duration") or 0.0)
        entry["clips"] += 1
        entry["episodes"].add(row.get("video_id"))
        entry["members"].add(row.get("speaker"))
        if row.get("squim_mos") is not None:
            entry["mos"].append(float(row["squim_mos"]))
        if len(entry["samples"]) < 5:
            entry["samples"].append(Path(row["audio_file"]).name)

    return {
        folder: {
            "hours": round(e["seconds"] / 3600, 3),
            "seconds": round(e["seconds"], 1),
            "clips": e["clips"],
            "episodes": len(e["episodes"]),
            "mean_mos": round(float(np.mean(e["mos"])), 3) if e["mos"] else None,
            "per_video_speakers": sorted(str(m) for m in e["members"]),
            # A few clips to listen to before trusting the ranking.
            "samples": e["samples"],
        }
        for folder, e in sorted(acc.items(), key=lambda kv: (kv[0] == UNASSIGNED, -kv[1]["seconds"]))
    }


def _write_folders(ws: Workspace, root: Path, rows: list[dict], result: SplitResult, only: set[str] | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Folders from an earlier split hold only hardlinks this module made; clear
    # them so a re-run at a new threshold cannot leave clips in two voices.
    for stale in root.iterdir():
        if stale.is_dir() and (stale.name.startswith(_PREFIX) or stale.name == UNASSIGNED):
            shutil.rmtree(stale)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        folder = result.assignment[row["audio_file"]]
        if only is None or folder in only:
            grouped[folder].append(row)

    for folder, members in grouped.items():
        directory = root / folder
        (directory / "wavs").mkdir(parents=True, exist_ok=True)
        with (directory / "metadata.jsonl").open("w", encoding="utf-8") as fh:
            for row in members:
                name = Path(row["audio_file"]).name
                _link(ws.wavs / row["audio_file"], directory / "wavs" / name)
                entry = dict(row, audio_file=name, source_audio_file=row["audio_file"])
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        write_json(directory / "speaker.json", result.summary.get(folder, {}))
        log.info("%s: %d clip(s)", folder, len(members))

    write_json(root / "report.json", result.summary)


def _link(source: Path, target: Path) -> None:
    if not source.exists():
        log.warning("missing clip, skipped: %s", source)
        return
    try:
        os.link(source, target)
    except OSError:  # another filesystem, or the link limit
        shutil.copy2(source, target)


def _rewrite_metadata(ws: Workspace, rows: list[dict]) -> None:
    """Record the linked identity in the dataset's own metadata too."""
    by_file = {r["audio_file"]: r for r in rows}
    updated = []
    for row in read_jsonl(ws.metadata):
        match = by_file.get(row.get("audio_file"))
        if match is not None:
            row["global_speaker"] = match.get("global_speaker")
            row["global_speaker_distance"] = match.get("global_speaker_distance")
        updated.append(row)
    tmp = ws.metadata.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in updated:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(ws.metadata)


def print_report(summary: dict[str, dict], limit: int = 30) -> None:
    print(f"\n{'voice':<20} {'hours':>7} {'clips':>7} {'eps':>5} {'MOS':>6}  per-video speakers")
    print("-" * 70)
    for i, (folder, s) in enumerate(summary.items()):
        if i >= limit:
            print(f"... {len(summary) - limit} more")
            break
        mos = f"{s['mean_mos']:.2f}" if s["mean_mos"] is not None else "  -  "
        print(f"{folder:<20} {s['hours']:>7.2f} {s['clips']:>7} {s['episodes']:>5} {mos:>6}  {len(s['per_video_speakers'])}")
