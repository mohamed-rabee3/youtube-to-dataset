"""`yt2ds speakers`: linking voices across videos and filing every clip."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from yt2ds.config import Config
from yt2ds.io import Workspace, read_jsonl
from yt2ds.speaker_split import UNASSIGNED, split


def unit(vector):
    v = np.asarray(vector, dtype=np.float32)
    return v / np.linalg.norm(v)


HOST = unit([1, 0, 0, 0])
GUEST = unit([0, 1, 0, 0])
STRANGER = unit([0, 0, 1, 0])


@pytest.fixture
def dataset(tmp_path):
    """Two episodes with a host each, one guest, and one mislabelled clip."""
    ws = Workspace(tmp_path / "ds").create()
    rows = []
    clips: dict[str, list[tuple[int, np.ndarray]]] = {"ep1": [], "ep2": []}

    def add(video, index, speaker, vector, seconds=20.0):
        name = f"{video}_{speaker}/{video}_{index:04d}.wav"
        (ws.wavs / name).parent.mkdir(parents=True, exist_ok=True)
        (ws.wavs / name).write_bytes(b"RIFF")
        rows.append(
            {"audio_file": name, "video_id": video, "chunk_index": index, "speaker": f"{video}_{speaker}",
             "duration": seconds, "squim_mos": 4.0}
        )
        clips[video].append((index, vector))

    for i in range(4):
        add("ep1", i, "SPK0", HOST)
        add("ep2", i, "SPK1", HOST)
    for i in range(4, 7):
        add("ep2", i, "SPK0", GUEST)
    # The diarizer filed a stranger's clip under ep1's host.
    add("ep1", 9, "SPK0", STRANGER)
    # A few seconds of crosstalk: a real cluster, but debris.
    add("ep1", 10, "SPK2", unit([0, 0, 0, 1]), seconds=3.0)

    with ws.metadata.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    np.savez(ws.embeddings / "ep1.npz", ep1_SPK0=HOST, ep1_SPK2=unit([0, 0, 0, 1]))
    np.savez(ws.embeddings / "ep2.npz", ep2_SPK1=HOST, ep2_SPK0=GUEST)
    for video, items in clips.items():
        np.savez(
            ws.embeddings / f"{video}.clips.npz",
            index=np.array([i for i, _ in items], dtype=np.int32),
            embedding=np.stack([v for _, v in items]),
        )
    return ws


def folder_rows(ws, folder):
    return list(read_jsonl(ws.root / "speakers" / folder / "metadata.jsonl"))


def test_the_host_is_one_voice_across_episodes(dataset):
    split(dataset, Config.load())
    host = folder_rows(dataset, "GLOBAL_SPEAKER_00")
    assert {r["video_id"] for r in host} == {"ep1", "ep2"}
    assert len(host) == 8


def test_a_mislabelled_clip_is_not_filed_under_the_voice(dataset):
    result = split(dataset, Config.load())
    assert result.assignment["ep1_SPK0/ep1_0009.wav"] == UNASSIGNED


def test_debris_identities_go_to_unassigned(dataset):
    result = split(dataset, Config.load())
    assert result.assignment["ep1_SPK2/ep1_0010.wav"] == UNASSIGNED
    assert set(result.summary) == {"GLOBAL_SPEAKER_00", "GLOBAL_SPEAKER_01", UNASSIGNED}


def test_folders_are_hardlinks_and_valid_dataset_roots(dataset):
    split(dataset, Config.load())
    row = folder_rows(dataset, "GLOBAL_SPEAKER_01")[0]
    linked = dataset.root / "speakers" / "GLOBAL_SPEAKER_01" / "wavs" / row["audio_file"]
    assert os.path.samefile(linked, dataset.wavs / row["source_audio_file"])
    assert (dataset.root / "speakers" / "report.json").exists()


def test_the_dataset_metadata_records_the_voice(dataset):
    split(dataset, Config.load())
    voices = {r["audio_file"]: r["global_speaker"] for r in read_jsonl(dataset.metadata)}
    assert voices["ep2_SPK0/ep2_0004.wav"] == "GLOBAL_SPEAKER_01"
    assert voices["ep1_SPK0/ep1_0009.wav"] is None


def test_report_only_writes_nothing(dataset):
    split(dataset, Config.load(), write=False)
    assert not (dataset.root / "speakers").exists()


def test_a_rerun_replaces_the_previous_split(dataset):
    cfg = Config.load()
    split(dataset, cfg)
    cfg.speakers.min_speaker_seconds = 0.0
    split(dataset, cfg)
    folders = sorted(p.name for p in (dataset.root / "speakers").iterdir() if p.is_dir())
    assert folders == ["GLOBAL_SPEAKER_00", "GLOBAL_SPEAKER_01", "GLOBAL_SPEAKER_02", UNASSIGNED]
    linked = [p.name for p in (dataset.root / "speakers").glob("*/wavs/*.wav")]
    assert len(linked) == len(set(linked)), "a clip sits in two voices"
