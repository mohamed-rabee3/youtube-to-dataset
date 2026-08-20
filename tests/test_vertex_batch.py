"""Vertex batch: clip identity round-trip and prediction parsing.

The network is not exercised here. What is testable offline is the mapping on
either side of it, which is exactly where a silent bug would attach transcripts
to the wrong clips and corrupt the dataset quietly instead of loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from transcribe_vertex_batch import (  # noqa: E402
    _prediction_entries,
    blob_clip,
    clip_blob,
)

PREFIX = "vertex-asr"


class TestClipIdentity:
    @pytest.mark.parametrize(
        "audio_file",
        [
            "001_-_2019-04-09_-_SPK1/001_-_2019-04-09_-_0067.wav",
            "209_-_2026-07-28_-_SPK0/209_-_2026-07-28_-_1117.wav",
            "flat.wav",
        ],
    )
    def test_round_trip(self, audio_file):
        assert blob_clip(PREFIX, clip_blob(PREFIX, audio_file)) == audio_file

    def test_blob_is_flac_under_the_prefix(self):
        blob = clip_blob(PREFIX, "SPK1/ep_0067.wav")
        assert blob == "vertex-asr/clips/SPK1/ep_0067.flac"

    def test_speaker_subdirectory_survives(self):
        # Two clips with the same basename in different speaker folders must
        # not collapse onto one object.
        a = clip_blob(PREFIX, "SPK0/ep_0001.wav")
        b = clip_blob(PREFIX, "SPK1/ep_0001.wav")
        assert a != b


def _record(clips, texts=None, status="", bucket="bkt"):
    parts = [
        {"fileData": {"mimeType": "audio/flac", "fileUri": f"gs://{bucket}/{clip_blob(PREFIX, c)}"}}
        for c in clips
    ]
    parts.append({"text": "prompt"})
    record = {"request": {"contents": [{"role": "user", "parts": parts}]}, "status": status}
    if texts is not None:
        payload = json.dumps([{"i": i, "text": t} for i, t in enumerate(texts, 1)], ensure_ascii=False)
        record["response"] = {"candidates": [{"content": {"parts": [{"text": payload}]}}]}
    return record


CLIPS = ["SPK1/ep_0067.wav", "SPK2/ep_0069.wav"]


class TestPredictionEntries:
    def test_transcripts_land_on_their_own_clips(self):
        # Distinct texts, so a swapped pairing would show up rather than hide
        # behind two similar strings.
        entries = _prediction_entries(_record(CLIPS, ["بِكُم", "سُقْرَاط"]), PREFIX)
        assert [e["audio_file"] for e in entries] == CLIPS
        assert all(e["status"] == "ok" for e in entries)
        assert entries[0]["text"] == "بِكُم"
        assert entries[1]["text"] == "سُقْرَاط"

    def test_final_letter_mark_is_stripped(self):
        # Rule 3 of the prompt is enforced in code, not trusted to the model.
        entries = _prediction_entries(_record(CLIPS[:1], ["كِتَابْ"]), PREFIX)
        assert not entries[0]["text"].endswith("ْ")
        assert entries[0]["tashkeel_ratio"] > 0

    def test_a_shifted_index_set_discards_the_whole_batch(self):
        # A response numbered from 0 would otherwise attach every transcript to
        # the wrong clip -- the one failure that corrupts silently.
        record = _record(CLIPS)
        payload = json.dumps([{"i": 0, "text": "أ"}, {"i": 1, "text": "ب"}], ensure_ascii=False)
        record["response"] = {"candidates": [{"content": {"parts": [{"text": payload}]}}]}
        entries = _prediction_entries(record, PREFIX)
        assert [e["status"] for e in entries] == ["index_mismatch"] * 2

    def test_row_level_error_marks_every_clip_retryable(self):
        entries = _prediction_entries(_record(CLIPS, status="429 quota"), PREFIX)
        assert len(entries) == 2
        # Anything that is not "ok" is re-queued by the next run.
        assert all(e["status"].startswith("batch:") for e in entries)

    def test_missing_response_is_not_silently_ok(self):
        entries = _prediction_entries(_record(CLIPS), PREFIX)
        assert all(e["status"] == "no_response" for e in entries)

    def test_unparseable_body(self):
        record = _record(CLIPS)
        record["response"] = {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}
        assert all(e["status"] == "unparseable" for e in _prediction_entries(record, PREFIX))

    def test_empty_transcript_is_ok_not_a_failure(self):
        # An empty transcript is a real answer: filters rejects the clip as
        # text:asr_empty rather than re-queueing it forever.
        entries = _prediction_entries(_record(CLIPS[:1], [""]), PREFIX)
        assert entries[0] == {"audio_file": CLIPS[0], "text": "", "status": "ok"}

    def test_prompt_only_part_is_not_mistaken_for_a_clip(self):
        entries = _prediction_entries(_record(CLIPS, ["أ", "ب"]), PREFIX)
        assert len(entries) == 2  # not 3, despite the trailing text part
