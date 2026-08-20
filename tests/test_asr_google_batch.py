"""Tests for the whole-episode BatchRecognize backend.

The cloud calls are not exercised here -- what matters and what breaks is the
arithmetic either side of them: cutting a flat word stream into the chunks the
diarizer chose, and applying the text gates to rows rather than chunk objects.
"""

from __future__ import annotations

import json

import pytest

from yt2ds.config import Config
from yt2ds.stages import filters
from yt2ds.stages.asr_google_batch import BatchWord, assign_to_rows, load_words, save_words


def _row(index: int, start: float, end: float, **extra) -> dict:
    row = {
        "audio_file": f"SPK0/vid_{index:04d}.wav",
        "video_id": "vid",
        "chunk_index": index,
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "text": None,
        "text_cohere": None,
        "text_source": "pending",
        "speaker": "vid_SPK0",
    }
    row.update(extra)
    return row


class TestAssignToRows:
    def test_words_land_in_the_chunk_holding_most_of_their_span(self):
        rows = [_row(0, 0.0, 5.0), _row(1, 5.0, 10.0)]
        words = [
            BatchWord("alpha", 0.5, 1.0, 0.9),
            BatchWord("beta", 4.0, 4.8, 0.9),
            BatchWord("gamma", 6.0, 6.5, 0.8),
        ]
        assert assign_to_rows(rows, words) == 2
        assert rows[0]["text_cohere"] == "alpha beta"
        assert rows[1]["text_cohere"] == "gamma"

    def test_a_word_straddling_a_boundary_goes_to_its_majority_owner(self):
        # 4.6-5.4: 0.4 s in each; 4.6-5.2 puts 0.4 of 0.6 in the first chunk.
        rows = [_row(0, 0.0, 5.0), _row(1, 5.0, 10.0)]
        assign_to_rows(rows, [BatchWord("edge", 4.6, 5.2, 0.9)])
        assert rows[0]["text_cohere"] == "edge"
        assert rows[1]["text_cohere"] == ""

    def test_confidence_is_averaged_over_the_words_a_chunk_owns(self):
        rows = [_row(0, 0.0, 5.0)]
        assign_to_rows(rows, [BatchWord("a", 1.0, 2.0, 0.8), BatchWord("b", 2.0, 3.0, 0.6)])
        assert rows[0]["asr_confidence"] == pytest.approx(0.7)
        assert rows[0]["aligned_words"] == 2

    def test_a_chunk_no_word_covers_is_left_empty_not_dropped(self):
        rows = [_row(0, 0.0, 5.0), _row(1, 50.0, 55.0)]
        assert assign_to_rows(rows, [BatchWord("only", 1.0, 2.0, 0.9)]) == 1
        assert rows[1]["text_cohere"] == ""
        assert rows[1]["asr_confidence"] is None

    def test_rows_out_of_order_are_still_matched_by_time(self):
        rows = [_row(1, 5.0, 10.0), _row(0, 0.0, 5.0)]
        assign_to_rows(rows, [BatchWord("first", 1.0, 2.0, 0.9)])
        assert rows[1]["text_cohere"] == "first"
        assert rows[0]["text_cohere"] == ""


class TestWordCache:
    def test_words_survive_a_save_load_round_trip(self, tmp_path):
        words = [BatchWord("مرحبا", 1.0, 1.5, 0.93), BatchWord("بكم", 1.5, 2.0, 0.81)]
        path = tmp_path / "vid.json"
        save_words(path, words)
        assert json.loads(path.read_text(encoding="utf-8"))[0]["t"] == "مرحبا"
        assert load_words(path) == words

    def test_a_missing_or_truncated_cache_reads_as_empty(self, tmp_path):
        assert load_words(tmp_path / "absent.json") == []
        broken = tmp_path / "broken.json"
        broken.write_text('[{"t": "x"', encoding="utf-8")
        assert load_words(broken) == []


class TestGateRows:
    def _cfg(self) -> Config:
        return Config.load()

    def test_a_good_row_is_kept_and_labelled_google_batch(self):
        rows = [_row(0, 0.0, 4.0, text_cohere="مرحبا بكم في هذا الحوار الطويل", text_source="google_batch")]
        kept, rejected = filters.gate_rows(rows, self._cfg())
        assert len(kept) == 1 and not rejected
        assert kept[0]["text_source"] == "google_batch"
        assert kept[0]["text"] == "مرحبا بكم في هذا الحوار الطويل"

    def test_an_empty_transcript_is_rejected(self):
        # Still pending: the episode never came back, so hold the row.
        kept, rejected = filters.gate_rows([_row(0, 0.0, 4.0, text_cohere="")], self._cfg())
        assert len(kept) == 1 and not rejected

        # Transcribed but no word covered this chunk -- that is a real empty.
        rows = [_row(0, 0.0, 4.0, text_cohere="", text_source="google_batch")]
        kept, rejected = filters.gate_rows(rows, self._cfg())
        assert not kept and rejected[0]["reject_reason"] == "text:empty"

    def test_a_silent_chunk_in_a_transcribed_episode_is_not_left_pending(self):
        # assign_to_rows must clear "pending" even for a chunk it found no
        # words for, or the row waits forever instead of being dropped.
        # 3 s of audio holding 13 characters clears the 4 chars/sec floor.
        rows = [_row(0, 0.0, 3.0), _row(1, 50.0, 55.0)]
        assign_to_rows(
            rows,
            [BatchWord(w, 1.0 + i * 0.4, 1.4 + i * 0.4, 0.9) for i, w in enumerate("مرحبا بكم في هذا".split())],
        )
        assert rows[1]["text_source"] == "google_batch"
        kept, rejected = filters.gate_rows(rows, self._cfg())
        # Only the silent chunk is dropped; the one with words survives.
        assert [r["chunk_index"] for r in rejected] == [1]
        assert [r["chunk_index"] for r in kept] == [0]

    def test_phase_one_audio_rejections_are_not_resurrected_as_text_rows(self):
        # A music/quality rejection carries its reason and never reaches the
        # text gates, because it is not in metadata.jsonl at all.
        rows = [_row(0, 0.0, 4.0, text_cohere="نعم بالتأكيد يا صديقي", text_source="google_batch")]
        kept, rejected = filters.gate_rows(rows, self._cfg())
        assert len(kept) == 1
        assert "reject_reason" not in kept[0]

    def test_too_few_words_is_rejected(self):
        rows = [_row(0, 0.0, 4.0, text_cohere="مرحبا", text_source="google_batch")]
        kept, rejected = filters.gate_rows(rows, self._cfg())
        assert not kept and "too_short" in rejected[0]["reject_reason"]

    def test_low_confidence_is_rejected_on_googles_zero_to_one_scale(self):
        cfg = self._cfg()
        rows = [
            _row(0, 0.0, 4.0, text_cohere="مرحبا بكم في هذا الحوار", asr_confidence=0.2, text_source="google_batch"),
        ]
        kept, rejected = filters.gate_rows(rows, cfg)
        assert not kept
        assert "asr_confidence" in rejected[0]["reject_reason"]

    def test_text_far_too_long_for_its_audio_is_rejected(self):
        rows = [_row(0, 0.0, 2.0, text_cohere="كلمة " * 40, text_source="google_batch")]
        kept, rejected = filters.gate_rows(rows, self._cfg())
        assert not kept and "cps_high" in rejected[0]["reject_reason"]

    def test_a_pending_row_whose_episode_has_no_transcript_is_kept(self):
        rows = [_row(0, 0.0, 4.0)]
        kept, rejected = filters.gate_rows(rows, self._cfg())
        assert len(kept) == 1 and not rejected
        assert kept[0]["text_source"] == "pending"


class TestConfig:
    def test_batch_defaults_match_the_verified_working_combination(self):
        batch = Config.load().asr.google_batch
        # ar-SA is only served by "long" in global/us; us-central1 rejects it.
        assert batch.location == "global"
        assert batch.model == "long"
        assert batch.language_codes == ["ar-SA"]
        assert batch.dynamic_batching is True

    def test_backend_can_be_selected(self):
        cfg = Config.load(None, {"asr.backend": "google_batch"})
        assert cfg.asr.backend == "google_batch"
