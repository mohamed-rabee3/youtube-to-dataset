"""Word-to-chunk assignment and long-audio window planning.

These cover the pure logic around forced alignment; the CTC pass itself needs
the model and is exercised by the end-to-end run.
"""

from __future__ import annotations

import pytest

from yt2ds.config import Config
from yt2ds.stages.align import AlignedWord, _plan_windows, _token_spans, assign_to_chunks
from yt2ds.stages.segment import Chunk
from yt2ds.stages.subtitles import SubWord, Subtitles


@pytest.fixture
def cfg():
    return Config.load()


def word(text, start, end, score=-2.0):
    return AlignedWord(text=text, start=start, end=end, score=score)


def chunk(index, start, end, speaker="A"):
    return Chunk(index=index, start=start, end=end, speaker=speaker)


class TestAssignToChunks:
    def test_words_land_in_the_chunk_that_holds_most_of_them(self, cfg):
        chunks = [chunk(0, 0.0, 5.0), chunk(1, 5.0, 10.0)]
        words = [word("مرحبا", 1.0, 1.5), word("بكم", 2.0, 2.5), word("جميعا", 6.0, 6.5)]
        assign_to_chunks(chunks, words, cfg)
        assert chunks[0].text_yt == "مرحبا بكم"
        assert chunks[1].text_yt == "جميعا"

    def test_align_score_is_averaged_over_the_chunk(self, cfg):
        chunks = [chunk(0, 0.0, 5.0)]
        assign_to_chunks(chunks, [word("a", 1.0, 1.5, -2.0), word("b", 2.0, 2.5, -4.0)], cfg)
        assert chunks[0].scores["align_score"] == pytest.approx(-3.0)
        assert chunks[0].scores["aligned_words"] == 2

    def test_chunk_with_no_words(self, cfg):
        chunks = [chunk(0, 0.0, 5.0)]
        assign_to_chunks(chunks, [word("x", 20.0, 20.5)], cfg)
        assert chunks[0].text_yt == ""
        assert chunks[0].scores["align_score"] is None
        assert chunks[0].scores["aligned_words"] == 0

    def test_word_extending_past_its_own_chunk_flags_clipping(self, cfg):
        # Mostly inside chunk 0, but its tail is outside: the clip's audio does
        # not contain the whole word its transcript names.
        chunks = [chunk(0, 0.0, 5.0)]
        assign_to_chunks(chunks, [word("مقطوعة", 4.6, 5.4)], cfg)
        assert chunks[0].scores["boundary_clipped"] is True

    def test_padding_overlap_between_neighbours_does_not_flag_clipping(self, cfg):
        """Regression: chunks are padded, so neighbours deliberately overlap.

        A word sitting wholly inside chunk 1 also falls within chunk 0's
        padded tail. That must not mark chunk 0 as clipped -- treating it as
        such rejected roughly half of all otherwise-good chunks.
        """
        chunks = [chunk(0, 0.0, 5.2), chunk(1, 4.8, 10.0)]
        words = [word("داخل", 1.0, 1.5), word("تمام", 5.0, 5.15)]
        assign_to_chunks(chunks, words, cfg)
        assert chunks[0].scores["boundary_clipped"] is False
        assert chunks[1].scores["boundary_clipped"] is False

    def test_fully_contained_words_never_flag_clipping(self, cfg):
        chunks = [chunk(0, 0.0, 5.0)]
        assign_to_chunks(chunks, [word("a", 1.0, 1.4), word("b", 2.0, 2.4)], cfg)
        assert chunks[0].scores["boundary_clipped"] is False

    def test_no_chunks_is_a_noop(self, cfg):
        assign_to_chunks([], [word("a", 0.0, 1.0)], cfg)


class TestPlanWindows:
    def _subs(self, count, spacing=10.0):
        return Subtitles(
            words=[SubWord(text=f"w{i}", start=i * spacing, end=i * spacing + 1) for i in range(count)],
            kind="yt_auto",
        )

    def test_long_audio_is_split_into_windows(self):
        subs = self._subs(60, spacing=10.0)  # 600 s of subtitle words
        kept = list(range(60))
        spans = _plan_windows(subs, kept, total_seconds=600.0, window=200.0)
        assert len(spans) == 3
        # Windows must tile the audio without gaps.
        assert spans[0][0] == 0.0
        for previous, following in zip(spans, spans[1:]):
            assert previous[1] == following[0]

    def test_every_word_is_assigned_to_exactly_one_window(self):
        subs = self._subs(60, spacing=10.0)
        kept = list(range(60))
        spans = _plan_windows(subs, kept, total_seconds=600.0, window=200.0)
        covered = []
        for _, _, first, last in spans:
            covered.extend(range(first, last))
        assert covered == sorted(set(covered))
        assert len(covered) == 60

    def test_windows_with_no_words_are_skipped(self):
        # All the speech is in the first minute of a ten-minute file.
        subs = self._subs(6, spacing=10.0)
        spans = _plan_windows(subs, list(range(6)), total_seconds=600.0, window=200.0)
        assert len(spans) == 1

    def test_falls_back_to_one_span_when_nothing_matches(self):
        subs = Subtitles(words=[], kind="yt_auto")
        spans = _plan_windows(subs, [], total_seconds=600.0, window=200.0)
        assert spans == [(0.0, 600.0, 0, 0)]


class TestTokenSpans:
    def test_collapses_repeats_and_skips_blanks(self):
        import numpy as np

        path = np.array([0, 5, 5, 0, 7, 0])
        scores = np.array([0.0, -1.0, -3.0, 0.0, -2.0, 0.0])
        spans = _token_spans(path, scores, blank=0)
        assert len(spans) == 2
        assert spans[0][:2] == (1, 3)
        assert spans[0][2] == pytest.approx(-2.0)
        assert spans[1][:2] == (4, 5)

    def test_all_blank_yields_nothing(self):
        import numpy as np

        assert _token_spans(np.array([0, 0, 0]), np.array([0.0, 0.0, 0.0]), blank=0) == []
