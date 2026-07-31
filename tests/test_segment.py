"""Chunk boundary rules: speaker splitting, overlap removal, merging, splitting."""

from __future__ import annotations

import pytest

from yt2ds.config import Config
from yt2ds.stages.diarize import Diarization, Turn
from yt2ds.stages.segment import build
from yt2ds.stages.vad import Region, subtract


@pytest.fixture
def cfg():
    return Config.load()


def diar(turns, overlaps=()):
    return Diarization(
        turns=[Turn(*t) for t in turns],
        overlaps=[Region(*o) for o in overlaps],
        speakers=sorted({t[2] for t in turns}),
    )


class TestSubtract:
    def test_removes_a_middle_hole(self):
        out = subtract([Region(0, 10)], [Region(4, 6)])
        assert [(r.start, r.end) for r in out] == [(0, 4), (6, 10)]

    def test_hole_covering_everything_leaves_nothing(self):
        assert subtract([Region(2, 5)], [Region(0, 10)]) == []

    def test_no_holes_is_identity(self):
        out = subtract([Region(0, 3)], [])
        assert [(r.start, r.end) for r in out] == [(0, 3)]

    def test_multiple_overlapping_holes(self):
        out = subtract([Region(0, 10)], [Region(2, 4), Region(3, 6)])
        assert [(r.start, r.end) for r in out] == [(0, 2), (6, 10)]


class TestSpeakerBoundaries:
    def test_chunk_never_spans_two_speakers(self, cfg):
        regions = [Region(0, 10)]
        annotation = diar([(0, 5, "A"), (5, 10, "B")])
        chunks = build(regions, annotation, cfg, duration=10)
        assert chunks
        for chunk in chunks:
            assert chunk.speaker in ("A", "B")
        # The A chunk must end by the turn boundary (plus padding).
        a = [c for c in chunks if c.speaker == "A"]
        assert all(c.end <= 5 + cfg.segment.pad + 1e-6 for c in a)

    def test_overlapped_speech_is_discarded(self, cfg):
        regions = [Region(0, 12)]
        annotation = diar([(0, 8, "A"), (6, 12, "B")], overlaps=[(6, 8)])
        chunks = build(regions, annotation, cfg, duration=12)
        for chunk in chunks:
            # No kept chunk may sit inside the overlap window.
            assert not (chunk.start >= 6 and chunk.end <= 8)

    def test_keeping_overlap_when_configured(self, cfg):
        cfg.diarize.drop_overlap = False
        regions = [Region(0, 12)]
        annotation = diar([(0, 8, "A"), (6, 12, "B")], overlaps=[(6, 8)])
        assert build(regions, annotation, cfg, duration=12)

    def test_unattributed_speech_is_dropped(self, cfg):
        # Speech in a stretch the diarizer assigned to nobody must not be
        # guessed onto a speaker.
        regions = [Region(0, 10)]
        annotation = diar([(0, 3, "A")])
        chunks = build(regions, annotation, cfg, duration=10)
        assert all(c.end <= 3 + cfg.segment.pad + 1e-6 for c in chunks)


class TestMergingAndSplitting:
    def test_adjacent_same_speaker_regions_merge(self, cfg):
        regions = [Region(0, 3), Region(3.2, 6)]
        annotation = diar([(0, 6, "A")])
        chunks = build(regions, annotation, cfg, duration=6)
        assert len(chunks) == 1
        assert chunks[0].duration > 5

    def test_large_gap_prevents_merging(self, cfg):
        regions = [Region(0, 3), Region(8, 11)]
        annotation = diar([(0, 12, "A")])
        chunks = build(regions, annotation, cfg, duration=12)
        assert len(chunks) == 2

    def test_long_span_is_split_at_the_deepest_pause(self, cfg):
        # 20s of speech with a 1s pause at 11s and a 0.3s pause at 5s: the
        # split must land on the longer pause.
        regions = [Region(0, 5), Region(5.3, 11), Region(12, 20)]
        annotation = diar([(0, 20, "A")])
        chunks = build(regions, annotation, cfg, duration=20)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.duration <= cfg.segment.max_duration + 2 * cfg.segment.pad + 1e-6
        boundaries = [round(c.end, 1) for c in chunks[:-1]]
        assert any(abs(b - 11.2) < 1.0 for b in boundaries), boundaries

    def test_split_falls_back_to_a_hard_cut_without_pauses(self, cfg):
        regions = [Region(0, 30)]
        annotation = diar([(0, 30, "A")])
        chunks = build(regions, annotation, cfg, duration=30)
        assert chunks
        for chunk in chunks:
            assert chunk.duration <= cfg.segment.max_duration + 2 * cfg.segment.pad + 1e-6


class TestDurationBounds:
    def test_too_short_is_dropped(self, cfg):
        regions = [Region(0, 0.5)]
        annotation = diar([(0, 1, "A")])
        assert build(regions, annotation, cfg, duration=1) == []

    def test_all_chunks_respect_configured_bounds(self, cfg):
        regions = [Region(0, 4), Region(5, 9), Region(10, 30)]
        annotation = diar([(0, 30, "A")])
        chunks = build(regions, annotation, cfg, duration=30)
        assert chunks
        for chunk in chunks:
            assert chunk.duration >= cfg.segment.min_duration
            assert chunk.duration <= cfg.segment.max_duration + 2 * cfg.segment.pad + 1e-6

    def test_padding_is_clamped_to_the_file(self, cfg):
        regions = [Region(0, 5)]
        annotation = diar([(0, 5, "A")])
        chunks = build(regions, annotation, cfg, duration=5)
        assert chunks[0].start >= 0.0
        assert chunks[0].end <= 5.0

    def test_indices_are_sequential(self, cfg):
        regions = [Region(0, 4), Region(5, 9), Region(10, 14)]
        annotation = diar([(0, 14, "A")])
        chunks = build(regions, annotation, cfg, duration=14)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_no_regions_yields_no_chunks(self, cfg):
        assert build([], diar([(0, 10, "A")]), cfg, duration=10) == []
