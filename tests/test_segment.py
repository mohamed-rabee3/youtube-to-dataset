"""Chunk boundary rules.

The property everything here protects: every clip starts and ends in a real
pause, no two clips share audio, and no clip is cut on a grid.
"""

from __future__ import annotations

import random

import pytest

from yt2ds.config import Config
from yt2ds.stages.diarize import Diarization, Turn
from yt2ds.stages.segment import build
from yt2ds.stages.vad import Region, subtract

EPS = 1e-6


@pytest.fixture
def cfg():
    return Config.load()


def diar(turns, overlaps=()):
    return Diarization(
        turns=[Turn(*t) for t in turns],
        overlaps=[Region(*o) for o in overlaps],
        speakers=sorted({t[2] for t in turns}),
    )


def regions(*spans):
    return [Region(*s) for s in spans]


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
    def test_turn_change_at_a_pause_gives_two_clean_clips(self, cfg):
        chunks = build(regions((0, 4), (4.5, 9)), diar([(0, 4.2, "A"), (4.3, 9, "B")]), cfg, duration=9)
        assert [c.speaker for c in chunks] == ["A", "B"]
        a, b = chunks
        assert a.speech_end == pytest.approx(4.0)
        assert b.speech_start == pytest.approx(4.5)
        assert a.end <= b.start + EPS

    def test_turn_change_mid_phrase_drops_the_partial_phrases(self, cfg):
        # The A->B change at 9 s falls inside a stretch of speech with no
        # pause. Cutting there would clip a word on both sides, so the phrase
        # either side of it is dropped instead; A keeps what precedes it.
        chunks = build(
            regions((0, 3), (3.4, 6), (6.5, 12)),
            diar([(0, 9, "A"), (9, 12, "B")]),
            cfg,
            duration=12,
        )
        assert len(chunks) == 1
        assert chunks[0].speaker == "A"
        assert chunks[0].speech_end == pytest.approx(6.0)
        assert all(not (c.start < 9 < c.end) for c in chunks)

    def test_overlapped_speech_is_never_inside_a_clip(self, cfg):
        chunks = build(
            regions((0, 5), (5.5, 12)),
            diar([(0, 8, "A"), (6, 12, "B")], overlaps=[(6, 8)]),
            cfg,
            duration=12,
        )
        assert chunks
        for chunk in chunks:
            assert chunk.end <= 6 + EPS or chunk.start >= 8 - EPS
        # The phrase the overlap interrupted is dropped, not truncated.
        assert chunks[0].speech_end == pytest.approx(5.0)

    def test_keeping_overlap_when_configured(self, cfg):
        cfg.diarize.drop_overlap = False
        chunks = build(
            regions((0, 5), (5.5, 12)),
            diar([(0, 8, "A"), (6, 12, "B")], overlaps=[(6, 8)]),
            cfg,
            duration=12,
        )
        assert chunks

    def test_unattributed_speech_is_never_guessed_onto_a_speaker(self, cfg):
        # The diarizer gave 3-10 s to nobody. A's piece ends mid-phrase where
        # that speech begins, so nothing here can be cut cleanly.
        assert build(regions((0, 10)), diar([(0, 3, "A")]), cfg, duration=10) == []


class TestMerging:
    def test_short_pause_stays_inside_one_clip(self, cfg):
        chunks = build(regions((0, 3), (3.2, 6)), diar([(0, 6, "A")]), cfg, duration=6)
        assert len(chunks) == 1
        assert chunks[0].speech_start == pytest.approx(0.0)
        assert chunks[0].speech_end == pytest.approx(6.0)
        assert chunks[0].scores["max_internal_pause"] == pytest.approx(0.2)

    def test_pause_longer_than_merge_gap_splits(self, cfg):
        chunks = build(regions((0, 3), (4.0, 7)), diar([(0, 7, "A")]), cfg, duration=7)
        assert len(chunks) == 2

    def test_another_voice_in_the_gap_prevents_merging(self, cfg):
        # VAD missed B's short reply in the gap, the diarizer did not.
        chunks = build(
            regions((0, 3), (3.5, 6.5)),
            diar([(0, 3, "A"), (3.1, 3.4, "B"), (3.5, 6.5, "A")]),
            cfg,
            duration=7,
        )
        assert all(c.speech_end <= 3 + EPS or c.speech_start >= 3.5 - EPS for c in chunks)


class TestSplitting:
    def test_long_speech_is_split_at_its_longest_pause(self, cfg):
        chunks = build(regions((0, 5), (5.3, 11), (12, 20), (20.4, 28)), diar([(0, 28, "A")]), cfg, duration=30)
        assert [(round(c.speech_start, 1), round(c.speech_end, 1)) for c in chunks] == [(0, 11), (12, 28)]
        assert all(c.speech_duration <= cfg.segment.max_duration + EPS for c in chunks)

    def test_speech_with_no_pause_is_dropped_not_chopped_on_a_grid(self, cfg):
        # The old fallback cut this into equal pieces, mid-word.
        assert build(regions((0, 30)), diar([(0, 30, "A")]), cfg, duration=30) == []

    def test_pauses_shorter_than_min_cut_pause_are_not_cut_points(self, cfg):
        assert build(regions((0, 10), (10.1, 25)), diar([(0, 25, "A")]), cfg, duration=25) == []


class TestPadding:
    def test_isolated_speech_gets_the_full_pad(self, cfg):
        (chunk,) = build(regions((5, 9)), diar([(5, 9, "A")]), cfg, duration=20)
        assert chunk.start == pytest.approx(5 - cfg.segment.pad)
        assert chunk.end == pytest.approx(9 + cfg.segment.pad)

    def test_padding_stops_half_way_to_the_next_voice(self, cfg):
        chunks = build(
            regions((0, 3), (3.4, 6)),
            diar([(0, 3, "A"), (3.4, 6, "B")]),
            cfg,
            duration=6,
        )
        a, b = chunks
        assert a.end == pytest.approx(3.2)
        assert b.start == pytest.approx(3.2)

    def test_padding_is_clamped_to_the_file(self, cfg):
        (chunk,) = build(regions((0.05, 5)), diar([(0, 5, "A")]), cfg, duration=5.1)
        assert chunk.start >= 0.0
        assert chunk.end <= 5.1 + EPS


class TestDurationBounds:
    def test_too_short_is_dropped(self, cfg):
        assert build(regions((0, 0.5)), diar([(0, 1, "A")]), cfg, duration=1) == []

    def test_indices_are_sequential(self, cfg):
        chunks = build(regions((0, 4), (5, 9), (10, 14)), diar([(0, 14, "A")]), cfg, duration=14)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_no_regions_yields_no_chunks(self, cfg):
        assert build([], diar([(0, 10, "A")]), cfg, duration=10) == []


class TestInvariants:
    """Random conversations: the guarantees must hold for every layout."""

    @pytest.mark.parametrize("seed", range(200))
    def test_clips_never_overlap_and_every_edge_sits_in_a_pause(self, cfg, seed):
        rng = random.Random(seed)
        speech: list[Region] = []
        turns: list[tuple[float, float, str]] = []
        t = 0.0
        speaker = "A"
        while t < 300:
            length = rng.uniform(0.3, 9.0)
            speech.append(Region(t, t + length))
            # Turn changes sometimes land at a pause, sometimes mid-phrase.
            if rng.random() < 0.15:
                cut = t + length * rng.uniform(0.2, 0.8) if rng.random() < 0.5 else t
                turns.append((cut, cut, speaker))
                speaker = "B" if speaker == "A" else "A"
            t += length + rng.choice([0.05, 0.1, 0.2, 0.3, 0.6, 1.0, 2.0])
        boundaries = sorted({c for c, _, _ in turns} | {0.0, t})
        labels = ["A", "B"] * len(boundaries)
        diarization = diar([(lo, hi, labels[i]) for i, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])) if hi > lo])

        chunks = build(speech, diarization, cfg, duration=t)
        starts = {r.start for r in speech}
        ends = {r.end for r in speech}
        for a, b in zip(chunks, chunks[1:]):
            assert a.end <= b.start + EPS, "clips share audio"
        for chunk in chunks:
            assert chunk.speech_start in starts and chunk.speech_end in ends, "edge inside speech"
            assert chunk.speech_duration <= cfg.segment.max_duration + EPS
            assert chunk.start <= chunk.speech_start and chunk.end >= chunk.speech_end
            before = [r.end for r in speech if r.end <= chunk.speech_start + EPS and r.start < chunk.speech_start]
            after = [r.start for r in speech if r.start >= chunk.speech_end - EPS and r.end > chunk.speech_end]
            if before:
                gap = chunk.speech_start - max(before)
                assert gap >= cfg.segment.min_cut_pause - EPS
                assert chunk.speech_start - chunk.start <= gap / 2 + EPS
            if after:
                gap = min(after) - chunk.speech_end
                assert gap >= cfg.segment.min_cut_pause - EPS
                assert chunk.end - chunk.speech_end <= gap / 2 + EPS


class TestDiarizerVadMismatch:
    """VAD and diarizer boundaries never agree exactly; that is not a cut."""

    def test_vad_hearing_speech_before_the_turn_starts_is_not_a_dirty_edge(self, cfg):
        chunks = build(regions((0.9, 5)), diar([(1.0, 5, "A")]), cfg, duration=10)
        assert len(chunks) == 1
        assert chunks[0].speech_start == pytest.approx(0.9)

    def test_gaps_between_one_speakers_turns_stay_inside_the_clip(self, cfg):
        # One VAD region, but the diarizer split it into three turns of A.
        chunks = build(regions((0, 9)), diar([(0, 3, "A"), (3.2, 6, "A"), (6.3, 9, "A")]), cfg, duration=9)
        assert len(chunks) == 1
        assert chunks[0].speech_duration == pytest.approx(9.0)

    def test_a_sliver_between_two_different_speakers_stays_unattributed(self, cfg):
        chunks = build(regions((0, 9)), diar([(0, 4, "A"), (4.2, 9, "B")]), cfg, duration=9)
        assert chunks == []
