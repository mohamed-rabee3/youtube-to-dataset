"""Subtitle parsing, with emphasis on YouTube's rolling auto-captions."""

from __future__ import annotations

import json

import pytest

from yt2ds.stages.subtitles import parse


def write(tmp_path, name, content):
    path = tmp_path / name
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")
    return path


class TestJson3:
    def test_word_timings_detected(self, tmp_path):
        data = {
            "events": [
                {
                    "tStartMs": 0,
                    "dDurationMs": 2000,
                    "segs": [{"utf8": "مرحبا"}, {"utf8": " بكم", "tOffsetMs": 500}],
                }
            ]
        }
        subs = parse(write(tmp_path, "a.json3", data), kind="yt_auto")
        assert subs.has_word_timings
        assert subs.text == "مرحبا بكم"
        assert subs.words[0].start == 0.0
        assert subs.words[1].start == pytest.approx(0.5)

    def test_rolling_repeats_at_same_timestamp_are_removed(self, tmp_path):
        # The same words re-emitted by a later rollup event carry identical
        # absolute times, which is what makes them safely removable.
        data = {
            "events": [
                {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "مرحبا"}, {"utf8": " بكم", "tOffsetMs": 500}]},
                {
                    "tStartMs": 0,
                    "dDurationMs": 4000,
                    "segs": [
                        {"utf8": "مرحبا"},
                        {"utf8": " بكم", "tOffsetMs": 500},
                        {"utf8": " في", "tOffsetMs": 2000},
                    ],
                },
                {"tStartMs": 4000, "dDurationMs": 2000, "segs": [{"utf8": "هذا"}, {"utf8": " الفيديو", "tOffsetMs": 600}]},
            ]
        }
        subs = parse(write(tmp_path, "b.json3", data), kind="yt_auto")
        assert subs.text == "مرحبا بكم في هذا الفيديو"

    def test_genuine_repetition_at_different_times_survives(self, tmp_path):
        data = {
            "events": [
                {
                    "tStartMs": 0,
                    "dDurationMs": 3000,
                    "segs": [
                        {"utf8": "لا"},
                        {"utf8": " لا", "tOffsetMs": 800},
                        {"utf8": " لا", "tOffsetMs": 1600},
                    ],
                }
            ]
        }
        subs = parse(write(tmp_path, "c.json3", data), kind="yt_auto")
        assert subs.text == "لا لا لا"

    def test_whitespace_and_newline_segments_ignored(self, tmp_path):
        data = {
            "events": [
                {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "\n"}]},
                {"tStartMs": 1000, "dDurationMs": 1000, "segs": [{"utf8": "مرحبا"}]},
            ]
        }
        assert parse(write(tmp_path, "d.json3", data)).text == "مرحبا"

    def test_music_annotation_stripped(self, tmp_path):
        data = {"events": [{"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "[موسيقى] مرحبا"}]}]}
        assert parse(write(tmp_path, "e.json3", data)).text == "مرحبا"

    def test_malformed_file_returns_empty_rather_than_raising(self, tmp_path):
        path = tmp_path / "bad.json3"
        path.write_text("{not json", encoding="utf-8")
        assert parse(path).words == []


class TestVtt:
    ROLLING = """WEBVTT

00:00:00.000 --> 00:00:02.000
مرحبا بكم

00:00:02.000 --> 00:00:04.000
مرحبا بكم في

00:00:04.000 --> 00:00:06.000
مرحبا بكم في هذا

00:00:06.000 --> 00:00:08.000
في هذا الفيديو
"""

    def test_rolling_cues_collapse_for_auto_captions(self, tmp_path):
        subs = parse(write(tmp_path, "a.vtt", self.ROLLING), kind="yt_auto")
        assert subs.text == "مرحبا بكم في هذا الفيديو"

    def test_manual_subs_are_not_block_collapsed(self, tmp_path):
        # A human-written track never rolls, so a deliberately repeated line
        # must survive verbatim.
        content = """WEBVTT

00:00:00.000 --> 00:00:02.000
يا ليل يا عين

00:00:02.000 --> 00:00:04.000
يا ليل يا عين
"""
        subs = parse(write(tmp_path, "b.vtt", content), kind="yt_manual")
        assert subs.text == "يا ليل يا عين يا ليل يا عين"

    def test_inline_word_timings_are_used(self, tmp_path):
        content = """WEBVTT

00:00:01.000 --> 00:00:04.000
مرحبا<00:00:02.000><c> بكم</c><00:00:03.000><c> جميعا</c>
"""
        subs = parse(write(tmp_path, "c.vtt", content), kind="yt_auto")
        assert subs.has_word_timings
        assert subs.text == "مرحبا بكم جميعا"
        assert subs.words[1].start == pytest.approx(2.0)
        assert subs.words[2].start == pytest.approx(3.0)

    def test_html_entities_decoded(self, tmp_path):
        content = """WEBVTT

00:00:00.000 --> 00:00:02.000
&quot;مرحبا&quot; &amp; أهلا
"""
        assert "مرحبا" in parse(write(tmp_path, "d.vtt", content)).text

    def test_cue_timings_are_distributed_across_words(self, tmp_path):
        content = """WEBVTT

00:00:00.000 --> 00:00:04.000
واحد اثنان ثلاثة أربعة
"""
        subs = parse(write(tmp_path, "e.vtt", content), kind="yt_manual")
        assert len(subs.words) == 4
        assert subs.words[0].start == pytest.approx(0.0)
        assert subs.words[-1].end == pytest.approx(4.0, abs=0.01)

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "x.srt.gz"
        path.write_text("nonsense", encoding="utf-8")
        assert parse(path).words == []
