"""Diarization post-processing."""

from __future__ import annotations

from yt2ds.config import Config
from yt2ds.stages.diarize import Diarization, Turn, collapse_if_dominant


def host_with_short_guest() -> Diarization:
    turns = [Turn(0, 92, "HOST"), Turn(92, 100, "GUEST")]
    return Diarization(turns=turns, speakers=["GUEST", "HOST"])


def test_a_short_guest_keeps_its_own_label_by_default():
    out = collapse_if_dominant(host_with_short_guest(), Config.load())
    assert {t.speaker for t in out.turns} == {"HOST", "GUEST"}


def test_collapse_is_available_when_asked_for():
    cfg = Config.load()
    cfg.diarize.collapse_single_speaker_ratio = 0.9
    out = collapse_if_dominant(host_with_short_guest(), cfg)
    assert {t.speaker for t in out.turns} == {"HOST"}
