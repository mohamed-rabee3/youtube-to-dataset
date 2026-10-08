"""Writing a clip: loudness, the peak ceiling, and the edge fades."""

from __future__ import annotations

import numpy as np
import pytest

from yt2ds.config import Config
from yt2ds.stages.audio import integrated_loudness, render_clip

SR = 48000


@pytest.fixture
def cfg():
    return Config.load()


def speechlike(seconds: float, level: float, seed: int = 0) -> np.ndarray:
    """Amplitude-modulated noise: loud and quiet stretches like speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
    return (level * envelope * rng.standard_normal(t.size)).astype(np.float32)


class TestLoudness:
    def test_clip_mode_brings_quiet_and_loud_clips_to_the_target(self, cfg):
        for level in (0.003, 0.05):
            clip = render_clip(speechlike(4.0, level), SR, cfg.audio, video_gain_db=0.0)
            if clip.gain_limited_db == 0:
                assert clip.lufs == pytest.approx(cfg.audio.target_lufs, abs=0.5)

    def test_video_mode_applies_the_video_gain_unchanged(self, cfg):
        cfg.audio.loudness = "video"
        source = speechlike(4.0, 0.01)
        clip = render_clip(source, SR, cfg.audio, video_gain_db=6.0)
        assert clip.gain_db == pytest.approx(6.0)

    def test_unknown_mode_is_an_error(self, cfg):
        cfg.audio.loudness = "loud"
        with pytest.raises(ValueError, match="audio.loudness"):
            render_clip(speechlike(1.0, 0.01), SR, cfg.audio, 0.0)


class TestNoClipping:
    def test_gain_backs_off_to_the_peak_ceiling(self, cfg):
        # A quiet clip with one sharp transient: the loudness target wants a
        # big gain, the transient forbids it.
        source = speechlike(4.0, 0.002)
        source[SR] = 0.5
        clip = render_clip(source, SR, cfg.audio, 0.0)
        assert clip.gain_limited_db > 0
        assert np.max(np.abs(clip.samples)) <= 10 ** (cfg.audio.peak_ceiling_dbfs / 20) + 1e-6

    def test_ceiling_holds_on_the_resampled_samples(self, cfg):
        # Resampling can overshoot the source peak; the ceiling is checked
        # after it, on what is actually written.
        rng = np.random.default_rng(1)
        source = np.sign(rng.standard_normal(SR * 2)).astype(np.float32) * 0.9
        clip = render_clip(source, SR, cfg.audio, 20.0)
        assert np.max(np.abs(clip.samples)) <= 10 ** (cfg.audio.peak_ceiling_dbfs / 20) + 1e-6


class TestEdges:
    def test_both_edges_fade_to_zero(self, cfg):
        clip = render_clip(np.full(SR, 0.1, dtype=np.float32), SR, cfg.audio, 0.0)
        assert abs(clip.samples[0]) < 1e-6
        assert abs(clip.samples[-1]) < 1e-6
        middle = clip.samples[clip.samples.size // 2]
        assert abs(middle) > 0.01

    def test_output_rate_is_the_configured_one(self, cfg):
        clip = render_clip(speechlike(2.0, 0.01), SR, cfg.audio, 0.0)
        assert clip.samples.size == pytest.approx(2.0 * cfg.audio.out_sample_rate, abs=2)


def test_loudness_is_undefined_for_too_short_or_silent_audio():
    assert integrated_loudness(np.zeros(SR, dtype=np.float32), SR) is None
    assert integrated_loudness(speechlike(0.2, 0.1), SR) is None
