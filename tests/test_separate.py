"""Vocal isolation: what replaces the working audio, and what stops it.

The isolator itself is a 900 MB GPU model, so the model boundary is faked here.
What is under test is the contract around it: which files the rest of the
pipeline is pointed at afterwards, that loudness is re-measured against the
isolated voice rather than the mix, and that a separation which silently
produced the wrong thing is refused instead of shipped.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from yt2ds.config import Config
from yt2ds.io import Workspace
from yt2ds.stages import music, separate
from yt2ds.stages.audio import PreparedAudio
from yt2ds.stages.segment import Chunk

SR_MASTER = 48000
SR_WORK = 16000
DURATION = 4.0


@pytest.fixture
def cfg():
    return Config.load()


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "dataset").create()


def _tone(path, sr, seconds=DURATION, amp=0.3):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    sf.write(str(path), (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="PCM_16")
    return path


@pytest.fixture
def prepared(ws):
    master = _tone(ws.audio / "vid.48k.wav", SR_MASTER)
    work = _tone(ws.audio / "vid.16k.wav", SR_WORK)
    return PreparedAudio(
        video_id="vid",
        master_path=master,
        work_path=work,
        mp3_path=None,
        duration=DURATION,
        master_sr=SR_MASTER,
        work_sr=SR_WORK,
        lufs=-23.0,
        gain_db=0.0,
    )


class _FakeSeparator:
    """Stands in for audio-separator: writes a quieter copy of its input."""

    def __init__(self, ws, seconds=DURATION, amp=0.1, outputs=None):
        self.ws = ws
        self.seconds = seconds
        self.amp = amp
        self.outputs = outputs
        self.model_instance = None
        self.output_dir = None
        self.calls = []

    def separate(self, path, custom_output_names=None):
        self.calls.append(path)
        if self.outputs is not None:
            return self.outputs
        name = f"{custom_output_names['Vocals']}.wav"
        _tone(self.ws.audio / name, separate.MODEL_SR, self.seconds, self.amp)
        return [name]


class _Registry:
    def __init__(self, separator):
        self.separator = separator


class TestApply:
    def test_disabled_passes_the_mix_through_untouched(self, prepared, ws, cfg):
        cfg.separate.enabled = False
        result = separate.apply(prepared, ws, cfg, _Registry(None))
        assert result is prepared
        assert not result.separated

    def test_working_audio_is_repointed_at_the_isolated_voice(self, prepared, ws, cfg):
        registry = _Registry(_FakeSeparator(ws))
        result = separate.apply(prepared, ws, cfg, registry)

        assert result.separated
        assert result.master_path.name == "vid.48k.vocals.wav"
        assert result.work_path.name == "vid.16k.vocals.wav"
        assert result.master_path.exists() and result.work_path.exists()
        # The rates the rest of the pipeline reads at are unchanged; only the
        # content behind them is.
        assert sf.info(str(result.master_path)).samplerate == SR_MASTER
        assert sf.info(str(result.work_path)).samplerate == SR_WORK

    def test_loudness_is_remeasured_on_the_voice(self, prepared, ws, cfg):
        # The stand-in drops the level by 10 dB, as taking out a music bed does.
        registry = _Registry(_FakeSeparator(ws, amp=0.1))
        result = separate.apply(prepared, ws, cfg, registry)

        assert result.lufs < prepared.lufs
        # The emit-time gain has to follow the level of what is actually kept.
        assert result.gain_db == pytest.approx(cfg.audio.target_lufs - result.lufs)

    def test_the_original_decode_is_dropped_once_it_has_no_reader(self, prepared, ws, cfg):
        registry = _Registry(_FakeSeparator(ws))
        result = separate.apply(prepared, ws, cfg, registry)

        assert not prepared.master_path.exists()
        assert result.source_master_path is None

    def test_keep_original_holds_on_to_the_mix(self, prepared, ws, cfg):
        cfg.separate.keep_original = True
        registry = _Registry(_FakeSeparator(ws))
        result = separate.apply(prepared, ws, cfg, registry)

        assert prepared.master_path.exists()
        assert result.source_master_path == prepared.master_path
        assert result.source_work_path == prepared.work_path

    def test_an_existing_isolation_is_reused(self, prepared, ws, cfg):
        separator = _FakeSeparator(ws)
        registry = _Registry(separator)
        separate.apply(prepared, ws, cfg, registry)
        # A resumed run finds both files already there and must not pay for the
        # separation a second time.
        again = separate.apply(prepared, ws, cfg, registry)

        assert len(separator.calls) == 1
        assert again.separated

    def test_a_silent_failure_is_raised_not_shipped(self, prepared, ws, cfg):
        # audio-separator swallows its own exceptions and returns [].
        registry = _Registry(_FakeSeparator(ws, outputs=[]))
        with pytest.raises(separate.SeparationError):
            separate.apply(prepared, ws, cfg, registry)

    def test_audio_that_no_longer_lines_up_is_refused(self, prepared, ws, cfg):
        # Every timestamp downstream was measured against the original decode,
        # so an isolated file of a different length is unusable.
        registry = _Registry(_FakeSeparator(ws, seconds=DURATION + 1.0))
        with pytest.raises(separate.SeparationError):
            separate.apply(prepared, ws, cfg, registry)


class TestMusicGate:
    """Music is now something to remove, not a reason to drop the speech."""

    @pytest.fixture
    def loud_music(self, monkeypatch):
        """Both detectors firing hard: a chunk that used to be dropped on sight."""

        def fake_demucs(items, sr, cfg, registry):
            for chunk, _ in items:
                chunk.scores["accompaniment_ratio"] = 0.9

        def fake_ast(items, sr, cfg, registry):
            for chunk, _ in items:
                chunk.scores["music_score"] = 0.9
                chunk.scores["music_label"] = "Music"

        monkeypatch.setattr(music, "_score_demucs", fake_demucs)
        monkeypatch.setattr(music, "_score_ast", fake_ast)
        return [(Chunk(index=0, start=0.0, end=3.0, speaker="SPK0"), np.zeros(0, dtype=np.float32))]

    def test_residual_music_is_scored_but_kept(self, cfg, loud_music):
        cfg.music.reject = False
        music.score_batch(loud_music, SR_MASTER, cfg, registry=None)

        chunk = loud_music[0][0]
        assert chunk.reject_reason is None
        # Still measured, so the residue is visible in the metadata.
        assert chunk.scores["accompaniment_ratio"] == 0.9
        assert chunk.scores["music_score"] == 0.9

    def test_reject_restores_the_old_gate(self, cfg, loud_music):
        cfg.music.reject = True
        music.score_batch(loud_music, SR_MASTER, cfg, registry=None)

        assert loud_music[0][0].reject_reason.startswith("music:accompaniment")
