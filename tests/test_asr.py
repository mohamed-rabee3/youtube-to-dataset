"""Backend dispatch and the Google Speech-to-Text backend's own logic.

The network call itself is not exercised here; what is testable without
credentials is the mapping either side of it -- languages in, transcript and
confidence out -- which is exactly where a silent mistake would corrupt a
dataset rather than crash the run.
"""

from __future__ import annotations

import numpy as np
import pytest

from yt2ds.config import Config
from yt2ds.stages import asr, asr_google


@pytest.fixture
def cfg():
    return Config.load()


class TestDispatch:
    def test_google_is_the_default_backend(self, cfg):
        assert cfg.asr.backend == "google"

    def test_unknown_backend_is_rejected(self, cfg):
        cfg.asr.backend = "whisper"
        with pytest.raises(ValueError, match="unknown asr.backend"):
            asr.transcribe_batch([(object(), np.zeros(16000, dtype=np.float32))], 16000, cfg, None)

    def test_empty_batch_needs_no_backend_at_all(self, cfg):
        # Called once per batch of chunks, and a batch can come back empty
        # after music rejection. That must not build a client or load a model.
        assert asr.transcribe_batch([], 16000, cfg, None) == []


class TestLocales:
    def test_short_codes_become_bcp47(self, cfg):
        assert asr_google._locales(cfg) == ["ar-SA", "en-US"]

    def test_a_locale_passes_through_unchanged(self, cfg):
        cfg.asr.languages = ["ar-EG", "en"]
        assert asr_google._locales(cfg) == ["ar-EG", "en-US"]

    def test_duplicates_collapse_but_order_holds(self, cfg):
        # ar and ar-SA map to the same locale; sending it twice would waste an
        # alternative slot (the API allows only three).
        cfg.asr.languages = ["en", "ar", "ar-SA"]
        assert asr_google._locales(cfg) == ["en-US", "ar-SA"]

    def test_empty_languages_falls_back_to_arabic(self, cfg):
        cfg.asr.languages = []
        assert asr_google._locales(cfg) == ["ar-SA"]

    @pytest.mark.parametrize(
        ("locale", "expected"),
        [("ar-SA", "ar"), ("en-US", "en"), ("AR-sa", "ar"), ("ar", "ar")],
    )
    def test_script_gate_sees_a_short_code(self, locale, expected):
        # filters.script_ratio keys off "ar"/"en"; a raw BCP-47 code would fall
        # through to the permissive unknown-language branch.
        assert asr_google._short(locale) == expected


class TestPcm16:
    def test_full_scale_maps_to_int16_range(self):
        pcm = np.frombuffer(asr_google._pcm16(np.array([1.0, -1.0, 0.0], dtype=np.float32)), dtype="<i2")
        assert list(pcm) == [32767, -32767, 0]

    def test_out_of_range_samples_are_clipped_not_wrapped(self):
        # Without the clip these overflow and a loud clip comes back as noise.
        pcm = np.frombuffer(asr_google._pcm16(np.array([1.8, -1.8], dtype=np.float32)), dtype="<i2")
        assert list(pcm) == [32767, -32767]

    def test_two_bytes_per_sample(self):
        assert len(asr_google._pcm16(np.zeros(160, dtype=np.float32))) == 320

    def test_empty_audio(self):
        assert asr_google._pcm16(np.array([], dtype=np.float32)) == b""


class _Alternative:
    def __init__(self, transcript, confidence=0.0):
        self.transcript = transcript
        self.confidence = confidence


class _Result:
    def __init__(self, alternatives, language_code=""):
        self.alternatives = alternatives
        self.language_code = language_code


class _Response:
    def __init__(self, results):
        self.results = results


class TestCollect:
    def test_single_result(self):
        response = _Response([_Result([_Alternative("مرحبا بكم", 0.93)], "ar-SA")])
        assert asr_google._collect(response) == ("مرحبا بكم", 0.93, "ar")

    def test_split_results_are_joined_into_one_chunk_transcript(self):
        # The API may split even a short clip; the chunk still gets one string.
        response = _Response(
            [
                _Result([_Alternative("مرحبا", 0.9)], "ar-SA"),
                _Result([_Alternative("بكم", 0.8)], "ar-SA"),
            ]
        )
        text, confidence, language = asr_google._collect(response)
        assert text == "مرحبا بكم"
        assert confidence == pytest.approx(0.85)
        assert language == "ar"

    def test_no_results_is_empty_rather_than_an_error(self):
        # An empty transcript is meaningful downstream: filters rejects the
        # chunk as text:asr_empty when subtitles claim there was speech.
        assert asr_google._collect(_Response([])) == ("", None, None)

    def test_unset_confidence_does_not_count_as_zero(self):
        # protobuf reads an unset float back as 0.0; treating that as a real
        # score would drop the chunk on the confidence floor.
        response = _Response([_Result([_Alternative("مرحبا بكم")], "ar-SA")])
        assert asr_google._collect(response)[1] is None

    def test_results_without_alternatives_are_skipped(self):
        response = _Response([_Result([]), _Result([_Alternative("مرحبا بكم", 0.7)], "ar-SA")])
        assert asr_google._collect(response) == ("مرحبا بكم", 0.7, "ar")


@pytest.fixture
def fast_cfg():
    """Config with the retry sleep taken out, and no leaked fallback state."""
    asr_google._alternatives_rejected = False
    cfg = Config.load()
    cfg.asr.google.retry_backoff = 0.0
    yield cfg
    asr_google._alternatives_rejected = False


def _items(count: int, seconds: float = 1.0):
    from yt2ds.stages.segment import Chunk

    audio = np.full(int(16000 * seconds), 0.1, dtype=np.float32)
    return [(Chunk(index=i, start=0.0, end=seconds, speaker="A"), audio) for i in range(count)]


class _Registry:
    def __init__(self, client):
        self.speech_client = client


class _OkClient:
    """Answers every request with the same one-result response."""

    def __init__(self):
        self.calls = []

    def recognize(self, config, audio, timeout=None):
        self.calls.append(list(config.alternative_language_codes))
        return _Response([_Result([_Alternative("مرحبا بكم في هذا", 0.91)], "ar-SA")])


class TestTranscribeBatch:
    def test_one_request_per_chunk_with_scores_recorded(self, fast_cfg):
        client = _OkClient()
        items = _items(3)
        texts = asr_google.transcribe_batch(items, 16000, fast_cfg, _Registry(client))

        assert texts == ["مرحبا بكم في هذا"] * 3
        assert len(client.calls) == 3
        for chunk, _ in items:
            assert chunk.scores["asr_confidence"] == 0.91
            # Short code, not "ar-SA" -- the script gate keys off this.
            assert chunk.scores["asr_language"] == "ar"

    def test_audio_shorter_than_a_tenth_of_a_second_is_not_sent(self, fast_cfg):
        # The API has nothing to work with and would be billed anyway.
        client = _OkClient()
        assert asr_google.transcribe_batch(_items(1, seconds=0.02), 16000, fast_cfg, _Registry(client)) == [""]
        assert client.calls == []

    def test_alternative_languages_are_dropped_once_and_stay_dropped(self, fast_cfg):
        # latest_long rejects alternative_language_codes. Rediscovering that
        # per chunk would burn a failed request on every clip in the run.
        from google.api_core import exceptions as gexc

        class RejectsAlternatives(_OkClient):
            def recognize(self, config, audio, timeout=None):
                alternatives = list(config.alternative_language_codes)
                self.calls.append(alternatives)
                if alternatives:
                    raise gexc.InvalidArgument("alternative_language_codes is not supported for this model")
                return _Response([_Result([_Alternative("نعم", 0.8)], "ar-SA")])

        client = RejectsAlternatives()
        texts = asr_google.transcribe_batch(_items(3), 16000, fast_cfg, _Registry(client))
        assert texts == ["نعم"] * 3
        # One rejected attempt, then every request goes out without them.
        assert sum(1 for call in client.calls if call) == 1

    def test_transient_failures_are_retried(self, fast_cfg):
        from google.api_core import exceptions as gexc

        class Flaky:
            def __init__(self):
                self.attempts = 0

            def recognize(self, config, audio, timeout=None):
                self.attempts += 1
                if self.attempts < 3:
                    raise gexc.ServiceUnavailable("503")
                return _Response([_Result([_Alternative("نعم", 0.8)], "ar-SA")])

        client = Flaky()
        assert asr_google.transcribe_batch(_items(1), 16000, fast_cfg, _Registry(client)) == ["نعم"]
        assert client.attempts == 3

    def test_a_chunk_that_never_succeeds_does_not_kill_the_run(self, fast_cfg):
        # An empty transcript is a result filters knows how to handle; an
        # exception here would lose the whole video.
        from google.api_core import exceptions as gexc

        class Dead:
            def recognize(self, config, audio, timeout=None):
                raise gexc.ServiceUnavailable("503")

        assert asr_google.transcribe_batch(_items(1), 16000, fast_cfg, _Registry(Dead())) == [""]
