"""Text resolution and the quality gates."""

from __future__ import annotations

import pytest

from yt2ds.config import Config
from yt2ds.stages.filters import resolve_text, summarize
from yt2ds.stages.segment import Chunk


@pytest.fixture
def cfg():
    return Config.load()


def chunk(yt="", cohere="", start=0.0, end=5.0, **scores):
    c = Chunk(index=0, start=start, end=end, speaker="A")
    c.text_yt = yt
    c.text_cohere = cohere
    c.scores.update(scores)
    return c


def arabic_at(cps: float, duration: float, word_len: int = 4) -> str:
    """Arabic text hitting a target chars-per-second over ``duration``.

    Real words, not one long run of letters -- the word-count gate runs before
    the rate gate, so a single token would never reach it.
    """
    total = max(word_len * 2, int(round(cps * duration)))
    words = [("ا" * word_len) for _ in range(total // word_len)]
    remainder = total - len(words) * word_len
    if remainder:
        words[-1] += "ا" * remainder
    return " ".join(words)


class TestTextSource:
    def test_youtube_text_wins_when_both_agree(self, cfg):
        # Same sentence, different orthography: the dataset keeps YouTube's
        # spelling rather than Cohere's.
        c = chunk("أهلاً وسهلاً بكم جميعاً", "اهلا وسهلا بكم جميعا")
        resolve_text([c], "yt_auto", cfg)
        assert c.text == "أهلاً وسهلاً بكم جميعاً"
        assert c.text_source == "yt_auto"
        assert c.reject_reason is None

    def test_cohere_is_canonical_when_there_are_no_subtitles(self, cfg):
        c = chunk("", "مرحبا بكم في هذا المقطع", start=0.0, end=3.0)
        resolve_text([c], None, cfg)
        assert c.text_source == "cohere"
        assert c.text == "مرحبا بكم في هذا المقطع"
        assert c.reject_reason is None

    def test_cer_gate_drops_disagreement(self, cfg):
        c = chunk("مرحبا بكم في هذا المقطع", "الرياض مدينة كبيرة جدا اليوم")
        resolve_text([c], "yt_auto", cfg)
        assert c.reject_reason is not None
        assert c.reject_reason.startswith("text:cer")

    def test_cer_gate_does_not_apply_to_the_cohere_only_path(self, cfg):
        c = chunk("", "مرحبا بكم في هذا المقطع", start=0.0, end=3.0)
        resolve_text([c], None, cfg)
        assert c.scores["cer_yt_vs_cohere"] is None
        assert c.reject_reason is None

    def test_silent_cohere_against_confident_subtitles_is_rejected(self, cfg):
        # Subtitles claim speech, the ASR heard nothing. Do not trust either.
        c = chunk("مرحبا بكم في هذا المقطع", "")
        resolve_text([c], "yt_auto", cfg)
        assert c.reject_reason == "text:cohere_empty"

    def test_no_text_at_all(self, cfg):
        c = chunk("", "")
        resolve_text([c], None, cfg)
        assert c.reject_reason == "text:empty"
        assert c.text_source == "none"

    def test_both_texts_are_preserved_for_retuning(self, cfg):
        c = chunk("مرحبا بكم في هذا", "مرحبا بكم في ذاك")
        resolve_text([c], "yt_auto", cfg)
        assert c.text_yt and c.text_cohere
        assert c.scores["cer_yt_vs_cohere"] is not None


class TestCharsPerSecond:
    """Regression guard: the gate was originally calibrated for Latin script.

    Arabic is far more compact -- measured Saudi podcast speech runs about
    5-13 characters per second -- so a Latin-calibrated floor rejected
    essentially every valid chunk.
    """

    @pytest.mark.parametrize("cps", [5.4, 9.5, 12.8])
    def test_realistic_arabic_speech_rates_pass(self, cfg, cps):
        text = arabic_at(cps, 5.0)
        c = chunk(text, text, start=0.0, end=5.0)
        resolve_text([c], "yt_auto", cfg)
        assert c.reject_reason is None, f"{cps} chars/sec should pass, got {c.reject_reason}"

    def test_text_far_too_long_for_its_audio_is_rejected(self, cfg):
        # 40 chars/sec -- text bled in from a neighbouring chunk.
        text = arabic_at(40.0, 5.0)
        c = chunk(text, text, start=0.0, end=5.0)
        resolve_text([c], "yt_auto", cfg)
        assert c.reject_reason is not None
        assert "cps_high" in c.reject_reason

    def test_text_far_too_short_for_its_audio_is_rejected(self, cfg):
        # 2 chars/sec -- the transcript missed most of the speech.
        text = arabic_at(2.0, 12.0)
        c = chunk(text, text, start=0.0, end=12.0)
        resolve_text([c], "yt_auto", cfg)
        assert c.reject_reason is not None
        assert "cps_low" in c.reject_reason


class TestOtherTextGates:
    def test_latin_text_claiming_to_be_arabic_is_rejected(self, cfg):
        # Decoded under the Arabic prompt but came out Latin: the model was
        # forced into a language the audio is not in.
        c = chunk(
            "hello everyone welcome back",
            "hello everyone welcome back",
            start=0.0,
            end=3.0,
            asr_language="ar",
        )
        resolve_text([c], None, cfg)
        assert c.reject_reason is not None
        assert "script=ar" in c.reject_reason

    def test_english_text_is_kept_when_decoded_as_english(self, cfg):
        # The model transcribes English as well as Arabic, so English output
        # is a valid result rather than a failure.
        c = chunk("", "hello everyone and welcome back", start=0.0, end=3.0, asr_language="en")
        resolve_text([c], None, cfg)
        assert c.reject_reason is None
        assert c.text_source == "cohere"
        assert c.scores["script_ratio"] == pytest.approx(1.0)

    def test_code_switched_arabic_with_english_words_passes(self, cfg):
        c = chunk("", "مرحبا بكم في بودكاست اليوم مع Google و Apple", start=0.0, end=4.0, asr_language="ar")
        resolve_text([c], None, cfg)
        assert c.reject_reason is None

    def test_low_asr_confidence_rejected_on_the_cohere_only_path(self, cfg):
        text = arabic_at(9.5, 5.0)
        c = chunk("", text, start=0.0, end=5.0, asr_language="ar", asr_confidence=-3.0)
        resolve_text([c], None, cfg)
        assert c.reject_reason is not None
        assert "asr_confidence" in c.reject_reason

    def test_asr_confidence_not_applied_when_subtitles_corroborate(self, cfg):
        # With a YouTube transcript to agree with, the CER gate is the
        # stronger signal and low model confidence alone should not drop it.
        text = arabic_at(9.5, 5.0)
        c = chunk(text, text, start=0.0, end=5.0, asr_language="ar", asr_confidence=-3.0)
        resolve_text([c], "yt_auto", cfg)
        assert c.reject_reason is None

    def test_single_word_is_rejected(self, cfg):
        c = chunk("مرحبا", "مرحبا", start=0.0, end=2.0)
        resolve_text([c], "yt_auto", cfg)
        assert c.reject_reason is not None

    def test_boundary_clipped_word_is_rejected(self, cfg):
        text = arabic_at(9.5, 5.0)
        c = chunk(text, text, start=0.0, end=5.0, boundary_clipped=True)
        resolve_text([c], "yt_auto", cfg)
        assert c.reject_reason == "text:boundary_clipped"

    def test_boundary_clipping_can_be_disabled(self, cfg):
        cfg.filters.drop_boundary_clipped_words = False
        text = arabic_at(9.5, 5.0)
        c = chunk(text, text, start=0.0, end=5.0, boundary_clipped=True)
        resolve_text([c], "yt_auto", cfg)
        assert c.reject_reason is None

    def test_low_alignment_score_rejected_only_on_the_subtitle_path(self, cfg):
        text = arabic_at(9.5, 5.0)
        subtitle = chunk(text, text, start=0.0, end=5.0, align_score=-9.0)
        resolve_text([subtitle], "yt_auto", cfg)
        assert subtitle.reject_reason is not None

        # The Cohere-only path has no alignment, so the score is irrelevant.
        cohere_only = chunk("", text, start=0.0, end=5.0, align_score=-9.0)
        resolve_text([cohere_only], None, cfg)
        assert cohere_only.reject_reason is None


class TestSummarize:
    def test_groups_by_category(self, cfg):
        chunks = [
            chunk(arabic_at(9.5, 5.0), arabic_at(9.5, 5.0)),
            chunk("", ""),
            chunk("مرحبا بكم في هذا المقطع", "الرياض مدينة كبيرة جدا اليوم"),
        ]
        resolve_text(chunks, "yt_auto", cfg)
        counts = summarize(chunks)
        assert counts["kept"] == 1
        assert counts["text"] == 2
