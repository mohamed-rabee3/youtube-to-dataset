"""Arabic normalization and text comparison."""

from __future__ import annotations

import pytest

from yt2ds.arabic import arabic_ratio, cer, clean_for_output, normalize, wer


class TestNormalize:
    def test_strips_diacritics(self):
        assert normalize("أَهْلاً وَسَهْلاً") == normalize("اهلا وسهلا")

    @pytest.mark.parametrize("variant", ["أحمد", "إحمد", "آحمد", "ٱحمد"])
    def test_unifies_alef_variants(self, variant):
        assert normalize(variant) == normalize("احمد")

    def test_unifies_final_ya_and_ta_marbuta(self):
        assert normalize("على") == normalize("علي")
        assert normalize("مدرسة") == normalize("مدرسه")

    def test_maps_arabic_indic_digits(self):
        assert normalize("٢٠٢٦") == "2026"

    def test_removes_tatweel(self):
        assert normalize("مرحبـــا") == normalize("مرحبا")

    def test_strips_punctuation_and_bidi_marks(self):
        assert normalize("مرحبا، كيف الحال؟") == "مرحبا كيف الحال"
        assert normalize("‏مرحبا‎") == "مرحبا"

    def test_removes_caption_annotations(self):
        assert normalize("[موسيقى] مرحبا") == "مرحبا"
        assert normalize("(تصفيق) أهلا") == normalize("اهلا")

    def test_empty(self):
        assert normalize("") == ""


class TestPreservesOutput:
    def test_clean_for_output_keeps_diacritics_and_spelling(self):
        # The dataset must keep real orthography; normalization is only ever
        # used for comparison.
        text = "أَهْلاً وَسَهْلاً"
        assert clean_for_output(text) == text

    def test_clean_for_output_strips_caption_noise(self):
        assert clean_for_output("[موسيقى]  مرحبا  بكم ") == "مرحبا بكم"


class TestCer:
    def test_orthographic_variants_are_not_errors(self):
        assert cer("أهلاً وسهلاً", "اهلا وسهلا") == 0.0

    def test_identical(self):
        assert cer("مرحبا بكم", "مرحبا بكم") == 0.0

    def test_completely_different(self):
        assert cer("مرحبا", "الرياض") > 0.5

    def test_one_side_empty_is_total_disagreement(self):
        assert cer("مرحبا بكم", "") == 1.0
        assert cer("", "مرحبا بكم") == 1.0

    def test_both_empty_is_agreement(self):
        assert cer("", "") == 0.0

    def test_single_substitution(self):
        # "مرحبا" vs "مرحبي" -> one of five characters differs.
        assert cer("مرحبا", "مرحبي") == pytest.approx(0.2)

    def test_dialect_word_difference_registers(self):
        # Saudi "وش تبي" vs Egyptian "عايز ايه" say the same thing but are
        # different words; the gate must see them as disagreement.
        assert cer("وش تبي", "عايز ايه") > 0.4


class TestWer:
    def test_identical(self):
        assert wer("مرحبا بكم في هذا", "مرحبا بكم في هذا") == 0.0

    def test_one_word_wrong(self):
        assert wer("مرحبا بكم في هذا", "مرحبا بكم في ذاك") == pytest.approx(0.25)


class TestArabicRatio:
    def test_pure_arabic(self):
        assert arabic_ratio("مرحبا بكم") == 1.0

    def test_pure_latin(self):
        assert arabic_ratio("hello world") == 0.0

    def test_code_switching_is_partial(self):
        ratio = arabic_ratio("مرحبا everyone")
        assert 0.0 < ratio < 1.0

    def test_empty_and_punctuation_only(self):
        assert arabic_ratio("") == 0.0
        assert arabic_ratio("...!؟") == 0.0
