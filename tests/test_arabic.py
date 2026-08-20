"""Arabic normalization and text comparison."""

from __future__ import annotations

import pytest

from yt2ds.arabic import (
    arabic_ratio,
    cer,
    clean_for_output,
    normalize,
    same_skeleton,
    strip_final_tashkeel,
    strip_tashkeel,
    tashkeel_ratio,
    wer,
)


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


class TestTashkeel:
    """Diacritics for TTS: no case endings, and the letters never move."""

    def test_strip_tashkeel_leaves_the_skeleton(self):
        assert strip_tashkeel("مَرْحَبًا") == "مرحبا"

    def test_ratio_counts_marks_per_letter(self):
        assert tashkeel_ratio("مرحبا") == 0.0
        assert tashkeel_ratio("مَرْح") > 0.0

    def test_final_haraka_is_removed(self):
        # bi-kum + sukun on the final meem
        assert strip_final_tashkeel("بِكُمْ") == "بِكُم"

    def test_final_tanwin_is_removed(self):
        # kitaab + dammatan
        assert strip_final_tashkeel("كِتَابٌ") == "كِتَاب"

    def test_tanwin_fath_before_a_silent_alef_is_removed(self):
        # marhaban: the tanwin sits on the BAA, not on the final alef, so
        # clearing only the last letter would leave the case ending in place.
        assert strip_final_tashkeel(
            "مَرْحَبًا"
        ) == "مَرْحَبا"

    def test_an_ordinary_fatha_before_a_final_alef_survives(self):
        # qaalaa: that fatha is part of the word, not an ending.
        word = "قَالَا"
        assert strip_final_tashkeel(word) == word

    def test_internal_shadda_survives(self):
        # ummu-ki: shadda stays, the final kasra goes.
        assert strip_final_tashkeel(
            "أُمُّكِ"
        ) == "أُمُّك"

    def test_punctuation_is_not_absorbed_into_the_word(self):
        # Regression: a bidi-mangled character class once swallowed the Arabic
        # question mark into the word, leaving its final letter marked.
        assert strip_final_tashkeel(
            "الحَالُ؟"
        ) == "الحَال؟"

    def test_latin_and_spacing_pass_through(self):
        text = "قال hello يا صَاح"
        assert strip_final_tashkeel(text) == text

    def test_stripping_never_changes_the_letters(self):
        for word in (
            "بِكُمْ",
            "كِتَابٌ",
            "مَرْحَبًا",
            "فَتَاةً",
        ):
            assert same_skeleton(word, strip_final_tashkeel(word))

    def test_same_skeleton_detects_a_rewritten_word(self):
        assert same_skeleton("مَرحبا", "مرحبا")
        assert not same_skeleton("مرحبا", "مرحبا بكم")


class TestTatweelIsNotOutput:
    """Tatweel is typography, not orthography: it must never reach the dataset.

    A diacritizer asked to add marks will occasionally insert one as somewhere
    to hang a vowel, which would put a silent character into a TTS target.
    """

    def test_tatweel_is_stripped_from_output(self):
        from yt2ds.arabic import clean_for_output

        assert clean_for_output("أَنْقُـل") == "أَنْقُل"
        assert clean_for_output("علـى طاولـه") == "على طاوله"

    def test_stripping_tatweel_keeps_every_mark_and_letter(self):
        from yt2ds.arabic import clean_for_output, same_skeleton, tashkeel_ratio

        marked = "أَنْقُـل أَسْئِلَتِكُـم"
        out = clean_for_output(marked)
        assert "ـ" not in out
        assert same_skeleton(out, "أنقل أسئلتكم")
        assert tashkeel_ratio(out) > 0

    def test_a_trailing_tatweel_cannot_shield_a_case_ending(self):
        # Tatweel sits inside the Arabic letter range, so left in place it
        # would pose as the final letter and keep strip_final_tashkeel from
        # reaching the real one.
        from yt2ds.arabic import clean_for_output, strip_final_tashkeel

        assert strip_final_tashkeel(clean_for_output("الْمَرْكَزُـ")) == "الْمَرْكَز"
