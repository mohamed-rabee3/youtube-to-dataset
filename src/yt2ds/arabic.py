"""Arabic text handling.

Normalization here exists for *comparison* only -- deciding whether the YouTube
subtitle and Cohere's transcription say the same thing. The text written to the
dataset always keeps its original orthography, because a TTS model needs to see
the real spelling, including dialectal forms.

The normalizer is deliberately dialect-agnostic: it collapses orthographic
variation (hamza seats, final ya, ta marbuta) that carries no pronunciation
difference worth splitting on, and does *not* try to map dialectal words onto
Modern Standard Arabic.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat, tanwin, shadda, sukun, superscript alef, and the Quranic annotation
# range. All are optional in ordinary Arabic writing, so their presence in one
# transcript and absence in the other is not a disagreement.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"

# Orthographic variants that sound identical.
_CHAR_MAP = {
    "آ": "ا",  # آ -> ا
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ٱ": "ا",  # ٱ -> ا
    "ى": "ي",  # ى -> ي
    "ة": "ه",  # ة -> ه
    "ک": "ك",  # ک -> ك  (Persian kaf, common in scraped text)
    "ی": "ي",  # ی -> ي  (Persian ya)
    "ھ": "ه",
}

# Arabic-Indic and extended Arabic-Indic digits -> ASCII.
_DIGIT_MAP = {chr(0x0660 + i): str(i) for i in range(10)}
_DIGIT_MAP.update({chr(0x06F0 + i): str(i) for i in range(10)})

_TRANSLATION = str.maketrans({**_CHAR_MAP, **_DIGIT_MAP, _TATWEEL: None})

# Punctuation, both ASCII and Arabic, plus the bidi/zero-width controls that
# ride along in scraped subtitles.
_PUNCT = re.compile(
    r"[!-/:-@\[-`{-~«»،؛؟٪-٭۔"
    r" -⁯⸀-⹿​-‏‪-‮﻿]"
)
_WS = re.compile(r"\s+")

_ARABIC_LETTER = re.compile(r"[ء-ي٠-٩ٮ-ۿ]")
_LETTER_OR_DIGIT = re.compile(r"[^\W_]", re.UNICODE)

# Bracketed caption annotations: [موسيقى], (تصفيق), [Music], >> speaker tags.
_CAPTION_ANNOTATION = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]|^>>+\s*", re.MULTILINE)


def strip_caption_annotations(text: str) -> str:
    """Remove ``[موسيقى]`` / ``[Music]`` / ``>>`` markers YouTube injects."""
    return _WS.sub(" ", _CAPTION_ANNOTATION.sub(" ", text)).strip()


def normalize(text: str) -> str:
    """Normalize for comparison. Not for output."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = strip_caption_annotations(text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_TRANSLATION)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def clean_for_output(text: str) -> str:
    """Light cleanup for the text stored in the dataset.

    Keeps orthography and diacritics intact; only removes caption artefacts,
    tatweel, and normalizes whitespace.

    Tatweel (U+0640) is the one character dropped rather than preserved. It is
    a typographic elongation that stretches the joining stroke and carries no
    sound, so it is not orthography -- and a diacritizer asked for marks will
    sometimes reach for it as somewhere to hang one, which would otherwise put
    a silent character into a TTS training target. Note that it falls inside
    the Arabic letter range, so leaving it in would also let it pose as a
    word's final letter and shield a real case ending from
    :func:`strip_final_tashkeel`.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = strip_caption_annotations(text)
    text = text.replace(_TATWEEL, "")
    return _WS.sub(" ", text).strip()


def arabic_ratio(text: str) -> float:
    """Fraction of alphanumeric characters that are Arabic script.

    Used to drop chunks whose transcript came out as English, or as the
    ``[Music]`` placeholder, or as digits only.
    """
    chars = _LETTER_OR_DIGIT.findall(text or "")
    if not chars:
        return 0.0
    arabic = sum(1 for c in chars if _ARABIC_LETTER.match(c))
    return arabic / len(chars)


# -- tashkeel -----------------------------------------------------------
# A diacritized transcript is written for TTS, where the marks say how a word
# is *pronounced*. Two rules follow from that and are enforced here in code
# rather than trusted to whatever produced the marks:
#
# 1. No mark on a word's final letter. Arabic case endings (i'rab) are a
#    grammatical property, not an audible one -- in ordinary speech the word
#    is stopped on (waqf) and the ending is not voiced. Marking it teaches a
#    TTS model to pronounce something the speaker never said.
# 2. The letters themselves must not change. Adding marks is a strictly
#    additive edit; if the bare skeleton moved, the text was rewritten rather
#    than diacritized, and it can no longer be trusted against the audio.


def strip_tashkeel(text: str) -> str:
    """Remove every diacritic, leaving the bare consonantal skeleton."""
    return _DIACRITICS.sub("", text or "")


def has_tashkeel(text: str) -> bool:
    return bool(_DIACRITICS.search(text or ""))


def tashkeel_ratio(text: str) -> float:
    """Diacritics per Arabic letter. Fully-marked prose sits near 0.8-1.0."""
    letters = _ARABIC_LETTER.findall(strip_tashkeel(text))
    if not letters:
        return 0.0
    return len(_DIACRITICS.findall(text or "")) / len(letters)


def strip_final_tashkeel(text: str) -> str:
    """Drop diacritics sitting on the last letter of each word (rule 1 above).

    Operates on runs of Arabic letters and marks, so punctuation, Latin text
    and whitespace pass through untouched. Marks *inside* a word are kept --
    only the trailing letter is cleared, however many marks it carries.

    One case needs more than "clear the last letter": tanwin fath is written
    on the *penultimate* letter, with a silent alef after it (``مرحبًا``). It
    is still a case ending and still inaudible in waqf, so it goes too. An
    ordinary fatha in that position is part of the word and stays.
    """
    if not text:
        return ""

    def clear(match: re.Match) -> str:
        word = match.group(0)
        # Walk back over the trailing marks to the final real letter.
        end = len(word)
        while end > 0 and _DIACRITICS.match(word[end - 1]):
            end -= 1
        word = word[:end]
        # ...then the tanwin-fath-plus-alef spelling of the same ending.
        if len(word) >= 2 and word[-1] in _SILENT_TANWIN_ALEF and word[-2] == _TANWIN_FATH:
            word = word[:-2] + word[-1]
        return word

    return _ARABIC_WORD.sub(clear, text)


def same_skeleton(a: str, b: str) -> bool:
    """True when two strings differ only by diacritics (rule 2 above)."""
    return strip_tashkeel(a or "").split() == strip_tashkeel(b or "").split()


def same_words(a: str, b: str) -> bool:
    """True when two strings are the same words under orthographic variation.

    Looser than :func:`same_skeleton`: hamza seats, ta marbuta, final ya and
    punctuation may differ, nothing else. A diacritizer that also restores the
    hamza a transcript was missing (``اصوات`` -> ``أصوات``) has improved the
    text rather than rewritten it, and its marks are still trustworthy; one
    that adds, drops or reorders a word has not.
    """
    left, right = normalize(a or "").split(), normalize(b or "").split()
    return left == right and len(left) > 0


# Built from explicit escapes rather than literal Arabic: typing the ranges
# as characters lets bidi reordering silently fuse them into a much wider
# class (one that swallowed "؟" into the preceding word).
_TASHKEEL_CHARS = "\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED"
_LETTER_CHARS = "\u0621-\u064A\u066E\u066F\u0671-\u06D5\u06EE\u06EF\u06FA-\u06FF"

#: A word for tashkeel purposes: Arabic letters with their marks, nothing else.
_ARABIC_WORD = re.compile(f"[{_LETTER_CHARS}][{_LETTER_CHARS}{_TASHKEEL_CHARS}]*")

_TANWIN_FATH = "\u064B"
_SILENT_TANWIN_ALEF = ("\u0627", "\u0649")  # alef, alef maqsura

_LATIN_LETTER = re.compile(r"[A-Za-z0-9]")


def latin_ratio(text: str) -> float:
    """Fraction of alphanumeric characters that are Latin script."""
    chars = _LETTER_OR_DIGIT.findall(text or "")
    if not chars:
        return 0.0
    latin = sum(1 for c in chars if _LATIN_LETTER.match(c))
    return latin / len(chars)


def script_ratio(text: str, language: str) -> float:
    """How much of ``text`` is in the script the language is written in.

    Code-switching is expected -- an Arabic sentence containing an English
    brand name should still read as mostly Arabic -- so this is a proportion
    rather than a purity test.
    """
    if language == "ar":
        return arabic_ratio(text)
    if language == "en":
        return latin_ratio(text)
    # Unknown language: accept whichever script dominates, so a new language
    # in the config does not silently reject everything.
    return max(arabic_ratio(text), latin_ratio(text))


def levenshtein(a: str, b: str) -> int:
    """Edit distance over sequences. Two rows, so memory is O(min(len))."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate between two already-normalized strings.

    Returns 0.0 when both are empty and 1.0 when only one is, so an empty
    transcript on either side reads as total disagreement rather than a
    perfect match.
    """
    ref = normalize(reference).replace(" ", "")
    hyp = normalize(hypothesis).replace(" ", "")
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate between two strings, normalized first."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    # Levenshtein over word tokens; map each word to a private-use codepoint so
    # the character routine can be reused.
    vocab: dict[str, str] = {}

    def encode(words: list[str]) -> str:
        out = []
        for w in words:
            if w not in vocab:
                vocab[w] = chr(0xE000 + len(vocab))
            out.append(vocab[w])
        return "".join(out)

    return levenshtein(encode(ref), encode(hyp)) / len(ref)


def words_for_alignment(text: str) -> list[str]:
    """Tokenize into words for forced alignment.

    Alignment needs the same token sequence the caller will later reassemble,
    so this keeps original spelling and only drops empty tokens.
    """
    cleaned = strip_caption_annotations(unicodedata.normalize("NFC", text or ""))
    return [w for w in _WS.split(cleaned) if w]
