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

    Keeps orthography and diacritics intact; only removes caption artefacts and
    normalizes whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = strip_caption_annotations(text)
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
