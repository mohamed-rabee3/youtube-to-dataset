"""The Hugging Face schema for yt2ds metadata, shared by the upload scripts.

Every column is typed explicitly. Inferring the schema from the rows does not
work: columns such as ``global_speaker`` or the alignment fields are null on
whole videos, and a shard where a column is null throughout would type itself
differently from its neighbours and break the loader.

Needs the ``datasets`` package, which only the upload scripts use.
"""

from __future__ import annotations

from typing import Any

# Datasets built before the ASR columns were renamed.
_LEGACY = {"text_asr": "text_cohere", "cer_yt_vs_asr": "cer_yt_vs_cohere"}

_FLOAT = (
    "start end duration speech_start speech_end voiced_seconds max_internal_pause "
    "speaker_conf speaker_distance global_speaker_distance cer_yt_vs_asr cer_verify "
    "asr_confidence verify_confidence align_score music_score speech_score vocal_ratio "
    "accompaniment_ratio squim_mos squim_stoi squim_pesq squim_si_sdr snr_db peak_dbfs "
    "rms_dbfs clipping_ratio arabic_ratio script_ratio chars_per_sec source_lufs lufs "
    "gain_db gain_limited_db out_peak_dbfs tashkeel_ratio"
).split()
_STRING = (
    "audio_file video_id video_url title channel upload_date speaker global_speaker text "
    "text_source text_yt text_asr text_verify asr_language music_label language tashkeel_source"
).split()

# Column order on the Hub.
_ORDER = (
    "audio_file video_id video_url title channel upload_date chunk_index start end duration "
    "speech_start speech_end voiced_seconds max_internal_pause speaker global_speaker "
    "speaker_conf speaker_distance global_speaker_distance text text_source text_yt text_asr "
    "cer_yt_vs_asr text_verify cer_verify asr_language asr_confidence verify_confidence "
    "asr_confidence_alt align_score aligned_words boundary_clipped vocals_isolated music_score "
    "music_label speech_score vocal_ratio accompaniment_ratio squim_mos squim_stoi squim_pesq "
    "squim_si_sdr snr_db peak_dbfs rms_dbfs clipping_ratio arabic_ratio script_ratio "
    "chars_per_sec source_lufs lufs gain_db gain_limited_db out_peak_dbfs sample_rate language "
    "tashkeel_ratio tashkeel_source"
).split()


def features(sample_rate: int = 24000, with_source_audio_file: bool = False):
    """The ``datasets.Features`` for a yt2ds dataset (or one speaker folder of it)."""
    from datasets import Audio, Features, Value

    def typed(name: str):
        if name in _FLOAT:
            return Value("float64")
        if name in _STRING:
            return Value("string")
        if name in ("chunk_index", "aligned_words", "sample_rate"):
            return Value("int32")
        if name in ("boundary_clipped", "vocals_isolated"):
            return Value("bool")
        if name == "asr_confidence_alt":
            # Flattened to a fixed struct: the raw dict is keyed by whichever
            # languages lost the decode, which is not a stable parquet schema.
            return {"ar": Value("float64"), "en": Value("float64")}
        raise KeyError(name)

    spec: dict[str, Any] = {"audio": Audio(sampling_rate=sample_rate)}
    for name in _ORDER:
        spec[name] = typed(name)
        if name == "audio_file" and with_source_audio_file:
            # Where the clip sits in the dataset's own wavs/ tree, before the
            # speaker split filed it under its voice.
            spec["source_audio_file"] = Value("string")
    return Features(spec)


def column_value(row: dict, column: str) -> Any:
    """``row``'s value for ``column``, as the schema types it."""
    if column == "asr_confidence_alt":
        alt = row.get(column) or {}
        return {"ar": alt.get("ar"), "en": alt.get("en")}
    value = row.get(column)
    if value is None and column in _LEGACY:
        value = row.get(_LEGACY[column])
    return value
