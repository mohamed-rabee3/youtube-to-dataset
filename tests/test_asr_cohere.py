"""The cohere backend's mapping from model rows back to clips.

Audio longer than the model's window comes back from the processor as several
rows. If those rows were read as one per clip, every transcript after a long
clip would land on the wrong clip -- the quietest way to corrupt a dataset.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yt2ds.config import Config
from yt2ds.stages import asr_cohere
from yt2ds.stages.segment import Chunk

SR = 16000
WINDOW = 35  # seconds, as the Cohere processors split


class _Processor:
    """Splits audio into WINDOW-second rows and decodes a row to its label."""

    tokenizer = SimpleNamespace(pad_token_id=0)

    def __init__(self):
        self.labels: list[str] = []

    def __call__(self, audio, sampling_rate, return_tensors, language):
        index, self.labels = [], []
        for clip, samples in enumerate(audio):
            pieces = max(1, int(np.ceil(samples.size / (WINDOW * SR))))
            for piece in range(pieces):
                index.append((clip, None if pieces == 1 else piece))
                self.labels.append(f"clip{int(samples[0])}-part{piece}")
        rows = len(index)
        return {
            "input_features": torch.zeros(rows, 4, 2),
            "attention_mask": torch.ones(rows, 4, dtype=torch.long),
            "audio_chunk_index": index,
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [self.labels[int(row[1]) - 10] for row in sequences]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(pad_token_id=0, eos_token_id=1)

    def generate(self, input_features, attention_mask, **kwargs):
        rows = input_features.shape[0]
        # A start token, then 10 + the row number so batch_decode can find the
        # row (offset clear of the pad and eos ids).
        sequences = torch.tensor([[2, 10 + i] for i in range(rows)])
        return SimpleNamespace(sequences=sequences, scores=None)

    def compute_transition_scores(self, sequences, scores, normalize_logits=True):
        return torch.full((sequences.shape[0], 1), -0.05)


def _clip(tag: int, seconds: float) -> np.ndarray:
    # The first sample carries the tag so the fake processor can name the row.
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    audio[0] = tag
    return audio


def test_long_clip_rows_are_joined_back_onto_their_own_clip():
    cfg = Config.load()
    registry = SimpleNamespace(device=torch.device("cpu"))
    audio = [_clip(0, 5), _clip(1, 80), _clip(2, 5)]  # the middle one spans 3 rows
    texts, scores = asr_cohere._transcribe(audio, "ar", _Processor(), _Model(), cfg, registry)
    assert texts == ["clip0-part0", "clip1-part0 clip1-part1 clip1-part2", "clip2-part0"]
    assert all(s == pytest.approx(-0.05) for s in scores)


def test_audio_too_short_to_decode_gets_empty_text_in_its_own_slot():
    cfg = Config.load()
    registry = SimpleNamespace(device=torch.device("cpu"))
    audio = [_clip(0, 5), np.zeros(100, dtype=np.float32), _clip(2, 5)]
    texts, scores = asr_cohere._transcribe(audio, "ar", _Processor(), _Model(), cfg, registry)
    assert texts == ["clip0-part0", "", "clip2-part0"]
    assert scores[1] is None


def test_verify_batch_writes_its_own_confidence(monkeypatch):
    cfg = Config.load()
    registry = SimpleNamespace(device=torch.device("cpu"), asr_verify=(_Processor(), _Model()))
    chunk = Chunk(index=0, start=0.0, end=5.0, speaker="A")
    chunk.scores["asr_language"] = "ar"
    texts = asr_cohere.verify_batch([(chunk, _clip(7, 5))], SR, cfg, registry)
    assert texts == ["clip7-part0"]
    assert chunk.scores["verify_confidence"] == pytest.approx(-0.05)
