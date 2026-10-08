"""Speaker encoders: WeSpeaker checkpoints load into the local network."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from yt2ds.embedders import WeSpeakerEncoder, build_resnet, load_encoder


@pytest.fixture
def checkpoint(tmp_path):
    """A WeSpeaker-format model directory with random ResNet34 weights."""
    torch.manual_seed(0)
    model = build_resnet("ResNet34", feat_dim=80, embed_dim=256)
    state = model.state_dict()
    # Real checkpoints carry the training classifier; the loader must drop it.
    state["projection.weight"] = torch.zeros(10, 256)
    torch.save(state, tmp_path / "avg_model.pt")
    config = {
        "model": "ResNet34",
        "model_args": {"feat_dim": 80, "embed_dim": 256},
        "dataset_args": {"fbank_args": {"num_mel_bins": 80, "frame_length": 25, "frame_shift": 10}},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    return tmp_path


def test_checkpoint_loads_and_embeds_to_unit_vectors(checkpoint):
    encoder = load_encoder(str(checkpoint), torch.device("cpu"))
    assert isinstance(encoder, WeSpeakerEncoder)
    rng = np.random.default_rng(0)
    out = encoder.embed([rng.standard_normal(16000).astype(np.float32) * 0.1 for _ in range(3)])
    assert out.shape == (3, 256)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embedding_is_independent_of_batch_company(checkpoint):
    encoder = load_encoder(str(checkpoint), torch.device("cpu"))
    rng = np.random.default_rng(1)
    a = rng.standard_normal(16000).astype(np.float32) * 0.1
    b = rng.standard_normal(24000).astype(np.float32) * 0.1
    alone = encoder.embed([a])[0]
    together = encoder.embed([a, b])[0]
    assert np.allclose(alone, together, atol=1e-5)


@pytest.mark.parametrize("name, pooled", [("ResNet34", 256 * 10 * 2), ("ResNet293", 1024 * 10 * 2)])
def test_architectures_match_wespeaker_shapes(name, pooled):
    model = build_resnet(name, feat_dim=80, embed_dim=256)
    assert model.seg_1.in_features == pooled


def test_unknown_model_family_is_rejected():
    with pytest.raises(ValueError, match="speakers.embed_model"):
        load_encoder("someone/some-model", torch.device("cpu"))
