"""Speaker embedding models behind one interface.

Two families are supported, chosen by the name in ``speakers.embed_model``:

* ``Wespeaker/...`` -- WeSpeaker ResNet checkpoints (default: ResNet293-LM).
  Loaded straight from the ``avg_model.pt`` on the Hub into the small network
  defined below, so it runs on the GPU without the ``wespeaker`` package.
* ``speechbrain/...`` -- SpeechBrain ECAPA-TDNN, the previous default.

Why ResNet293 is the default. On this project's own Saudi podcast data
(2,185 clips, 18 episodes, DiariZen's within-episode labels as ground truth) it
cut the equal error rate from 2.94% to 2.32% within episodes and from 2.72% to
1.58% for host-versus-guest across episodes -- the comparison speaker linking
actually makes. That matches published results showing deep ResNets holding up
better than ECAPA out of domain.

Every encoder takes 16 kHz mono float32 audio and returns one L2-normalized
row per input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class SpeakerEncoder(Protocol):
    dim: int

    def embed(self, audio: list[np.ndarray]) -> np.ndarray: ...


def load_encoder(name: str, device: torch.device, cache_dir: str = ".cache") -> SpeakerEncoder:
    """Build the encoder named by ``speakers.embed_model``."""
    lowered = name.lower()
    if lowered.startswith("speechbrain/"):
        return SpeechBrainEncoder(name, device, cache_dir)
    if lowered.startswith("wespeaker/") or Path(name, "avg_model.pt").exists():
        return WeSpeakerEncoder(name, device)
    raise ValueError(f"unsupported speakers.embed_model: {name!r} (expected a Wespeaker/ or speechbrain/ model)")


def _l2(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms > 0, norms, 1.0)


# ------------------------------------------------------------------ SpeechBrain
class SpeechBrainEncoder:
    dim = 192

    def __init__(self, name: str, device: torch.device, cache_dir: str) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        # SpeechBrain parses the device string itself and cannot handle a bare
        # "cuda", so hand it an explicit index.
        device_string = f"cuda:{device.index or 0}" if device.type == "cuda" else str(device)
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=name,
            savedir=f"{cache_dir}/speechbrain/{name.replace('/', '_')}",
            run_opts={"device": device_string},
        )

    @torch.inference_mode()
    def embed(self, audio: list[np.ndarray]) -> np.ndarray:
        rows = []
        for samples in audio:
            wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0).to(self.device)
            rows.append(self.model.encode_batch(wav).reshape(-1).float().cpu().numpy())
        return _l2(np.stack(rows)) if rows else np.zeros((0, self.dim), dtype=np.float32)


# -------------------------------------------------------------------- WeSpeaker
class WeSpeakerEncoder:
    """WeSpeaker ResNet on 80-dim Kaldi fbank with per-utterance mean removal."""

    def __init__(self, name: str, device: torch.device) -> None:
        import yaml

        directory = Path(name)
        if not (directory / "avg_model.pt").exists():
            from huggingface_hub import snapshot_download

            directory = Path(snapshot_download(name, allow_patterns=["avg_model.pt", "config.yaml"]))
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        args = config["model_args"]
        self.model = build_resnet(config["model"], feat_dim=args["feat_dim"], embed_dim=args["embed_dim"])
        state = torch.load(directory / "avg_model.pt", map_location="cpu", weights_only=True)
        # The checkpoint carries the training-time classifier head; only the
        # embedding network is needed.
        state = {k: v for k, v in state.items() if not k.startswith("projection.")}
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.device = device
        self.dim = args["embed_dim"]
        self.fbank_args = config.get("dataset_args", {}).get("fbank_args", {})

    def _features(self, samples: np.ndarray) -> torch.Tensor:
        from torchaudio.compliance import kaldi

        # WeSpeaker computes fbank on int16-scaled audio, and dither is a
        # training-time augmentation, so it is off for inference.
        wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0) * (1 << 15)
        feats = kaldi.fbank(
            wav,
            num_mel_bins=self.fbank_args.get("num_mel_bins", 80),
            frame_length=self.fbank_args.get("frame_length", 25),
            frame_shift=self.fbank_args.get("frame_shift", 10),
            dither=0.0,
            sample_frequency=SAMPLE_RATE,
            window_type="hamming",
            use_energy=False,
        )
        return feats - feats.mean(dim=0, keepdim=True)

    @torch.inference_mode()
    def embed(self, audio: list[np.ndarray]) -> np.ndarray:
        rows = []
        for samples in audio:
            feats = self._features(samples).unsqueeze(0).to(self.device)
            rows.append(self.model(feats).reshape(-1).float().cpu().numpy())
        return _l2(np.stack(rows)) if rows else np.zeros((0, self.dim), dtype=np.float32)


# The network below follows WeSpeaker's ResNet (Apache-2.0,
# github.com/wenet-e2e/wespeaker, wespeaker/models/resnet.py) closely enough to
# load its checkpoints unchanged: same module names, same shapes.
class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        out_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False), nn.BatchNorm2d(out_planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class _ResNet(nn.Module):
    def __init__(self, block, num_blocks: list[int], feat_dim: int, embed_dim: int, m_channels: int = 32) -> None:
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], 2)
        # Temporal statistics pooling: mean and std over time of every
        # (channel, frequency) cell, then one linear embedding layer.
        pooled = m_channels * 8 * block.expansion * (feat_dim // 8) * 2
        self.seg_1 = nn.Linear(pooled, embed_dim)

    def _make_layer(self, block, planes: int, count: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (count - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.permute(0, 2, 1).unsqueeze(1)  # (B, T, F) -> (B, 1, F, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        mean = x.mean(dim=-1).flatten(1)
        std = torch.sqrt(torch.var(x, dim=-1) + 1e-7).flatten(1)
        return self.seg_1(torch.cat((mean, std), dim=1))


_ARCHITECTURES = {
    "ResNet34": (_BasicBlock, [3, 4, 6, 3]),
    "ResNet221": (_Bottleneck, [6, 16, 48, 3]),
    "ResNet293": (_Bottleneck, [10, 20, 64, 3]),
}


def build_resnet(name: str, feat_dim: int, embed_dim: int) -> nn.Module:
    if name not in _ARCHITECTURES:
        raise ValueError(f"unsupported WeSpeaker architecture {name!r}; expected one of {sorted(_ARCHITECTURES)}")
    block, layers = _ARCHITECTURES[name]
    return _ResNet(block, layers, feat_dim=feat_dim, embed_dim=embed_dim)
