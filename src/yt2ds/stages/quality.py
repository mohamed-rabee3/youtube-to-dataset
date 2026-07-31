"""Per-chunk audio quality metrics.

Uses TorchAudio-SQUIM, which estimates reference-based metrics without a
reference: SQUIM_OBJECTIVE predicts STOI, PESQ and SI-SDR from the degraded
signal alone, and SQUIM_SUBJECTIVE predicts a listener MOS. Both ship inside
torchaudio, so this costs no extra model downloads.

SQUIM_SUBJECTIVE needs a *non-matching reference* -- any clean speech clip,
unrelated to the input. Rather than bundling an audio asset, the pipeline picks
the highest-STOI chunk of the same video as the reference. That is
self-contained, and it means the MOS is effectively "how does this clip compare
to the cleanest thing this speaker recorded", which is the comparison worth
making when deciding what to keep for voice cloning.

Plain signal statistics (clipping, peak, SNR) are computed alongside, because
they catch failures SQUIM does not care about -- a chunk that is simply too
quiet, or digitally clipped.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..config import Config
from ..models import ModelRegistry
from .audio import resample
from .segment import Chunk

log = logging.getLogger(__name__)

_SQUIM_SR = 16000
# SQUIM is trained on short utterances; cap the input so a 12 s chunk and a
# 2 s chunk are scored comparably and memory stays flat.
_MAX_SECONDS = 10.0


def score_batch(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    cfg: Config,
    registry: ModelRegistry,
    reference: np.ndarray | None = None,
) -> None:
    """Write quality metrics into ``chunk.scores`` for a batch."""
    if not items:
        return

    for chunk, samples in items:
        _signal_stats(chunk, samples)

    _squim_objective(items, sr, registry)
    if reference is not None and reference.size:
        _squim_subjective(items, sr, registry, reference)


def apply_gates(chunks: list[Chunk], cfg: Config) -> None:
    """Reject chunks failing the configured quality thresholds."""
    q = cfg.quality
    for chunk in chunks:
        if chunk.reject_reason is not None:
            continue
        s = chunk.scores

        clipping = s.get("clipping_ratio")
        if clipping is not None and clipping > q.max_clipping_ratio:
            chunk.reject(f"quality:clipping={clipping:.4f}")
            continue

        peak = s.get("peak_dbfs")
        if peak is not None and not np.isnan(peak):
            if peak < q.min_peak_dbfs:
                chunk.reject(f"quality:too_quiet={peak:.1f}dBFS")
                continue
            if peak > q.max_peak_dbfs:
                chunk.reject(f"quality:too_hot={peak:.1f}dBFS")
                continue

        snr = s.get("snr_db")
        if snr is not None and not np.isnan(snr) and snr < q.min_snr_db:
            chunk.reject(f"quality:snr={snr:.1f}dB")
            continue

        stoi = s.get("squim_stoi")
        if stoi is not None and stoi < q.min_stoi:
            chunk.reject(f"quality:stoi={stoi:.3f}")
            continue

        pesq = s.get("squim_pesq")
        if pesq is not None and pesq < q.min_pesq:
            chunk.reject(f"quality:pesq={pesq:.2f}")
            continue

        mos = s.get("squim_mos")
        if mos is not None and mos < q.min_mos:
            chunk.reject(f"quality:mos={mos:.2f}")
            continue


def pick_reference(items: list[tuple[Chunk, np.ndarray]], sr: int) -> np.ndarray | None:
    """Choose the cleanest chunk in a batch to serve as the SQUIM reference."""
    best: np.ndarray | None = None
    best_stoi = -1.0
    for chunk, samples in items:
        stoi = chunk.scores.get("squim_stoi")
        if stoi is not None and stoi > best_stoi and samples.size:
            best_stoi = stoi
            best = samples
    if best is None:
        return None
    audio = resample(best, sr, _SQUIM_SR)
    return audio[: int(_MAX_SECONDS * _SQUIM_SR)]


# ------------------------------------------------------------------ stats
def _signal_stats(chunk: Chunk, samples: np.ndarray) -> None:
    if samples.size == 0:
        chunk.scores.update(
            {"peak_dbfs": float("nan"), "rms_dbfs": float("nan"), "clipping_ratio": 1.0, "snr_db": float("nan")}
        )
        return

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples**2)))
    chunk.scores["peak_dbfs"] = round(20 * np.log10(peak), 3) if peak > 0 else float("-inf")
    chunk.scores["rms_dbfs"] = round(20 * np.log10(rms), 3) if rms > 0 else float("-inf")
    # Samples sitting at or above full scale, which indicates the source was
    # already clipped before we touched it.
    chunk.scores["clipping_ratio"] = round(float(np.mean(np.abs(samples) >= 0.999)), 6)
    chunk.scores["snr_db"] = round(_estimate_snr(samples), 3)


def _estimate_snr(samples: np.ndarray, frame: int = 1024) -> float:
    """Crude SNR: loud frames vs the quietest decile of frames.

    Not a calibrated measurement, but it reliably separates a clean voice in a
    quiet room from a voice buried in hiss, which is all the gate needs.
    """
    if samples.size < frame * 4:
        return float("nan")
    usable = (samples.size // frame) * frame
    frames = samples[:usable].reshape(-1, frame)
    energies = np.mean(frames**2, axis=1)
    energies = energies[np.isfinite(energies)]
    if energies.size < 4:
        return float("nan")

    noise = float(np.percentile(energies, 10))
    signal = float(np.percentile(energies, 90))
    if noise <= 1e-12 or signal <= 1e-12:
        return float("nan")
    return float(10 * np.log10(signal / noise))


# ------------------------------------------------------------------ SQUIM
@torch.inference_mode()
def _squim_objective(items: list[tuple[Chunk, np.ndarray]], sr: int, registry: ModelRegistry) -> None:
    model = registry.squim_objective
    for chunk, samples in items:
        audio = _prepare(samples, sr)
        if audio is None:
            continue
        wav = torch.from_numpy(audio).unsqueeze(0).to(registry.device)
        try:
            stoi, pesq, si_sdr = model(wav)
        except Exception as exc:  # noqa: BLE001 - a chunk SQUIM cannot handle
            log.debug("SQUIM objective failed on chunk %d: %s", chunk.index, exc)
            continue
        chunk.scores["squim_stoi"] = round(float(stoi[0]), 5)
        chunk.scores["squim_pesq"] = round(float(pesq[0]), 5)
        chunk.scores["squim_si_sdr"] = round(float(si_sdr[0]), 5)


@torch.inference_mode()
def _squim_subjective(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    registry: ModelRegistry,
    reference: np.ndarray,
) -> None:
    model = registry.squim_subjective
    ref = torch.from_numpy(reference).unsqueeze(0).to(registry.device)
    for chunk, samples in items:
        audio = _prepare(samples, sr)
        if audio is None:
            continue
        wav = torch.from_numpy(audio).unsqueeze(0).to(registry.device)
        # The reference must match the input length for the model's framing.
        ref_matched = _match_length(ref, wav.shape[-1])
        try:
            mos = model(wav, ref_matched)
        except Exception as exc:  # noqa: BLE001
            log.debug("SQUIM subjective failed on chunk %d: %s", chunk.index, exc)
            continue
        chunk.scores["squim_mos"] = round(float(mos[0]), 5)


def _prepare(samples: np.ndarray, sr: int) -> np.ndarray | None:
    if samples.size == 0:
        return None
    audio = resample(samples, sr, _SQUIM_SR)
    audio = audio[: int(_MAX_SECONDS * _SQUIM_SR)]
    if audio.size < _SQUIM_SR // 2:
        return None
    return np.ascontiguousarray(audio, dtype=np.float32)


def _match_length(reference: torch.Tensor, length: int) -> torch.Tensor:
    """Tile or trim the reference so it is exactly ``length`` samples."""
    have = reference.shape[-1]
    if have == length:
        return reference
    if have > length:
        return reference[..., :length]
    repeats = (length + have - 1) // have
    return reference.repeat(1, repeats)[..., :length]
