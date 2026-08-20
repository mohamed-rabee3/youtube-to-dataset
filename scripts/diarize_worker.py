#!/usr/bin/env python
"""DiariZen worker. Runs inside .venv-diarize, never inside the main venv.

DiariZen vendors pyannote-audio and expects an older torch than the
transformers 5.x / torch 2.9 stack that Cohere Transcribe Arabic needs, so the
two never share an interpreter. The contract between them is this script:

    python scripts/diarize_worker.py <input.wav> <output.json> [--model ID]

It writes JSON of the form::

    {
      "turns":    [{"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00"}, ...],
      "overlaps": [{"start": 4.0, "end": 4.4}, ...],
      "speakers": ["SPEAKER_00", "SPEAKER_01"],
      "model":    "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    }

Overlap regions are derived by intersecting turns from different speakers.
DiariZen's powerset classification head models simultaneous speech directly, so
these are real detections rather than a heuristic -- and for voice cloning they
matter as much as the turns themselves, since a clip containing two voices is
unusable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"


def compute_overlaps(turns: list[dict]) -> list[dict]:
    """Intervals covered by two or more distinct speakers.

    Sweep over turn boundaries counting active distinct speakers, then merge
    adjacent intervals where the count exceeds one.
    """
    if len(turns) < 2:
        return []

    points = sorted({round(t[key], 4) for t in turns for key in ("start", "end")})
    raw: list[tuple[float, float]] = []
    for lo, hi in zip(points, points[1:]):
        if hi <= lo:
            continue
        mid = (lo + hi) / 2.0
        active = {t["speaker"] for t in turns if t["start"] <= mid < t["end"]}
        if len(active) > 1:
            raw.append((lo, hi))

    merged: list[dict] = []
    for lo, hi in raw:
        if merged and lo - merged[-1]["end"] <= 1e-3:
            merged[-1]["end"] = hi
        else:
            merged.append({"start": lo, "end": hi})
    return merged


def diarize(wav: Path, model_id: str, device: str, batch_size: int = 0) -> dict:
    import torch
    from diarizen.pipelines.inference import DiariZenPipeline

    pipeline = DiariZenPipeline.from_pretrained(model_id)
    if device.startswith("cuda") and torch.cuda.is_available():
        pipeline.to(torch.device(device))

    # The model's own config.toml picks the batch size (32), which assumes it
    # has the GPU to itself. It does not: the parent process keeps its models
    # resident across videos. A smaller batch trades speed for fitting.
    if batch_size > 0:
        pipeline.segmentation_batch_size = batch_size
        pipeline.embedding_batch_size = batch_size

    annotation = pipeline(str(wav))

    turns: list[dict] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        start, end = float(segment.start), float(segment.end)
        if end > start:
            turns.append({"start": start, "end": end, "speaker": str(speaker)})
    turns.sort(key=lambda t: (t["start"], t["end"]))

    return {
        "turns": turns,
        "overlaps": compute_overlaps(turns),
        "speakers": sorted({t["speaker"] for t in turns}),
        "model": model_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DiariZen on a 16 kHz mono WAV.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="segmentation/embedding batch size; 0 keeps the model config's own (32)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    try:
        result = diarize(args.input, args.model, args.device, args.batch_size)
    except Exception as exc:  # noqa: BLE001 - report cleanly to the parent process
        print(f"diarization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{len(result['turns'])} turns, {len(result['speakers'])} speakers, "
        f"{len(result['overlaps'])} overlap regions",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
