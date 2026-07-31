"""Command line interface.

    yt2ds run URL [URL ...] --out dataset/
    yt2ds run --urls-file links.txt --out dataset/ --workers 4
    yt2ds report dataset/ --link-speakers
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import Config
from .io import Workspace, read_jsonl, write_json


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity < 0 else logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These drown out pipeline progress -- httpcore in particular logs every
    # HTTP header of every model download at DEBUG.
    for noisy in (
        "urllib3",
        "httpcore",
        "httpx",
        "filelock",
        "speechbrain",
        "numba",
        "matplotlib",
        "huggingface_hub",
        "fsspec",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt2ds",
        description="Turn YouTube links into an Arabic TTS / voice-cloning dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="build a dataset from one or more YouTube links")
    run.add_argument("urls", nargs="*", help="video, playlist, or channel URLs")
    run.add_argument("--urls-file", type=Path, help="file with one URL per line (# comments allowed)")
    run.add_argument("--out", type=Path, required=True, help="dataset output directory")
    run.add_argument("--config", type=Path, help="YAML config (defaults to configs/default.yaml)")
    run.add_argument("--workers", type=int, help="parallel downloads")
    run.add_argument("--device", help="cuda | cuda:1 | cpu")
    run.add_argument("--no-resume", action="store_true", help="reprocess videos already finished")
    run.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="keep each video's raw download and working WAVs (~0.5 GB per source hour)",
    )

    # Common threshold overrides, so tuning does not require editing YAML.
    run.add_argument("--min-mos", type=float, help="minimum SQUIM MOS")
    run.add_argument("--max-cer", type=float, help="max CER between YouTube subs and Cohere")
    run.add_argument("--max-music", type=float, help="max AST music score")
    run.add_argument("--max-accompaniment", type=float, help="max Demucs accompaniment energy ratio")
    run.add_argument("--min-duration", type=float, help="minimum chunk seconds")
    run.add_argument("--max-duration", type=float, help="maximum chunk seconds")
    run.add_argument("--out-sample-rate", type=int, help="output WAV sample rate")
    run.add_argument("--keep-overlap", action="store_true", help="do not discard overlapped speech")
    run.add_argument(
        "--languages",
        help="comma-separated ASR languages, highest confidence wins (default: ar,en). "
        "Use 'ar' alone to halve ASR time on Arabic-only material.",
    )
    run.add_argument("-v", "--verbose", action="count", default=0)
    run.add_argument("-q", "--quiet", action="store_true")

    report = sub.add_parser("report", help="summarize an existing dataset")
    report.add_argument("dataset", type=Path)
    report.add_argument("--link-speakers", action="store_true", help="cluster speakers across videos")
    report.add_argument("--config", type=Path)
    report.add_argument("-v", "--verbose", action="count", default=0)
    report.add_argument("-q", "--quiet", action="store_true")

    return parser


def _collect_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.urls or [])
    if args.urls_file:
        if not args.urls_file.exists():
            raise SystemExit(f"urls file not found: {args.urls_file}")
        for line in args.urls_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    mapping = {
        "runtime.download_workers": args.workers,
        "runtime.device": args.device,
        "quality.min_mos": args.min_mos,
        "filters.max_cer_yt_vs_cohere": args.max_cer,
        "music.max_music_score": args.max_music,
        "music.max_accompaniment_ratio": args.max_accompaniment,
        "segment.min_duration": args.min_duration,
        "segment.max_duration": args.max_duration,
        "audio.out_sample_rate": args.out_sample_rate,
    }
    if args.keep_overlap:
        mapping["diarize.drop_overlap"] = False
    if args.keep_intermediates:
        mapping["runtime.keep_intermediates"] = True
    if getattr(args, "languages", None):
        mapping["asr.languages"] = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    return {k: v for k, v in mapping.items() if v is not None}


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline

    urls = _collect_urls(args)
    if not urls:
        raise SystemExit("no URLs given; pass them as arguments or via --urls-file")

    cfg = Config.load(args.config, _overrides(args))
    ws = Workspace(args.out)
    pipeline = Pipeline(ws, cfg)
    results = pipeline.run(urls, resume=not args.no_resume)

    kept = sum(r.kept for r in results)
    rejected = sum(r.rejected for r in results)
    hours = sum(r.seconds_kept for r in results) / 3600
    failed = [r for r in results if r.error]

    print()
    print(f"videos processed : {len(results)}")
    print(f"clips kept       : {kept}")
    print(f"clips rejected   : {rejected}")
    print(f"audio kept       : {hours:.2f} h")
    print(f"output           : {ws.root}")
    if failed:
        print(f"\n{len(failed)} video(s) failed:")
        for r in failed:
            print(f"  {r.video_id}: {r.error}")
    return 1 if failed and kept == 0 else 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    ws = Workspace(args.dataset)
    if not ws.metadata.exists():
        raise SystemExit(f"no metadata.jsonl in {ws.root}")

    rows = list(read_jsonl(ws.metadata))
    rejected = list(read_jsonl(ws.rejected))

    per_speaker: dict[str, float] = defaultdict(float)
    per_video: dict[str, float] = defaultdict(float)
    per_source: dict[str, int] = defaultdict(int)
    total = 0.0
    for row in rows:
        duration = float(row.get("duration") or 0.0)
        total += duration
        per_speaker[row.get("speaker", "?")] += duration
        per_video[row.get("video_id", "?")] += duration
        per_source[row.get("text_source", "?")] += 1

    reasons: dict[str, int] = defaultdict(int)
    for row in rejected:
        reason = str(row.get("reject_reason", "?"))
        reasons[reason.split(":", 1)[0]] += 1

    print(f"dataset : {ws.root}")
    print(f"clips   : {len(rows)} kept, {len(rejected)} rejected")
    print(f"audio   : {total / 3600:.2f} h across {len(per_video)} video(s)")

    print("\ntext source:")
    for source, count in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"  {source:<12} {count:>6}")

    if reasons:
        print("\nrejection reasons:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            share = 100 * count / max(len(rejected), 1)
            print(f"  {reason:<12} {count:>6}  ({share:4.1f}%)")

    linked = _link_speakers(ws, cfg, per_speaker) if args.link_speakers else {}
    if linked:
        _rewrite_global_speakers(ws, linked)
        pooled: dict[str, float] = defaultdict(float)
        for row in read_jsonl(ws.metadata):
            if row.get("global_speaker"):
                pooled[row["global_speaker"]] += float(row.get("duration") or 0.0)
        print(f"\nglobal speakers after linking: {len(pooled)} (from {len(per_speaker)} per-video)")
        for speaker, seconds in sorted(pooled.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {speaker:<22} {seconds / 60:7.1f} min")
    else:
        print("\ntop speakers:")
        for speaker, seconds in sorted(per_speaker.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {speaker:<28} {seconds / 60:7.1f} min")

    write_json(
        ws.speakers,
        {
            "per_speaker_seconds": dict(per_speaker),
            "per_video_seconds": dict(per_video),
            "global_speakers": linked,
        },
    )
    return 0


def _link_speakers(ws: Workspace, cfg: Config, per_speaker: dict[str, float]) -> dict[str, str]:
    """Cluster the per-video speaker centroids saved during the run.

    Kept seconds per speaker are passed through so global labels are numbered
    by how much usable audio each voice has.
    """
    from .stages.speakers import link_across_videos

    centroids: dict[str, np.ndarray] = {}
    for path in sorted(ws.embeddings.glob("*.npz")):
        with np.load(path) as data:
            for key in data.files:
                centroids[key] = data[key]
    if not centroids:
        print("\nno speaker embeddings found; run the pipeline first", file=sys.stderr)
        return {}
    return link_across_videos(centroids, cfg.speakers.link_threshold, weights=per_speaker)


def _rewrite_global_speakers(ws: Workspace, linked: dict[str, str]) -> None:
    """Fill in the ``global_speaker`` column now that clustering has run."""
    rows = list(read_jsonl(ws.metadata))
    for row in rows:
        row["global_speaker"] = linked.get(row.get("speaker", ""))
    tmp = ws.metadata.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(ws.metadata)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(-1 if getattr(args, "quiet", False) else getattr(args, "verbose", 0))

    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
