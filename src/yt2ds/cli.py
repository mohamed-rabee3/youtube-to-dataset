"""Command line interface.

    yt2ds run URL [URL ...] --out dataset/
    yt2ds run --urls-file links.txt --out dataset/ --workers 4
    yt2ds run mp3-corpus/ --out dataset/
    yt2ds report dataset/ --link-speakers
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path


from .config import Config
from .io import Workspace, read_jsonl, write_json


# Where clips land when their speaker pooled less than min_speaker_seconds.


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
        # The Speech-to-Text client logs every gRPC channel event at DEBUG,
        # and there is one request per chunk.
        "grpc",
        "google.auth",
        "google.api_core",
        # audio-separator logs every ffmpeg line of every conversion at DEBUG.
        "pydub",
        "pydub.converter",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt2ds",
        description="Turn YouTube links into an Arabic TTS / voice-cloning dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="build a dataset from YouTube links or local audio files")
    run.add_argument(
        "urls",
        nargs="*",
        help="video, playlist, or channel URLs, or local audio files / directories",
    )
    run.add_argument("--urls-file", type=Path, help="file with one URL or local path per line (# comments allowed)")
    run.add_argument(
        "--cookies",
        type=Path,
        help="cookies.txt (Netscape format); the fix for 'sign in to confirm you're not a bot'",
    )
    run.add_argument("--cookies-from-browser", help="read cookies from a logged-in browser, e.g. chrome / firefox")
    run.add_argument(
        "--no-playlist",
        action="store_true",
        help="for a watch?v=...&list=... URL take only that video, not the playlist",
    )
    run.add_argument(
        "--player-clients",
        help="comma-separated YouTube clients to try per video, in order "
        "(default: default,tv_simply,web_safari,android_vr,ios,mweb)",
    )
    run.add_argument("--out", type=Path, required=True, help="dataset output directory")
    run.add_argument("--config", type=Path, help="YAML config (defaults to configs/default.yaml)")
    run.add_argument("--workers", type=int, help="parallel downloads")
    run.add_argument("--device", help="cuda | cuda:1 | cpu")
    run.add_argument("--no-resume", action="store_true", help="reprocess videos already finished")
    run.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="keep each video's raw download, working WAVs, captions and info JSON (~0.5 GB per source hour)",
    )
    run.add_argument(
        "--keep-mp3",
        action="store_true",
        help="keep a full-length archival mp3 per video (~27 MB each); deleted with the rest by default",
    )

    run.add_argument(
        "--no-separate",
        action="store_true",
        help="do not isolate the voice from the music bed (chunks then carry whatever music is behind them)",
    )
    run.add_argument("--separate-model", help="audio-separator model filename for vocal isolation")
    run.add_argument(
        "--reject-music",
        action="store_true",
        help="drop chunks whose residual music is over the thresholds, instead of only scoring them",
    )

    # Common threshold overrides, so tuning does not require editing YAML.
    run.add_argument("--min-mos", type=float, help="minimum SQUIM MOS")
    run.add_argument("--max-cer", type=float, help="max CER between YouTube subs and the ASR transcript")
    run.add_argument("--max-music", type=float, help="max AST music score")
    run.add_argument("--max-accompaniment", type=float, help="max Demucs accompaniment energy ratio")
    run.add_argument("--min-duration", type=float, help="minimum chunk seconds")
    run.add_argument("--max-duration", type=float, help="maximum seconds of speech per chunk")
    run.add_argument("--out-sample-rate", type=int, help="output WAV sample rate")
    run.add_argument("--keep-overlap", action="store_true", help="do not discard overlapped speech")
    run.add_argument(
        "--asr-backend",
        choices=["google", "cohere", "google_batch"],
        help="transcription backend: cohere = local GPU models (default: diacritized "
        "transcript from Cohere-Speech-Tashkeel-2B, cross-checked by Cohere Transcribe Arabic), "
        "google = per-chunk Cloud Speech-to-Text (needs credentials, billed per audio minute), "
        "google_batch = whole-episode V2 BatchRecognize -- this run emits chunks without text "
        "and `yt2ds transcribe` fills them in",
    )
    run.add_argument("--asr-model", help="Hub id of the cohere backend's transcription model")
    run.add_argument(
        "--verify-model",
        help="Hub id of the cohere backend's cross-check model, or 'none' to skip the check",
    )
    run.add_argument(
        "--google-asr-model",
        help="Speech-to-Text model for the google backend (default: latest_long)",
    )
    run.add_argument(
        "--google-credentials",
        type=Path,
        help="service-account JSON for the google backend "
        "(defaults to $GOOGLE_APPLICATION_CREDENTIALS, then application default credentials)",
    )
    run.add_argument(
        "--languages",
        help="comma-separated ASR languages (default: ar). On google the first is the primary "
        "and the rest are alternatives it detects among; on cohere each chunk is decoded once per "
        "language and the highest-confidence result wins. Use 'ar' alone for Arabic-only material.",
    )
    run.add_argument("-v", "--verbose", action="count", default=0)
    run.add_argument("-q", "--quiet", action="store_true")

    transcribe = sub.add_parser(
        "transcribe",
        help="fill in the text of a dataset built with --asr-backend google_batch, "
        "by transcribing whole episodes through Speech-to-Text V2 BatchRecognize",
    )
    transcribe.add_argument("dataset", type=Path)
    transcribe.add_argument("--config", type=Path)
    transcribe.add_argument("--bucket", help="Cloud Storage bucket for the uploaded audio (required)")
    transcribe.add_argument("--location", help="Speech-to-Text location; ar-SA needs 'global' or 'us' (default global)")
    transcribe.add_argument("--model", help="V2 model name (default 'long', the V2 name for v1's latest_long)")
    transcribe.add_argument("--languages", help="comma-separated BCP-47 codes (default ar-SA)")
    transcribe.add_argument(
        "--no-dynamic-batching",
        action="store_true",
        help="use the standard tier instead of the ~75%% cheaper dynamic-batch tier, "
        "trading cost for a much shorter latency SLA",
    )
    transcribe.add_argument("--workers", type=int, help="episodes transcribed concurrently")
    transcribe.add_argument(
        "--limit",
        type=int,
        help="transcribe only the first N untranscribed episodes, for a costed trial run",
    )
    transcribe.add_argument("-v", "--verbose", action="count", default=0)
    transcribe.add_argument("-q", "--quiet", action="store_true")

    report = sub.add_parser("report", help="summarize an existing dataset")
    report.add_argument("dataset", type=Path)
    report.add_argument("--config", type=Path)
    report.add_argument("-v", "--verbose", action="count", default=0)
    report.add_argument("-q", "--quiet", action="store_true")

    voices = sub.add_parser(
        "speakers",
        help="link speakers across videos and write one folder per voice under <dataset>/speakers/",
    )
    voices.add_argument("dataset", type=Path)
    voices.add_argument(
        "--threshold",
        type=float,
        help="cosine distance below which two per-video speakers are the same voice (default 0.40); "
        "raise to pool a voice recorded under varying conditions, lower if distinct people merge",
    )
    voices.add_argument(
        "--min-seconds",
        type=float,
        help="a voice pooling less audio than this is diarization debris and goes to _unassigned/ "
        "(default 30); 0 keeps every identity",
    )
    voices.add_argument(
        "--max-clip-distance",
        type=float,
        help="a clip further than this from its voice's centroid goes to _unassigned/ (default 0.45)",
    )
    voices.add_argument("--only", help="comma-separated voices to write, e.g. GLOBAL_SPEAKER_00")
    voices.add_argument("--out", type=Path, help="output root (default <dataset>/speakers)")
    voices.add_argument("--report-only", action="store_true", help="print the ranking, write nothing")
    voices.add_argument("--config", type=Path)
    voices.add_argument("-v", "--verbose", action="count", default=0)
    voices.add_argument("-q", "--quiet", action="store_true")

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
        "filters.max_cer_yt_vs_asr": args.max_cer,
        "music.max_music_score": args.max_music,
        "music.max_accompaniment_ratio": args.max_accompaniment,
        "segment.min_duration": args.min_duration,
        "segment.max_duration": args.max_duration,
        "audio.out_sample_rate": args.out_sample_rate,
        "separate.model": getattr(args, "separate_model", None),
        "asr.backend": getattr(args, "asr_backend", None),
        "asr.google.model": getattr(args, "google_asr_model", None),
        "asr.model": getattr(args, "asr_model", None),
    }
    verify = getattr(args, "verify_model", None)
    if verify is not None:
        mapping["asr.verify_model"] = "" if verify.lower() == "none" else verify
    if getattr(args, "google_credentials", None):
        mapping["asr.google.credentials_file"] = str(args.google_credentials)
    if getattr(args, "no_separate", False):
        mapping["separate.enabled"] = False
    if getattr(args, "reject_music", False):
        mapping["music.reject"] = True
    if args.keep_overlap:
        mapping["diarize.drop_overlap"] = False
    if args.keep_intermediates:
        mapping["runtime.keep_intermediates"] = True
    if args.keep_mp3:
        mapping["audio.keep_mp3"] = True
    if getattr(args, "languages", None):
        mapping["asr.languages"] = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    if getattr(args, "no_playlist", False):
        mapping["download.follow_playlist"] = False
    if getattr(args, "cookies", None):
        mapping["download.cookies_file"] = str(args.cookies)
    if getattr(args, "cookies_from_browser", None):
        mapping["download.cookies_from_browser"] = args.cookies_from_browser
    if getattr(args, "player_clients", None):
        mapping["download.player_clients"] = [c.strip() for c in args.player_clients.split(",") if c.strip()]
    return {k: v for k, v in mapping.items() if v is not None}


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline

    urls = _collect_urls(args)
    if not urls:
        raise SystemExit("no sources given; pass URLs or local paths as arguments, or via --urls-file")

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
        print(f"\nretry them with:  yt2ds run --urls-file {ws.failed} --out {ws.root}")
    return 1 if failed and kept == 0 else 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Phase two: transcribe whole episodes, then cut the words into chunks.

    The GPU pass has already decided which chunks exist and written their
    audio; all that is missing is text. So this uploads each episode, waits for
    the batch operations, assigns words to chunks by time, and only then runs
    the text-quality gates -- which is the first point at which they can run.
    """
    from .stages import asr_google_batch as batch

    overrides = {
        "asr.google_batch.bucket": getattr(args, "bucket", None),
        "asr.google_batch.location": getattr(args, "location", None),
        "asr.google_batch.model": getattr(args, "model", None),
        "asr.google_batch.max_workers": getattr(args, "workers", None),
    }
    if getattr(args, "languages", None):
        overrides["asr.google_batch.language_codes"] = [c.strip() for c in args.languages.split(",") if c.strip()]
    if getattr(args, "no_dynamic_batching", False):
        overrides["asr.google_batch.dynamic_batching"] = False
    cfg = Config.load(args.config, {k: v for k, v in overrides.items() if v is not None})

    ws = Workspace(args.dataset)
    if not ws.metadata.exists():
        raise SystemExit(f"no metadata.jsonl in {ws.root}")

    staged = sorted(ws.asr_audio.glob("*.flac"))
    pending = [p for p in staged if not (ws.asr_words / f"{p.stem}.json").exists()]
    if args.limit:
        pending = pending[: args.limit]

    print(f"dataset  : {ws.root}")
    print(f"episodes : {len(staged)} staged, {len(pending)} still to transcribe")

    if pending:
        minutes = sum(_flac_minutes(p) for p in pending)
        rate = 0.003 if cfg.asr.google_batch.dynamic_batching else 0.016
        tier = "dynamic batch" if cfg.asr.google_batch.dynamic_batching else "standard"
        print(f"audio    : {minutes / 60:.1f} h -> ~${minutes * rate:.2f} at the {tier} rate")

        try:
            uris = batch.upload(pending, cfg)
        except batch.GoogleBatchUnavailable as exc:
            raise SystemExit(str(exc))
        for video_id, words in batch.transcribe(uris, cfg).items():
            batch.save_words(ws.asr_words / f"{video_id}.json", words)

    return _apply_batch_text(ws, cfg)


def _flac_minutes(path: Path) -> float:
    """Duration of a staged FLAC, for the pre-flight cost estimate."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate / 60.0
    except Exception:  # noqa: BLE001 - an estimate must never block the run
        return 0.0


def _apply_batch_text(ws: Workspace, cfg: Config) -> int:
    """Merge cached words into the metadata, then run the text gates.

    Chunks the gates reject move to ``rejected.jsonl`` and their WAVs are
    deleted, so the deliverable ends up exactly as it would have had the text
    been there from the start.
    """
    from .stages import asr_google_batch as batch
    from .stages import filters

    rows = list(read_jsonl(ws.metadata))
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_video[str(row.get("video_id"))].append(row)

    filled_videos = 0
    for video_id, video_rows in by_video.items():
        words = batch.load_words(ws.asr_words / f"{video_id}.json")
        if not words:
            continue
        batch.assign_to_rows(video_rows, words)
        filled_videos += 1

    kept, rejected = filters.gate_rows(rows, cfg)

    _rewrite_jsonl(ws.metadata, kept)
    # Keep phase one's audio-side rejections (music, quality, speaker) and drop
    # only text rejections from an earlier transcription pass, so re-running
    # this command neither loses them nor duplicates its own.
    existing = [r for r in read_jsonl(ws.rejected) if not str(r.get("reject_reason") or "").startswith("text:")]
    _rewrite_jsonl(ws.rejected, existing + rejected)

    removed = 0
    for row in rejected:
        target = ws.wavs / str(row.get("audio_file") or "")
        if target.is_file():
            target.unlink()
            removed += 1

    print(f"\ntranscribed {filled_videos} episode(s)")
    print(f"clips kept  : {len(kept)}")
    print(f"clips cut   : {len(rejected)} on the text gates ({removed} wav(s) deleted)")
    still = sum(1 for r in kept if r.get("text_source") == "pending")
    if still:
        print(f"still pending: {still} clip(s) whose episode has no transcript yet")
    return 0


def _rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def cmd_report(args: argparse.Namespace) -> int:
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
        per_speaker[row.get("global_speaker") or row.get("speaker", "?")] += duration
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
        print(f"  {source:<16} {count:>6}")

    if reasons:
        print("\nrejection reasons:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            share = 100 * count / max(len(rejected), 1)
            print(f"  {reason:<12} {count:>6}  ({share:4.1f}%)")

    print("\ntop speakers:")
    for speaker, seconds in sorted(per_speaker.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {speaker:<28} {seconds / 60:7.1f} min")

    write_json(ws.speakers, {"per_speaker_seconds": dict(per_speaker), "per_video_seconds": dict(per_video)})
    return 0


def cmd_speakers(args: argparse.Namespace) -> int:
    from .speaker_split import print_report, split

    overrides = {
        "speakers.link_threshold": args.threshold,
        "speakers.min_speaker_seconds": args.min_seconds,
        "speakers.max_clip_distance": args.max_clip_distance,
    }
    cfg = Config.load(args.config, {k: v for k, v in overrides.items() if v is not None})
    ws = Workspace(args.dataset)
    if not ws.metadata.exists():
        raise SystemExit(f"no metadata.jsonl in {ws.root}")

    only = {name.strip() for name in args.only.split(",") if name.strip()} if args.only else None
    result = split(ws, cfg, out_root=args.out, only=only, write=not args.report_only)
    print_report(result.summary)
    if args.report_only:
        print("\nreport only: nothing written")
    else:
        print(f"\nspeaker folders written under {args.out or ws.root / 'speakers'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(-1 if getattr(args, "quiet", False) else getattr(args, "verbose", 0))

    if args.command == "run":
        return cmd_run(args)
    if args.command == "transcribe":
        return cmd_transcribe(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "speakers":
        return cmd_speakers(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
