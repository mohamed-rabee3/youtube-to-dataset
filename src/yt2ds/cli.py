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
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import Config
from .io import Workspace, read_jsonl, write_json


# Where clips land when their speaker pooled less than min_speaker_seconds.
_UNASSIGNED = "_unassigned"


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
    run.add_argument("--max-duration", type=float, help="maximum chunk seconds")
    run.add_argument("--out-sample-rate", type=int, help="output WAV sample rate")
    run.add_argument("--keep-overlap", action="store_true", help="do not discard overlapped speech")
    run.add_argument(
        "--asr-backend",
        choices=["google", "cohere", "google_batch"],
        help="transcription backend: google = per-chunk Cloud Speech-to-Text (default, needs "
        "credentials, billed per audio minute), cohere = local Cohere Transcribe Arabic on the "
        "GPU, google_batch = whole-episode V2 BatchRecognize -- this run emits chunks without "
        "text and `yt2ds transcribe` fills them in, far cheaper than per-chunk requests",
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
        help="comma-separated ASR languages (default: ar,en). On google the first is the primary "
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
    report.add_argument("--link-speakers", action="store_true", help="cluster speakers across videos")
    report.add_argument(
        "--link-threshold",
        type=float,
        help="cosine distance below which two per-video speakers are the same voice (default 0.35); "
        "raise to pool a voice recorded under varying conditions, lower if distinct people are merging",
    )
    report.add_argument(
        "--min-speaker-seconds",
        type=float,
        help="a global speaker pooling less than this many seconds is debris, not a voice; "
        "its clips go to _unassigned/ (default 30). 0 disables the floor",
    )
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
        "separate.model": getattr(args, "separate_model", None),
        "asr.backend": getattr(args, "asr_backend", None),
        "asr.google.model": getattr(args, "google_asr_model", None),
    }
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
    overrides = {
        "speakers.link_threshold": getattr(args, "link_threshold", None),
        "speakers.min_speaker_seconds": getattr(args, "min_speaker_seconds", None),
    }
    cfg = Config.load(args.config, {k: v for k, v in overrides.items() if v is not None})
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
    linked = link_across_videos(centroids, cfg.speakers.link_threshold, weights=per_speaker)
    return _drop_thin_speakers(linked, per_speaker, cfg.speakers.min_speaker_seconds)


def _drop_thin_speakers(
    linked: dict[str, str],
    per_speaker: dict[str, float],
    min_seconds: float,
) -> dict[str, str]:
    """Unassign global identities that pooled too little audio to be a voice.

    Diarization splits off a cluster for every scrap it cannot place -- an
    intro voiceover, a burst of crosstalk, a few seconds of phone audio. Each
    is a real cluster and survives linking, so without a floor a two-person
    interview yields ten "speakers", eight of them holding seconds.

    The floor is applied to the *pooled* total rather than per video: a genuine
    voice accumulates across the corpus, while debris stays debris no matter
    how many videos it appears in. Survivors are renumbered contiguously so the
    labels stay dense, still ordered by duration.
    """
    if min_seconds <= 0:
        return linked

    pooled: dict[str, float] = defaultdict(float)
    for key, label in linked.items():
        pooled[label] += per_speaker.get(key, 0.0)

    keep = [label for label in pooled if pooled[label] >= min_seconds]
    if len(keep) == len(pooled):
        return linked

    order = sorted(keep, key=lambda label: -pooled[label])
    renumbered = {label: f"GLOBAL_SPEAKER_{i:02d}" for i, label in enumerate(order)}
    dropped = len(pooled) - len(keep)
    dropped_seconds = sum(pooled[label] for label in pooled if label not in renumbered)
    print(
        f"\ndropped {dropped} speaker(s) under {min_seconds:.0f}s "
        f"({dropped_seconds / 60:.1f} min total) -> {_UNASSIGNED}/"
    )
    return {key: renumbered[label] for key, label in linked.items() if label in renumbered}


def _rewrite_global_speakers(ws: Workspace, linked: dict[str, str]) -> None:
    """Fill in the ``global_speaker`` column now that clustering has run.

    The run wrote each clip under its per-video speaker folder, which is the
    finest split available while videos are processed one at a time. Now that
    the same voice has been recognised across videos, the clips move into one
    directory per global identity -- so a podcast host across two hundred
    episodes ends up as a single folder rather than two hundred.
    """
    rows = list(read_jsonl(ws.metadata))
    for row in rows:
        row["global_speaker"] = linked.get(row.get("speaker", ""))
    _regroup_wavs(ws, rows)
    tmp = ws.metadata.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(ws.metadata)


def _regroup_wavs(ws: Workspace, rows: list[dict[str, object]]) -> None:
    """Move each clip into ``wavs/<global_speaker>/`` and update its row.

    Filenames already carry the video id, so clips pooled from different videos
    cannot collide. A row whose file is missing -- an older flat dataset, or a
    move this already did on a previous run -- is left alone rather than
    failing the report.
    """
    moved = 0
    touched: set[Path] = set()
    for row in rows:
        old = str(row.get("audio_file") or "")
        if not old:
            continue
        # A clip whose speaker did not clear the duration floor is not deleted
        # -- it is real audio with a real transcript, just of an identity too
        # thin to name. It goes to one drawer so the numbered folders stay
        # clean and it can still be recovered.
        speaker = row.get("global_speaker") or _UNASSIGNED
        new = f"{speaker}/{Path(old).name}"
        if new == old:
            continue
        source = ws.wavs / old
        if not source.exists():
            continue
        target = ws.wavs / new
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        row["audio_file"] = new
        touched.add(source.parent)
        moved += 1

    for directory in touched:
        # The per-video folders are empty once their clips have gone.
        if directory != ws.wavs and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    if moved:
        print(f"\nregrouped {moved} clip(s) into per-speaker folders under {ws.wavs}")


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
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
