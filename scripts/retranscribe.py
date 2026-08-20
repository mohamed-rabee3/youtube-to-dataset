"""Re-transcribe an existing dataset's clips with a different ASR backend.

The pipeline transcribes while it builds a dataset. This re-runs only the ASR
and everything downstream of it -- transcript, canonical-text choice, CER and
the text gates -- over clips that are already cut, so switching backend does
not mean re-downloading, re-separating and re-diarizing hours of video.

    scripts/retranscribe.py dataset-02 dataset_elshmesy2

What it touches:

* ``metadata.jsonl`` is rewritten in place, with the previous file preserved
  as ``metadata.<backend>.jsonl`` first. Every row is kept -- the gates are
  re-run but only *recorded*, in ``gate_status``, rather than dropping rows.
* Rows whose ``audio_file`` is no longer on disk cannot be transcribed. They
  move to ``orphans.jsonl`` and leave the dataset.
* ``retranscribe_report.json`` gets the before/after summary, and
  ``retranscribe_diff.jsonl`` the per-clip old vs new text with the CER
  between them, so the swap can be judged without re-reading the audio.
* ``rejected.jsonl`` is not touched. Clips the original run threw away were
  never written to disk, so there is nothing left to re-transcribe.

Transcripts are cached in ``.work/retranscribe_cache.jsonl`` as they arrive.
A re-run reads the cache first, so an interrupted job resumes without paying
for the same clip twice -- which matters, because a cloud backend bills per
request and these are billed per 15-second increment however short the clip.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yt2ds.arabic import cer, clean_for_output  # noqa: E402
from yt2ds.config import Config  # noqa: E402
from yt2ds.models import ModelRegistry  # noqa: E402
from yt2ds.stages import asr, filters  # noqa: E402
from yt2ds.stages.segment import Chunk  # noqa: E402

log = logging.getLogger("retranscribe")

# How many clips are handed to the backend at once. Concurrency *within* a
# batch is the backend's own (asr.google.max_workers); this only bounds how
# much audio is resident and how often the cache is flushed.
BATCH = 64

# Scores carried from the old row into the re-run, because the gates read them
# and nothing here recomputes them: they describe the audio, not the text.
CARRIED_SCORES = ("align_score", "boundary_clipped")

# Scores the re-run replaces.
REFRESHED_SCORES = (
    "asr_confidence",
    "asr_language",
    "asr_confidence_alt",
    "cer_yt_vs_cohere",
    "arabic_ratio",
    "script_ratio",
    "chars_per_sec",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-transcribe an existing dataset's clips with a different ASR backend.",
    )
    parser.add_argument("datasets", nargs="+", type=Path, help="dataset directories")
    parser.add_argument("--config", type=Path, help="YAML config (defaults to configs/default.yaml)")
    parser.add_argument("--backend", default="google", choices=["google", "cohere"])
    parser.add_argument("--google-asr-model", help="Speech-to-Text model (default: from config)")
    parser.add_argument("--google-credentials", type=Path, help="service-account JSON")
    parser.add_argument("--languages", help="comma-separated ASR languages (default: from config)")
    parser.add_argument("--limit", type=int, help="only process the first N clips; for a costed trial run")
    parser.add_argument("--dry-run", action="store_true", help="report what would be sent, transcribe nothing")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("grpc", "google.auth", "google.api_core", "urllib3", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    overrides: dict[str, object] = {"asr.backend": args.backend}
    if args.google_asr_model:
        overrides["asr.google.model"] = args.google_asr_model
    if args.google_credentials:
        overrides["asr.google.credentials_file"] = str(args.google_credentials)
    if args.languages:
        overrides["asr.languages"] = [x.strip() for x in args.languages.split(",") if x.strip()]
    cfg = Config.load(args.config, overrides)

    missing = [d for d in args.datasets if not (d / "metadata.jsonl").exists()]
    if missing:
        raise SystemExit(f"no metadata.jsonl in: {', '.join(str(d) for d in missing)}")

    registry = ModelRegistry(cfg) if not args.dry_run else None
    for dataset in args.datasets:
        _process(dataset, cfg, registry, args)
    return 0


def _process(dataset: Path, cfg: Config, registry, args) -> None:
    backend = cfg.asr.backend
    rows = [json.loads(line) for line in (dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    live, orphans = [], []
    for row in rows:
        (live if (dataset / "wavs" / row["audio_file"]).exists() else orphans).append(row)

    seconds = sum(float(r.get("duration") or 0) for r in live)
    billable = sum(max(15.0, math.ceil(float(r.get("duration") or 0) / 15) * 15) for r in live)
    log.info(
        "%s: %d clips to transcribe (%d orphaned), %.2f h audio -> %.1f h billable at 15s increments",
        dataset, len(live), len(orphans), seconds / 3600, billable / 3600,
    )
    if args.dry_run:
        return

    cache_path = dataset / ".work" / "retranscribe_cache.jsonl"
    cache = _load_cache(cache_path)
    if cache:
        log.info("%s: %d clips already cached from an earlier run", dataset, len(cache))

    todo = [r for r in live if r["audio_file"] not in cache]
    if args.limit:
        # Bounds what is *sent*, never what is written: every live row still
        # reaches the new metadata, the un-transcribed ones unchanged.
        todo = todo[: args.limit]
        log.info("--limit %d: transcribing %d of %d clips", args.limit, len(todo), len(live))
    _transcribe(dataset, todo, cfg, registry, cache, cache_path)

    updated, diffs, statuses = _rebuild(dataset, live, cache, cfg)
    _write(dataset, backend, updated, orphans, diffs, statuses, cache)


def _transcribe(dataset: Path, todo: list[dict], cfg: Config, registry, cache: dict, cache_path: Path) -> None:
    """Send every clip through the backend, appending results to the cache as they land."""
    if not todo:
        log.info("%s: nothing to transcribe, cache covers every clip", dataset)
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done = 0
    with cache_path.open("a", encoding="utf-8") as fh:
        for batch in (todo[i : i + BATCH] for i in range(0, len(todo), BATCH)):
            items = []
            for row in batch:
                samples, sr = sf.read(dataset / "wavs" / row["audio_file"], dtype="float32", always_2d=False)
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                chunk = Chunk(index=0, start=0.0, end=float(row.get("duration") or len(samples) / sr), speaker="")
                items.append((chunk, samples, sr, row))

            # One sample rate per batch is what the backend signature takes;
            # dataset clips are uniform, but do not assume it across a batch.
            for rate in {sr for _, _, sr, _ in items}:
                group = [(c, s) for c, s, r, _ in items if r == rate]
                rows_for = [row for _, _, r, row in items if r == rate]
                texts = asr.transcribe_batch(group, rate, cfg, registry)
                for (chunk, _), row, text in zip(group, rows_for, texts):
                    entry = {
                        "audio_file": row["audio_file"],
                        "text": text,
                        "asr_confidence": chunk.scores.get("asr_confidence"),
                        "asr_language": chunk.scores.get("asr_language"),
                    }
                    cache[row["audio_file"]] = entry
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()

            done += len(batch)
            rate_per_s = done / max(time.time() - started, 1e-6)
            remaining = (len(todo) - done) / max(rate_per_s, 1e-6)
            log.info(
                "%s: %d/%d clips (%.1f clips/s, ~%.0f min left)",
                dataset, done, len(todo), rate_per_s, remaining / 60,
            )


def _rebuild(dataset: Path, live: list[dict], cache: dict, cfg: Config) -> tuple[list[dict], list[dict], dict]:
    """Apply the new transcripts, re-run the text gates, record what they decided."""
    updated, diffs = [], []
    statuses: dict[str, int] = {}

    for row in live:
        entry = cache.get(row["audio_file"])
        if entry is None:
            # Only reachable under --limit; leave the row exactly as it was.
            updated.append(row)
            continue

        previous_text = row.get("text") or ""
        previous_asr = row.get("text_cohere") or ""
        previous_source = row.get("text_source") or ""

        chunk = Chunk(
            index=int(row.get("chunk_index") or 0),
            start=0.0,
            end=float(row.get("duration") or 0.0),
            speaker=row.get("speaker") or "",
        )
        chunk.text_yt = row.get("text_yt") or ""
        chunk.text_cohere = entry["text"]
        for key in CARRIED_SCORES:
            if row.get(key) is not None:
                chunk.scores[key] = row[key]
        chunk.scores["asr_confidence"] = entry["asr_confidence"]
        chunk.scores["asr_language"] = entry["asr_language"]

        # subtitle_kind is what the row already recorded, so a clip whose text
        # came from manual subs is not silently relabelled as auto.
        subtitle_kind = previous_source if previous_source.startswith("yt") else None
        filters.resolve_text([chunk], subtitle_kind, cfg)

        new = dict(row)
        new["text"] = chunk.text
        new["text_cohere"] = chunk.text_cohere
        new["text_source"] = chunk.text_source
        new["language"] = chunk.scores.get("asr_language") or row.get("language")
        # asr_confidence_alt goes to None on google: there is no runner-up
        # language, unlike Cohere's one-decode-per-language pass.
        for key in REFRESHED_SCORES:
            new[key] = chunk.scores.get(key)
        new["gate_status"] = chunk.reject_reason or "kept"

        category = (chunk.reject_reason or "kept").split(":", 1)[0]
        statuses[category] = statuses.get(category, 0) + 1
        updated.append(new)

        if clean_for_output(previous_asr) != clean_for_output(chunk.text_cohere):
            diffs.append(
                {
                    "audio_file": row["audio_file"],
                    "duration": row.get("duration"),
                    "text_yt": row.get("text_yt"),
                    "asr_before": previous_asr,
                    "asr_after": chunk.text_cohere,
                    "cer_before_vs_after": round(cer(previous_asr, chunk.text_cohere), 5) if previous_asr else None,
                    "text_before": previous_text,
                    "text_after": chunk.text,
                    "source_before": previous_source,
                    "source_after": chunk.text_source,
                    "gate_status": new["gate_status"],
                }
            )

    return updated, diffs, statuses


def _write(dataset: Path, backend: str, updated: list[dict], orphans: list[dict], diffs: list[dict], statuses: dict, cache: dict) -> None:
    metadata = dataset / "metadata.jsonl"

    # Preserve the transcripts being replaced before anything overwrites them:
    # they cost GPU hours (or API spend) and are not reproducible for free.
    backup = dataset / "metadata.pre-retranscribe.jsonl"
    if not backup.exists():
        shutil.copy2(metadata, backup)
        log.info("%s: previous metadata preserved at %s", dataset, backup.name)

    _write_jsonl(metadata, updated)
    if orphans:
        _write_jsonl(dataset / "orphans.jsonl", orphans)
    if diffs:
        _write_jsonl(dataset / "retranscribe_diff.jsonl", diffs)

    empty = sum(1 for r in updated if not (r.get("text_cohere") or "").strip())
    changed = len(diffs)
    report = {
        "backend": backend,
        "rows_before": len(updated) + len(orphans),
        "rows_after": len(updated),
        "orphans_removed": len(orphans),
        "clips_transcribed": len(cache),
        "transcripts_changed": changed,
        "transcripts_identical": len(updated) - changed,
        "empty_transcripts": empty,
        "gate_status": statuses,
        "hours": round(sum(float(r.get("duration") or 0) for r in updated) / 3600, 3),
    }
    (dataset / "retranscribe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info(
        "%s: %d rows written (%d orphans removed), %d transcripts changed, gates: %s",
        dataset, len(updated), len(orphans), changed, statuses,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # A run killed mid-write leaves a partial last line; the clip is
            # simply re-sent.
            continue
        cache[entry["audio_file"]] = entry
    return cache


if __name__ == "__main__":
    raise SystemExit(main())
