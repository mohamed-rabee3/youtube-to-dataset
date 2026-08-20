"""Flag clips that are not Saudi/Gulf Arabic, for review before removal.

    scripts/flag_dialect.py dataset-02 dataset_elshmesy2
    # review dataset-02/dialect_flagged.jsonl, delete any false positives
    scripts/flag_dialect.py dataset-02 --apply

This script never deletes anything on its own. The first run only writes
``dialect_flagged.jsonl``; ``--apply`` later moves exactly the rows still
listed in that file out of ``metadata.jsonl`` and into ``removed_dialect.jsonl``.
So the review loop is: run, open the flag file, delete the lines that were
wrongly flagged, then apply.

**Why the audio and not just the text.** Dialect shows in pronunciation at
least as much as in word choice, and these clips average under four seconds --
often too few words to be distinctive. ``مو زين`` is Gulf and ``مش كويس``
is Egyptian, but plenty of utterances are lexically neutral and separable only
by how they sound. So each clip is classified from its audio, with the
transcript supplied as context.

**Why a per-speaker consensus on top.** A single short clip is a noisy sample.
Dialect, though, is a property of the speaker, so the per-clip verdicts are
pooled by ``(video_id, speaker)`` and a clip whose own call disagrees with a
confident majority for its speaker is scored on the majority instead. That
recovers the neutral-sounding clips inside an obviously Egyptian speaker's
turns, and suppresses one-off misfires inside a Saudi speaker's.

**Lexical markers** are computed too, but only as a cross-check shown in the
output -- they are far too coarse to flag on alone, since ``ايش`` and ``وين``
are shared across several dialects.

MSA is not flagged by default: it is dialect-neutral rather than foreign, and
a Saudi speaker reading formally is still a Saudi voice. ``--flag-msa``
includes it.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yt2ds.arabic import normalize  # noqa: E402

log = logging.getLogger("flag_dialect")

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SEND_SR = 16000
RETRYABLE = {429, 500, 502, 503, 504}

#: The label set the model must choose from. Kept coarse on purpose -- finer
#: distinctions (Hijazi vs Najdi) are not reliable from four seconds of audio.
LABELS = (
    "saudi",      # incl. Najdi, Hijazi, Qassimi, southern
    "gulf",       # Kuwaiti, Emirati, Bahraini, Qatari -- near-neighbours
    "egyptian",
    "levantine",  # Syrian, Lebanese, Palestinian, Jordanian
    "iraqi",
    "yemeni",
    "sudanese",
    "maghrebi",
    "msa",
    "unclear",
)

#: Verdicts that count as in-scope for a Saudi dataset.
KEEP = {"saudi", "gulf", "msa", "unclear"}
KEEP_WITHOUT_MSA = {"saudi", "gulf", "unclear"}

PROMPT = """استمع إلى المقطع الصوتي وحدد لهجة المتحدث.

النص المكتوب للمقطع (للمساعدة فقط): {text}

اعتمد على النطق واللكنة أولًا، ثم على المفردات.

اختر تصنيفًا واحدًا فقط من هذه القائمة:
saudi, gulf, egyptian, levantine, iraqi, yemeni, sudanese, maghrebi, msa, unclear

ملاحظات:
- saudi تشمل النجدية والحجازية والقصيمية والجنوبية.
- gulf تشمل الكويتية والإماراتية والبحرينية والقطرية.
- msa تعني فصحى معيارية بدون لهجة محلية واضحة.
- unclear إذا كان المقطع قصيرًا جدًا أو غير واضح للحكم.

أجب بصيغة JSON فقط، بدون أي نص آخر:
{{"dialect": "...", "confidence": 0.0}}

confidence رقم بين 0 و 1 يعبر عن مدى ثقتك."""

# Deliberately small and high-signal. Shared words (ايش، وين، عشان) are left
# out precisely because they do not separate the dialects.
MARKERS: dict[str, tuple[str, ...]] = {
    "egyptian": (
        "دلوقتي", "ازاي", "إزاي", "ازيك", "عايز", "عاوز", "بتاع", "بتاعة",
        "مش", "ده", "دي", "دول", "كده", "ليه", "بقى", "خالص", "معلش",
        "حاجة", "الجدعان", "اهو", "ماشي", "بص", "يبقى",
    ),
    "levantine": (
        "هلق", "هلأ", "شو", "هيك", "منيح", "منيحة", "كتير", "بدي", "بدك",
        "لسا", "هون", "كيفك", "عنجد", "منشان", "تنين", "بتحكي",
    ),
    "saudi": (
        "وش", "ابغى", "أبغى", "زين", "ترى", "يمديك", "مو", "مب", "تبي",
        "تبين", "شلون", "عاد", "كذا", "قد", "حيل", "يالله", "طيب",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="only classify the first N clips")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--flag-msa", action="store_true", help="also flag clips judged Modern Standard Arabic")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="do not flag a clip the model was less sure than this about (default 0.6)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="remove the rows still listed in dialect_flagged.jsonl from metadata.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    for dataset in args.datasets:
        if not (dataset / "metadata.jsonl").exists():
            raise SystemExit(f"no metadata.jsonl in {dataset}")

    if args.apply:
        for dataset in args.datasets:
            _apply(dataset)
        return 0

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("GEMINI_API_KEY is not set")
    for dataset in args.datasets:
        _process(dataset, api_key, args)
    return 0


# -- classification -----------------------------------------------------
def _process(dataset: Path, api_key: str, args) -> None:
    rows = _read(dataset / "metadata.jsonl")
    live = [r for r in rows if (r.get("text") or "").strip() and (dataset / "wavs" / r["audio_file"]).exists()]
    log.info("%s: %d/%d rows classifiable", dataset, len(live), len(rows))
    if args.dry_run:
        return

    cache_path = dataset / ".work" / "dialect_cache.jsonl"
    cache = _load_cache(cache_path)
    if cache:
        log.info("%s: %d clips already cached", dataset, len(cache))

    todo = [r for r in live if cache.get(r["audio_file"], {}).get("dialect") is None]
    if args.limit:
        todo = todo[: args.limit]
        log.info("--limit %d: classifying %d of %d", args.limit, len(todo), len(live))
    if todo:
        _classify(dataset, todo, api_key, cache, cache_path, args)

    _flag(dataset, rows, cache, args)


def _classify(dataset: Path, todo: list[dict], api_key: str, cache: dict, cache_path: Path, args) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount(
        "https://",
        requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers),
    )
    url = ENDPOINT.format(model=args.model)
    started, done = time.time(), 0

    with cache_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        def job(row: dict) -> dict:
            audio = _wav_bytes(dataset / "wavs" / row["audio_file"])
            return _call(session, url, api_key, audio, row, args)

        for entry in pool.map(job, todo):
            cache[entry["audio_file"]] = entry
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0 or done == len(todo):
                fh.flush()
                rate = done / max(time.time() - started, 1e-6)
                log.info(
                    "%s: %d/%d (%.1f clips/s, ~%.0f min left)",
                    dataset, done, len(todo), rate, (len(todo) - done) / max(rate, 1e-6) / 60,
                )


def _call(session, url, api_key, audio: bytes, row: dict, args) -> dict:
    entry: dict = {"audio_file": row["audio_file"], "dialect": None, "confidence": None, "status": "ok"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(audio).decode("ascii")}},
                    {"text": PROMPT.format(text=(row.get("text") or "")[:400])},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.0, "candidateCount": 1, "responseMimeType": "application/json"},
    }

    delay = 2.0
    for attempt in range(1, args.max_retries + 1):
        try:
            response = session.post(url, params={"key": api_key}, json=payload, timeout=args.timeout)
        except requests.RequestException as exc:
            if attempt == args.max_retries:
                entry["status"] = f"network:{type(exc).__name__}"
                return entry
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            continue

        if response.status_code == 200:
            try:
                parts = response.json()["candidates"][0]["content"]["parts"]
                data = json.loads("".join(p.get("text", "") for p in parts))
                label = str(data.get("dialect", "")).strip().lower()
                entry["dialect"] = label if label in LABELS else "unclear"
                entry["confidence"] = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
            except (KeyError, IndexError, ValueError, TypeError):
                entry["status"] = "unparsable"
            return entry
        if response.status_code in RETRYABLE and attempt < args.max_retries:
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            continue
        entry["status"] = f"http:{response.status_code}"
        return entry
    entry["status"] = "exhausted"
    return entry


# -- flagging -----------------------------------------------------------
def _flag(dataset: Path, rows: list[dict], cache: dict, args) -> None:
    keep = KEEP_WITHOUT_MSA if args.flag_msa else KEEP

    # Pool per-clip verdicts by speaker: dialect belongs to the person, so a
    # confident majority is better evidence than one four-second sample.
    votes: dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        entry = cache.get(row["audio_file"])
        if entry and entry.get("dialect") and entry["dialect"] != "unclear":
            votes[(row.get("video_id"), row.get("speaker"))][entry["dialect"]] += 1

    consensus: dict[tuple, tuple[str, float]] = {}
    for group, counter in votes.items():
        label, count = counter.most_common(1)[0]
        consensus[group] = (label, count / sum(counter.values()))

    flagged, per_dialect = [], Counter()
    for row in rows:
        entry = cache.get(row["audio_file"])
        if not entry or not entry.get("dialect"):
            continue
        group = (row.get("video_id"), row.get("speaker"))
        speaker_label, speaker_share = consensus.get(group, (None, 0.0))
        clip_label = entry["dialect"]
        clip_conf = entry.get("confidence") or 0.0

        # A clip is judged on its speaker when that speaker has a clear
        # majority over several clips; otherwise on its own call.
        group_size = sum(votes[group].values()) if group in votes else 0
        use_speaker = group_size >= 3 and speaker_share >= 0.67
        label = speaker_label if use_speaker else clip_label
        confidence = speaker_share if use_speaker else clip_conf
        basis = "speaker_consensus" if use_speaker else "clip"

        if label in keep or confidence < args.min_confidence:
            continue

        markers = _markers(row.get("text") or "")
        per_dialect[label] += 1
        flagged.append(
            {
                **row,
                "flag_dialect": label,
                "flag_confidence": round(confidence, 3),
                "flag_basis": basis,
                "flag_clip_dialect": clip_label,
                "flag_clip_confidence": round(clip_conf, 3),
                "flag_speaker_dialect": speaker_label,
                "flag_speaker_votes": dict(votes.get(group, {})),
                "flag_lexical_markers": markers,
            }
        )

    # Most confident first: the top of the file should be the obvious ones, so
    # a review can stop reading once the calls get marginal.
    flagged.sort(key=lambda r: (-r["flag_confidence"], r["flag_dialect"]))
    out = dataset / "dialect_flagged.jsonl"
    _write(out, flagged)

    statuses = Counter(e.get("status", "?") for e in cache.values())
    report = {
        "model": args.model,
        "rows": len(rows),
        "classified": sum(1 for e in cache.values() if e.get("dialect")),
        "flagged": len(flagged),
        "flagged_by_dialect": dict(per_dialect),
        "all_verdicts": dict(Counter(e["dialect"] for e in cache.values() if e.get("dialect"))),
        "kept_labels": sorted(keep),
        "min_confidence": args.min_confidence,
        "api_status": dict(statuses),
        "speakers": len(consensus),
        "speakers_non_saudi": sum(1 for label, _ in consensus.values() if label not in keep),
    }
    (dataset / "dialect_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("%s: flagged %d/%d rows -> %s", dataset, len(flagged), len(rows), out.name)
    log.info("%s: by dialect %s", dataset, dict(per_dialect))


def _markers(text: str) -> dict[str, list[str]]:
    """Dialect marker words present, as a cross-check for the reviewer."""
    words = set(normalize(text).split())
    hits = {}
    for dialect, markers in MARKERS.items():
        found = sorted(w for w in {normalize(m) for m in markers} if w in words)
        if found:
            hits[dialect] = found
    return hits


# -- removal ------------------------------------------------------------
def _apply(dataset: Path) -> None:
    """Remove exactly the rows still listed in dialect_flagged.jsonl."""
    flag_path = dataset / "dialect_flagged.jsonl"
    if not flag_path.exists():
        raise SystemExit(f"{flag_path} not found -- run without --apply first")

    doomed = {r["audio_file"] for r in _read(flag_path)}
    rows = _read(dataset / "metadata.jsonl")
    keep = [r for r in rows if r["audio_file"] not in doomed]
    removed = [r for r in rows if r["audio_file"] in doomed]

    if not removed:
        log.info("%s: nothing to remove", dataset)
        return

    backup = dataset / "metadata.pre-dialect.jsonl"
    if not backup.exists():
        shutil.copy2(dataset / "metadata.jsonl", backup)
        log.info("%s: previous metadata preserved at %s", dataset, backup.name)

    _write(dataset / "metadata.jsonl", keep)
    _write(dataset / "removed_dialect.jsonl", removed)
    # The wavs are deliberately left on disk: undoing a removal should not
    # require rebuilding audio.
    log.info(
        "%s: removed %d rows, %d remain (wavs left in place, rows kept in removed_dialect.jsonl)",
        dataset, len(removed), len(keep),
    )


# -- io -----------------------------------------------------------------
def _wav_bytes(path: Path) -> bytes:
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != SEND_SR:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=SEND_SR, res_type="soxr_hq")
    buffer = io.BytesIO()
    sf.write(buffer, samples, SEND_SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        cache[entry["audio_file"]] = entry
    return cache


if __name__ == "__main__":
    raise SystemExit(main())
