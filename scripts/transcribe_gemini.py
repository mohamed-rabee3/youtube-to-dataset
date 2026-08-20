"""Transcribe *and* diacritize a dataset's clips in one Gemini pass.

    scripts/transcribe_gemini.py dataset-socrates --dialect najdi

Why one pass. The two-pass alternative -- Speech-to-Text for the words, then a
second model for the marks -- pays for the same audio twice and, more to the
point, pays to repeat its instructions once per clip. Measured on this corpus
the prompt overhead of a per-clip pass ($46) exceeds the cost of sending every
second of audio ($28). Batching many clips into one request collapses that
overhead to about a dollar, and once the batching exists there is no reason to
run two models: one call can return the transcript already marked.

What is lost, and what replaces it. Without a second independent transcript
there is nothing to run ``same_words`` against, so a fluent but wrong sentence
cannot be caught by disagreement. The guards that remain are the audio-side
gates from the GPU pass, and the text gates in ``yt2ds.stages.filters`` --
chars-per-second in particular, which catches a transcript grossly too long or
too short for its audio. That is weaker than a cross-check and is a deliberate
trade.

Two rules are enforced in code rather than trusted to the model
(see ``yt2ds.arabic``):

* no mark on any word's final letter -- case endings are grammatical and
  inaudible in ordinary speech, so a TTS model must not learn to pronounce
  them. ``strip_final_tashkeel`` also removes the tanwin-fath-plus-alef
  spelling of the same ending;
* no tatweel -- a model asked for marks will sometimes insert one as somewhere
  to hang a vowel, and it would otherwise pose as a word's final letter and
  shield a real case ending from the rule above.

How far batching can go, and why not further. Sending many clips in one request
is what makes the prompt overhead disappear, but the model does not reliably
keep N audio parts apart: past a handful it starts answering clip N with clip
M's words, returning the right *count* and the right *indices*, so nothing in
the response looks wrong. Measured against an independent Speech-to-Text
transcript of the same clips: 4 per request agreed on 24/24 (median CER 0.02),
12 per request on 16/23, and 60 per request lost 45% to the text gates. The
default is therefore 4, which still removes three quarters of the overhead.
Raising it is only safe with a reference transcript to check alignment against
-- misalignment is silent otherwise, and silently wrong text is worse for a TTS
dataset than missing text.

Requests are batched by *duration*, not clip count: clips run 2-12 s, so a
fixed count would build a request four times larger for a batch of long clips
than for short ones, and the inline-audio limit is on bytes. Audio goes up as
FLAC, which is lossless and about half the bytes of the equivalent WAV.

Results are cached per clip in ``.work/gemini_transcribe_cache.jsonl`` as they
arrive, so an interrupted run resumes without re-paying for a clip.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yt2ds.arabic import (  # noqa: E402
    clean_for_output,
    strip_final_tashkeel,
    tashkeel_ratio,
)
from yt2ds.config import Config  # noqa: E402
from yt2ds.stages import filters  # noqa: E402

log = logging.getLogger("transcribe_gemini")

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# 16 kHz mono is plenty for the model and a quarter the bytes of the 24 kHz
# master. Audio is billed by duration, so this only buys request size.
SEND_SR = 16000

DIALECTS = {
    "najdi": "النجدية (وسط السعودية، الرياض والقصيم)",
    "hijazi": "الحجازية (غرب السعودية، جدة ومكة)",
    "gulf": "الخليجية",
    "saudi": "السعودية",
    "egyptian": "المصرية",
    "levantine": "الشامية",
}

# A prior, never an override. Interview speech code-switches constantly toward
# MSA and a guest need not share the host's dialect, so a model told flatly
# "this is Najdi" would impose Najdi vowels on Modern Standard pronunciation.
DIALECT_CLAUSE = """اللهجة المتوقعة لدى المتحدثين هي {name}. استعن بها عند الالتباس أو عند عدم وضوح الحركة.
   والنطق المسموع هو الحكم النهائي دائمًا: إذا نطق المتحدث بالفصحى أو بلهجة أخرى، فاتبع ما سمعته ولا تفرض اللهجة المتوقعة."""

GENERIC_CLAUSE = "اتبع لهجة المتحدث كما نطقها في الصوت."

PROMPT = """ستستمع إلى {count} مقطعًا صوتيًا منفصلًا، مرقّمة من 1 إلى {count} بالترتيب.

المطلوب لكل مقطع: اكتب ما قيل فيه نصًّا عربيًّا مُشكَّلًا (بالحركات).

القواعد:
1. اكتب ما سمعته فقط. لا تخمّن ولا تكمل ولا تصحّح كلام المتحدث، وإذا لم تسمع كلامًا مفهومًا فاترك النص فارغًا.
2. التشكيل يجب أن يطابق النطق الفعلي في الصوت، وليس قواعد الفصحى المعيارية.
   {dialect_rule}
3. لا تضع أي حركة على الحرف الأخير من أي كلمة (وقف، بدون إعراب).
4. اكتب الإملاء العربي الصحيح: الهمزات (أ، إ، ئ، ؤ) والتاء المربوطة (ة) في مواضعها.
5. لا تكتب أي شرح أو تعليق أو وصف للصوت، ولا علامات مثل [موسيقى].
6. كل مقطع مستقل تمامًا عن غيره. لا تنقل كلامًا من مقطع إلى آخر.

أعد النتيجة على هيئة JSON: مصفوفة من كائنات، لكل كائن حقل i (رقم المقطع) وحقل text (النص المُشكَّل)."""

RETRYABLE = {429, 500, 502, 503, 504}

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {"i": {"type": "INTEGER"}, "text": {"type": "STRING"}},
        "required": ["i", "text"],
    },
}


def build_prompt(dialect: str | None) -> str:
    """Fill the dialect clause in, leaving ``{count}`` for each batch."""
    clause = GENERIC_CLAUSE
    if dialect:
        clause = DIALECT_CLAUSE.format(name=DIALECTS.get(dialect.lower(), dialect))
    return PROMPT.replace("{dialect_rule}", clause)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("datasets", nargs="+", type=Path)
    p.add_argument("--model", default="gemini-3.7-flash")
    p.add_argument(
        "--dialect",
        help=f"expected dialect, a prior for ambiguous vowels only. Named: {', '.join(sorted(DIALECTS))}",
    )
    p.add_argument(
        "--batch-seconds",
        type=float,
        default=20.0,
        help="audio seconds per request (default 20). Batching by duration rather than clip "
        "count keeps request bytes even, since clips run 2-12 s",
    )
    p.add_argument(
        "--max-batch",
        type=int,
        default=4,
        help="clips per request (default 4). Raising this is NOT free: the model stops "
        "reliably telling one audio part from another and silently returns clip N's words "
        "for clip M. Measured on this corpus: 4 clips agreed with an independent transcript "
        "on 24/24, 12 on 16/23, 60 was unusable. Do not raise it without a reference "
        "transcript to check alignment against",
    )
    p.add_argument("--workers", type=int, default=8, help="requests in flight")
    p.add_argument("--limit", type=int, help="only send the first N clips; for a costed trial")
    p.add_argument("--config", type=Path)
    p.add_argument("--dry-run", action="store_true", help="report what would be sent, and stop")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("GEMINI_API_KEY is not set")

    cfg = Config.load(args.config)
    for dataset in args.datasets:
        _process(dataset, api_key or "", cfg, args)
    return 0


def _process(dataset: Path, api_key: str, cfg: Config, args) -> None:
    metadata = dataset / "metadata.jsonl"
    if not metadata.exists():
        log.error("%s: no metadata.jsonl", dataset)
        return

    rows = [json.loads(l) for l in metadata.read_text(encoding="utf-8").splitlines() if l.strip()]
    live = [r for r in rows if (dataset / "wavs" / r["audio_file"]).is_file()]

    cache_path = dataset / ".work" / "gemini_transcribe_cache.jsonl"
    cache = _load_cache(cache_path)
    todo = [r for r in live if cache.get(r["audio_file"], {}).get("status") != "ok"]
    if args.limit:
        todo = todo[: args.limit]

    seconds = sum(float(r.get("duration") or 0) for r in todo)
    batches = _batches(todo, args)
    log.info(
        "%s: %d/%d clips to transcribe, %.2f h audio, %d request(s)",
        dataset, len(todo), len(live), seconds / 3600, len(batches),
    )
    # Audio dominates; the prompt is sent once per request rather than per clip.
    audio_tokens = seconds * 32 / 1e6
    log.info("%s: ~%.1fM audio tokens -> ~$%.2f input at $1.00/M", dataset, audio_tokens, audio_tokens)
    if args.dry_run:
        return

    if batches:
        _run(dataset, batches, api_key, cache, cache_path, args)
    _write(dataset, rows, cache, cfg)


def _batches(todo: list[dict], args) -> list[list[dict]]:
    """Group clips into requests bounded by total audio duration.

    Byte size, not clip count, is what the inline-audio limit constrains, and
    clips here vary by 6x in length.
    """
    out: list[list[dict]] = []
    current: list[dict] = []
    total = 0.0
    for row in todo:
        duration = float(row.get("duration") or 0.0)
        if current and (total + duration > args.batch_seconds or len(current) >= args.max_batch):
            out.append(current)
            current, total = [], 0.0
        current.append(row)
        total += duration
    if current:
        out.append(current)
    return out


def _run(dataset: Path, batches: list[list[dict]], api_key: str, cache: dict, cache_path: Path, args) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("https://", adapter)
    url = ENDPOINT.format(model=args.model)
    prompt = build_prompt(args.dialect)
    started = time.time()
    done = clips = 0

    with cache_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:

        def job(batch: list[dict]) -> list[dict]:
            audio = [_flac_bytes(dataset / "wavs" / r["audio_file"]) for r in batch]
            texts, error = _call(session, url, api_key, audio, prompt, args)
            return _entries(batch, texts, error)

        for entries in pool.map(job, batches):
            for entry in entries:
                cache[entry["audio_file"]] = entry
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                clips += 1
            done += 1
            if done % 5 == 0 or done == len(batches):
                fh.flush()
                rate = clips / max(time.time() - started, 1e-6)
                remaining = (len(batches) - done) / max(done / max(time.time() - started, 1e-6), 1e-9)
                log.info(
                    "%s: %d/%d requests, %d clips (%.1f clips/s, ~%.0f min left)",
                    dataset, done, len(batches), clips, rate, remaining / 60,
                )


def _entries(batch: list[dict], texts: dict[int, str] | None, error: str | None) -> list[dict]:
    """Turn one response into one cache entry per clip in the batch.

    A response whose indices do not line up with the batch is discarded whole
    rather than guessed at: a shifted list would attach every transcript to the
    wrong clip, which is the one failure mode that would corrupt the dataset
    silently instead of loudly.
    """
    if error or texts is None:
        return [{"audio_file": r["audio_file"], "text": None, "status": error or "no_response"} for r in batch]
    if set(texts) - set(range(1, len(batch) + 1)):
        return [{"audio_file": r["audio_file"], "text": None, "status": "index_mismatch"} for r in batch]

    entries = []
    for i, row in enumerate(batch, start=1):
        raw = texts.get(i)
        if raw is None:
            entries.append({"audio_file": row["audio_file"], "text": None, "status": "missing_index"})
            continue
        # clean_for_output strips tatweel; strip_final_tashkeel clears the
        # case ending. Both are enforced here, not asked of the model.
        text = strip_final_tashkeel(clean_for_output(raw))
        if not text:
            entries.append({"audio_file": row["audio_file"], "text": "", "status": "ok"})
            continue
        entries.append(
            {
                "audio_file": row["audio_file"],
                "text": text,
                "tashkeel_ratio": round(tashkeel_ratio(text), 4),
                "status": "ok",
            }
        )
    return entries


def _call(session, url, api_key, audio: list[bytes], prompt: str, args) -> tuple[dict[int, str] | None, str | None]:
    parts: list[dict] = [
        {"inline_data": {"mime_type": "audio/flac", "data": base64.b64encode(a).decode("ascii")}} for a in audio
    ]
    parts.append({"text": prompt.format(count=len(audio))})
    payload = {
        "contents": [{"parts": parts}],
        # Deterministic: the same batch must not drift between resumed runs.
        "generationConfig": {
            "temperature": 0.0,
            "candidateCount": 1,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }

    delay = 2.0
    for attempt in range(1, args.max_retries + 1):
        try:
            response = session.post(url, params={"key": api_key}, json=payload, timeout=args.timeout)
        except requests.RequestException as exc:
            if attempt == args.max_retries:
                return None, f"network:{type(exc).__name__}"
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            continue

        if response.status_code == 200:
            try:
                body = response.json()["candidates"][0]["content"]["parts"]
                raw = "".join(p.get("text", "") for p in body)
                return {int(item["i"]): item.get("text") or "" for item in json.loads(raw)}, None
            except (KeyError, IndexError, ValueError, TypeError):
                return None, "unparseable"

        if response.status_code in RETRYABLE and attempt < args.max_retries:
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            continue
        return None, f"http:{response.status_code}"

    return None, "exhausted"


def _flac_bytes(path: Path) -> bytes:
    """One clip as mono 16 kHz FLAC -- lossless, and half the bytes of WAV."""
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != SEND_SR:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=SEND_SR, res_type="soxr_hq")
    buffer = io.BytesIO()
    sf.write(buffer, samples, SEND_SR, format="FLAC")
    return buffer.getvalue()


def _write(dataset: Path, rows: list[dict], cache: dict, cfg: Config) -> None:
    """Merge transcripts into the metadata, then run the text gates."""
    metadata = dataset / "metadata.jsonl"
    backup = dataset / "metadata.pre-gemini.jsonl"
    if not backup.exists():
        shutil.copy2(metadata, backup)
        log.info("%s: previous metadata preserved at %s", dataset, backup.name)

    statuses: dict[str, int] = {}
    for row in rows:
        entry = cache.get(row["audio_file"])
        if entry is None:
            statuses["not_sent"] = statuses.get("not_sent", 0) + 1
            continue
        statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
        if entry["status"] != "ok":
            continue
        row["text"] = entry["text"]
        row["text_cohere"] = entry["text"]
        row["text_source"] = "gemini" if entry["text"] else "pending"
        row["tashkeel_ratio"] = entry.get("tashkeel_ratio")

    kept, rejected = filters.gate_rows(rows, cfg)
    _rewrite(metadata, kept)

    existing = [
        json.loads(l)
        for l in (dataset / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ] if (dataset / "rejected.jsonl").exists() else []
    existing = [r for r in existing if not str(r.get("reject_reason") or "").startswith("text:")]
    _rewrite(dataset / "rejected.jsonl", existing + rejected)

    removed = 0
    for row in rejected:
        target = dataset / "wavs" / str(row.get("audio_file") or "")
        if target.is_file():
            target.unlink()
            removed += 1

    marked = sum(1 for r in kept if tashkeel_ratio(r.get("text") or "") > 0)
    log.info("%s: statuses %s", dataset, statuses)
    log.info("%s: %d kept (%d with tashkeel), %d cut on text gates, %d wav(s) deleted",
             dataset, len(kept), marked, len(rejected), removed)


def _rewrite(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
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
            continue
        cache[entry["audio_file"]] = entry
    return cache


if __name__ == "__main__":
    raise SystemExit(main())
