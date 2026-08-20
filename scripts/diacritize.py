"""Add tashkeel to an existing dataset's transcripts, from the audio.

    scripts/diacritize.py dataset-02 dataset_elshmesy2

Why this is a separate pass and not an ASR backend: Cloud Speech-to-Text
cannot emit diacritics at all -- there is no flag for it, and it returns none.
Neither can any of the local ASR models here. So the marks have to come from a
model that *hears* the clip, which is what makes them match the speaker's
actual dialectal pronunciation rather than Modern Standard grammar. A text-only
diacritizer (Farasa, Mishkal, CATT) infers marks from spelling and is
MSA-trained, so on Saudi or Egyptian speech it produces textbook endings that
contradict the audio.

The words are *not* re-transcribed. Each clip already has a transcript that was
gated against YouTube's captions when the dataset was built, so this pass sends
the audio plus that text and asks only for marks to be added. Two rules are
then enforced in code rather than trusted to the model
(see ``yt2ds.arabic``):

* the letters must not change -- if the bare skeleton moved, the model
  rewrote the line instead of diacritizing it, and the row is left alone;
* no mark on any word's final letter, because Arabic case endings are
  grammatical and inaudible in ordinary speech.

Output, per the dataset schema: ``text`` becomes the diacritized string and
``text_cohere`` keeps the undiacritized transcript it was built from, so
nothing is lost and ``to_voxcpm.py`` picks up the marks with no change.

Results are cached in ``.work/diacritize_cache.jsonl`` as they arrive, so an
interrupted run resumes without re-paying for a clip.
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
    same_skeleton,
    same_words,
    strip_final_tashkeel,
    strip_tashkeel,
    tashkeel_ratio,
)

log = logging.getLogger("diacritize")

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# 16 kHz mono is plenty for the model and a quarter the bytes of the 24 kHz
# master. Audio is billed by duration, so this only buys upload speed.
SEND_SR = 16000

# Named dialects, as the prompt should say them. `--dialect` picks one; the
# clause it produces is a *prior*, never an override -- see DIALECT_CLAUSE.
DIALECTS = {
    "najdi": "النجدية (وسط السعودية، الرياض والقصيم)",
    "hijazi": "الحجازية (غرب السعودية، جدة ومكة)",
    "gulf": "الخليجية",
    "saudi": "السعودية",
    "egyptian": "المصرية",
    "levantine": "الشامية",
}

# Naming the expected dialect helps only where the audio is ambiguous -- a
# short vowel that could be fatha or kasra, a word the speaker clips. Where it
# is not ambiguous the audio is the ground truth, and saying so matters:
# interview speech code-switches constantly toward MSA, and a guest need not
# share the host's dialect. A model told flatly "this is Najdi" would impose
# Najdi vowels on a sentence pronounced in Modern Standard, which is precisely
# the failure a text-only diacritizer has and the reason this pass listens.
DIALECT_CLAUSE = """اللهجة المتوقعة لدى المتحدثين هي {name}. استعن بها فقط عند الالتباس أو عند عدم وضوح الحركة في الصوت.
   والنطق المسموع هو الحكم النهائي دائمًا: إذا نطق المتحدث بالفصحى أو بلهجة أخرى، فاتبع ما سمعته ولا تفرض اللهجة المتوقعة."""

GENERIC_CLAUSE = "اتبع لهجة المتحدث كما نطقها في الصوت."

PROMPT = """أنت مُشكِّل نصوص عربية. ستستمع إلى مقطع صوتي، ومعه النص المكتوب لنفس المقطع.

المطلوب: أعد كتابة النص نفسه مع إضافة التشكيل.

القواعد:
1. التشكيل يجب أن يطابق النطق الفعلي في الصوت (لهجة المتحدث كما نطقها)، وليس قواعد الفصحى المعيارية.
   {dialect_rule}
2. لا تضع أي حركة على الحرف الأخير من أي كلمة (وقف، بدون إعراب).
3. لا تغيّر الحروف ولا الكلمات ولا ترتيبها ولا علامات الترقيم. أضف الحركات فقط.
4. لا تحذف كلمة ولا تضيف كلمة.
5. استثناء واحد فقط من القاعدة الثالثة، وهو تصحيح الإملاء في حالتين لا غير:
   - الهمزة: إذا كُتبت ألفًا مجردة وكان أصلها همزة، فصحّحها (اسئله ← أسئلة، ابراهيم ← إبراهيم).
   - التاء المربوطة: إذا انتهت الكلمة بهاء وكان أصلها تاءً مربوطة، فصحّحها (طاوله ← طاولة، سعاده ← سعادة).
   ولا تفعل ذلك إلا إذا كانت الكلمة تقتضيه فعلًا؛ الكلمات التي تنتهي بهاء أصلية (له، به، منه، عنده) تبقى كما هي.
   وما عدا هاتين الحالتين لا تغيّر أي حرف.
6. أعد النص المُشكَّل فقط، في سطر واحد، بدون أي شرح أو مقدمة أو علامات اقتباس.

النص:
{text}"""

# Sent only after the first attempt changed the letters. Naming the usual
# failure is what makes the difference: the model simplifies dialectal doubled
# consonants (اللي -> الي) or splits a word (من -> م ن).
CORRECTION_PROMPT = """أنت مُشكِّل نصوص عربية. ستستمع إلى مقطع صوتي، ومعه النص المكتوب لنفس المقطع.

في محاولة سابقة غيّرتَ حروف النص. هذا خطأ.

انسخ النص حرفًا بحرف تمامًا كما هو، ثم أضف الحركات فوق الحروف وتحتها فقط.

تحذيرات مهمة:
- لا تحذف أي حرف مكرر. مثال: "اللي" تبقى "اللي" ولا تصبح "الي". "يللا" تبقى "يللا".
- لا تفصل كلمة إلى كلمتين ولا تدمج كلمتين. مثال: "من" تبقى كلمة واحدة.
- عدد الكلمات في جوابك يجب أن يساوي عدد الكلمات في النص بالضبط.
- لا تضع أي حركة على الحرف الأخير من أي كلمة.
- التشكيل يطابق النطق في الصوت (اللهجة)، وليس الفصحى المعيارية.
  {dialect_rule}
- أعد النص المُشكَّل فقط، في سطر واحد، بدون شرح.

النص:
{text}"""


def build_prompts(dialect: str | None) -> tuple[str, str]:
    """Fill the dialect clause into both prompts, leaving ``{text}`` for later.

    ``replace`` rather than ``format`` so the ``{text}`` placeholder survives to
    be filled per clip.
    """
    clause = GENERIC_CLAUSE
    if dialect:
        clause = DIALECT_CLAUSE.format(name=DIALECTS.get(dialect.lower(), dialect))
    return PROMPT.replace("{dialect_rule}", clause), CORRECTION_PROMPT.replace("{dialect_rule}", clause)


RETRYABLE = {429, 500, 502, 503, 504}

#: Cache statuses that count as a usable result; anything else is re-sent.
OK_STATUSES = ("ok", "ok_orthography_fixed", "ok_on_retry", "ok_orthography_fixed_on_retry")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument(
        "--source",
        choices=["auto", "text", "text_cohere"],
        default="auto",
        help="which transcript to diacritize. auto = text_cohere when present "
        "(best orthography; YouTube captions drop ta marbuta and hamza), else text",
    )
    parser.add_argument(
        "--dialect",
        help="expected dialect, used only to resolve ambiguous vowels -- the audio always wins. "
        f"Named: {', '.join(sorted(DIALECTS))}. Any other string is passed through as written. "
        "Omit for no dialect hint (the previous behaviour)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only send the first N clips; for a trial run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("GEMINI_API_KEY is not set")

    for dataset in args.datasets:
        if not (dataset / "metadata.jsonl").exists():
            raise SystemExit(f"no metadata.jsonl in {dataset}")
    for dataset in args.datasets:
        _process(dataset, api_key, args)
    return 0


def _source_field(row: dict, mode: str) -> str:
    """Which column the text to diacritize comes from."""
    if mode != "auto":
        return mode
    # text_cohere first: YouTube's Arabic captions drop ta marbuta and hamza
    # seats, so diacritizing them would build marks on damaged spelling.
    return "text_cohere" if (row.get("text_cohere") or "").strip() else "text"


def _source_text(row: dict, mode: str) -> str:
    return clean_for_output(row.get(_source_field(row, mode)) or "")


def _process(dataset: Path, api_key: str, args) -> None:
    rows = [json.loads(l) for l in (dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    live = []
    for row in rows:
        text = _source_text(row, args.source)
        if text and (dataset / "wavs" / row["audio_file"]).exists():
            live.append(row)

    seconds = sum(float(r.get("duration") or 0) for r in live)
    log.info("%s: %d/%d rows diacritizable, %.2f h audio", dataset, len(live), len(rows), seconds / 3600)
    if args.dry_run:
        return

    cache_path = dataset / ".work" / "diacritize_cache.jsonl"
    cache = _load_cache(cache_path)
    if cache:
        log.info("%s: %d clips already cached", dataset, len(cache))

    todo = [r for r in live if cache.get(r["audio_file"], {}).get("status") not in OK_STATUSES]
    if args.limit:
        todo = todo[: args.limit]
        log.info("--limit %d: sending %d of %d clips", args.limit, len(todo), len(live))

    if todo:
        _run(dataset, todo, api_key, cache, cache_path, args)

    _write(dataset, rows, cache, args)


def _run(dataset: Path, todo: list[dict], api_key: str, cache: dict, cache_path: Path, args) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    # urllib3 pools 10 connections by default. With more workers than that the
    # surplus are opened, used and discarded on every call, which both spams
    # warnings and makes rate-limit rejections far more likely.
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("https://", adapter)
    url = ENDPOINT.format(model=args.model)
    prompt, correction_prompt = build_prompts(getattr(args, "dialect", None))
    started = time.time()
    done = 0

    with cache_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        def job(row: dict) -> dict:
            source, source_field = _source_text(row, args.source), _source_field(row, args.source)
            audio = _wav_bytes(dataset / "wavs" / row["audio_file"])
            marked, error = _call(session, url, api_key, audio, source, args, prompt)
            entry = _validate(row["audio_file"], source, marked, error)
            if entry["status"] == "words_changed":
                # Decoding is deterministic, so re-sending the same prompt
                # returns the same answer. Show the model what it broke
                # instead. In practice it drops a doubled lam (اللي -> الي) or
                # splits a word, and naming that is usually enough.
                marked, error = _call(session, url, api_key, audio, source, args, correction_prompt)
                retried = _validate(row["audio_file"], source, marked, error)
                if retried["status"] in OK_STATUSES:
                    retried["status"] += "_on_retry"
                    return retried
            entry["source_field"] = source_field
            return entry

        for entry in pool.map(job, todo):
            cache[entry["audio_file"]] = entry
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            done += 1
            if done % 25 == 0 or done == len(todo):
                fh.flush()
                rate = done / max(time.time() - started, 1e-6)
                log.info(
                    "%s: %d/%d (%.1f clips/s, ~%.0f min left)",
                    dataset, done, len(todo), rate, (len(todo) - done) / max(rate, 1e-6) / 60,
                )


def _validate(audio_file: str, source: str, marked: str | None, error: str | None) -> dict:
    """Accept the model's output only if it added marks and changed nothing else."""
    entry = {"audio_file": audio_file, "source": source, "text": None, "status": "ok"}
    if error:
        entry["status"] = error
        return entry
    marked = clean_for_output(marked or "")
    if not marked:
        entry["status"] = "empty"
        return entry
    exact = same_skeleton(source, marked)
    if not exact and not same_words(source, marked):
        # The model rewrote the line rather than marking it. Its tashkeel can
        # no longer be trusted against text nobody verified, so drop it.
        entry["status"] = "words_changed"
        entry["rejected_text"] = marked
        return entry
    if not exact:
        # Same words, different orthography: the model restored hamza seats or
        # a ta marbuta the transcript was missing. Keep it, but say so.
        entry["status"] = "ok_orthography_fixed"
    final = strip_final_tashkeel(marked)
    if not same_skeleton(marked, final):  # defensive: enforcement must be additive-only
        entry["status"] = "enforcement_broke_skeleton"
        return entry
    if tashkeel_ratio(final) == 0.0:
        entry["status"] = "no_marks"
        return entry
    entry["text"] = final
    return entry


def _call(session, url, api_key, audio: bytes, text: str, args, prompt: str) -> tuple[str | None, str | None]:
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(audio).decode("ascii")}},
                    {"text": prompt.format(text=text)},
                ]
            }
        ],
        # Deterministic: the same clip must not drift between resumed runs.
        "generationConfig": {"temperature": 0.0, "candidateCount": 1},
    }

    delay = 2.0
    for attempt in range(1, args.max_retries + 1):
        try:
            response = session.post(
                url, params={"key": api_key}, json=payload, timeout=args.timeout
            )
        except requests.RequestException as exc:
            if attempt == args.max_retries:
                return None, f"network:{type(exc).__name__}"
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            continue

        if response.status_code == 200:
            try:
                parts = response.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts).strip(), None
            except (KeyError, IndexError, ValueError):
                # A blocked or truncated candidate carries no parts.
                return None, "no_candidate"
        if response.status_code in RETRYABLE and attempt < args.max_retries:
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            continue
        return None, f"http:{response.status_code}"
    return None, "exhausted"


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


def _write(dataset: Path, rows: list[dict], cache: dict, args) -> None:
    metadata = dataset / "metadata.jsonl"
    backup = dataset / "metadata.pre-tashkeel.jsonl"
    if not backup.exists():
        shutil.copy2(metadata, backup)
        log.info("%s: previous metadata preserved at %s", dataset, backup.name)

    statuses: dict[str, int] = {}
    updated = []
    marked_rows = 0
    for row in rows:
        entry = cache.get(row["audio_file"])
        new = dict(row)
        if entry is None:
            statuses["not_sent"] = statuses.get("not_sent", 0) + 1
        else:
            statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
            if entry["status"] in OK_STATUSES and entry["text"]:
                # text carries the marks; text_cohere keeps the plain
                # transcript they were derived from.
                new["text"] = entry["text"]
                new["text_cohere"] = entry["source"]
                new["tashkeel_ratio"] = round(tashkeel_ratio(entry["text"]), 4)
                new["tashkeel_source"] = entry.get("source_field", args.source)
                # On a subtitle-sourced row, `text` was YouTube's caption and
                # is now a diacritized Cohere transcript. text_source names
                # where the canonical text came from, so it has to follow.
                if new["tashkeel_source"] == "text_cohere" and not str(new.get("text_source", "")).startswith("cohere"):
                    new["text_source"] = "cohere+tashkeel"
                elif not str(new.get("text_source", "")).endswith("+tashkeel"):
                    new["text_source"] = f"{new.get('text_source', 'asr')}+tashkeel"
                marked_rows += 1
        updated.append(new)

    tmp = metadata.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in updated:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(metadata)

    failures = [e for e in cache.values() if e["status"] not in OK_STATUSES]
    if failures:
        with (dataset / "diacritize_failures.jsonl").open("w", encoding="utf-8") as fh:
            for entry in failures:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    report = {
        "model": args.model,
        "source": args.source,
        "rows": len(rows),
        "rows_diacritized": marked_rows,
        "status": statuses,
        "mean_tashkeel_ratio": round(
            sum(tashkeel_ratio(r["text"]) for r in updated if r.get("tashkeel_ratio")) / max(marked_rows, 1), 4
        ),
    }
    (dataset / "diacritize_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("%s: %d/%d rows diacritized, statuses: %s", dataset, marked_rows, len(rows), statuses)


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
