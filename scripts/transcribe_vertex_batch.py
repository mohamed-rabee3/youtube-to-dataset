"""Transcribe *and* diacritize a dataset's clips through Vertex AI batch prediction.

The sibling ``transcribe_gemini.py`` talks to the AI Studio endpoint with an API
key, which bills against a prepaid balance. When that balance empties every
request returns 429 and the run stalls with no way to buy its way out mid-flight.
Vertex speaks to the same models through the project's own billing, so it does
not have a separate balance to exhaust, and batch prediction is roughly half the
price of the online path for work that nobody is waiting on.

The trade is latency and shape. Batch reads its requests from Cloud Storage and
answers as a job, so audio must be uploaded first and results collected after.
That is why this runs in four resumable phases -- ``upload``, ``submit``,
``poll``, ``merge`` -- each safe to re-enter after an interruption.

Two things are deliberately shared with the online path rather than reimplemented:

* the prompt, response schema and per-clip cache entry format, so a corpus can
  be transcribed partly by one path and partly by the other and still merge;
* ``_write``, which applies the text gates and rewrites ``metadata.jsonl``.

**Clip identity travels in the object name.** A clip is uploaded to
``<prefix>/clips/<audio_file>.flac``, so the transcript that comes back is
matched to its clip by reversing that path. Batch echoes each request beside its
response, so the mapping survives even if the shard files are lost -- and unlike
an index-based mapping there is no ordering assumption that could silently
attach every transcript to the wrong clip.

Thinking is left at the model's default. ``thinkingLevel: low`` is about 25%
cheaper and measurably worse: it marks case endings the prompt forbids and
mis-places the article's vowel, and a wrong vowel in a TTS corpus teaches a
wrong pronunciation rather than merely adding noise.

    scripts/transcribe_vertex_batch.py dataset-socrates --dialect najdi \
        --bucket my-bucket --project my-project
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcribe_gemini import (  # noqa: E402
    RESPONSE_SCHEMA,
    SEND_SR,
    _entries,
    _load_cache,
    _write,
    build_prompt,
)

from yt2ds.config import Config  # noqa: E402
from yt2ds.io import read_jsonl  # noqa: E402

log = logging.getLogger("transcribe_vertex_batch")

# gemini-3.7-flash is published only in "global"; us-central1 answers 404 for
# both the online call and a batch job that names it.
LOCATION = "global"
HOST = "https://aiplatform.googleapis.com"
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


# ---------------------------------------------------------------- clip <-> URI
def clip_blob(prefix: str, audio_file: str) -> str:
    """``SPK1/ep_0067.wav`` -> ``<prefix>/clips/SPK1/ep_0067.flac``."""
    return f"{prefix}/clips/{audio_file.rsplit('.', 1)[0]}.flac"


def blob_clip(prefix: str, blob: str) -> str:
    """Inverse of :func:`clip_blob`, for reading a transcript back to its clip."""
    stem = blob[len(f"{prefix}/clips/"):]
    return f"{stem.rsplit('.', 1)[0]}.wav"


def flac_bytes(path: Path) -> bytes:
    """One clip as mono 16 kHz FLAC -- lossless, and about a third of the WAV."""
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != SEND_SR:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=SEND_SR, res_type="soxr_hq")
    buffer = io.BytesIO()
    sf.write(buffer, samples, SEND_SR, format="FLAC")
    return buffer.getvalue()


# ---------------------------------------------------------------------- phases
def outstanding(dataset: Path) -> list[dict]:
    """Rows whose clip has no successful transcript yet, in metadata order."""
    cache = _load_cache(dataset / ".work" / "gemini_transcribe_cache.jsonl")
    rows = [r for r in read_jsonl(dataset / "metadata.jsonl") if r.get("audio_file")]
    return [r for r in rows if cache.get(r["audio_file"], {}).get("status") != "ok"]


def phase_upload(dataset: Path, rows: list[dict], bucket, prefix: str, workers: int) -> None:
    """Upload each outstanding clip as FLAC, skipping what is already there.

    Listing the bucket once is far cheaper than an existence check per clip:
    at this scale that is one paged listing against ~168k round trips.
    """
    log.info("listing existing uploads under %s/clips/", prefix)
    present = {b.name for b in bucket.list_blobs(prefix=f"{prefix}/clips/")}
    todo = [r for r in rows if clip_blob(prefix, r["audio_file"]) not in present]
    log.info("%d clip(s) already uploaded, %d to go", len(rows) - len(todo), len(todo))
    if not todo:
        return

    done = 0
    started = time.time()

    def send(row: dict) -> None:
        name = clip_blob(prefix, row["audio_file"])
        blob = bucket.blob(name)
        blob.upload_from_string(flac_bytes(dataset / "wavs" / row["audio_file"]), content_type="audio/flac")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(send, r): r for r in todo}
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - one clip must not stop the upload
                log.error("upload failed for %s: %s", row["audio_file"], exc)
            done += 1
            if done % 500 == 0 or done == len(todo):
                rate = done / max(time.time() - started, 1e-6)
                left = (len(todo) - done) / max(rate, 1e-6) / 60
                log.info("uploaded %d/%d (%.1f clips/s, ~%.0f min left)", done, len(todo), rate, left)


def phase_submit(
    dataset: Path, rows: list[dict], bucket, prefix: str, args, session, project: str
) -> list[dict]:
    """Shard the outstanding clips into batch jobs and submit them."""
    prompt = build_prompt(args.dialect)
    requests_ = []
    for start in range(0, len(rows), args.clips_per_request):
        batch = rows[start : start + args.clips_per_request]
        parts = [
            {
                "fileData": {
                    "mimeType": "audio/flac",
                    "fileUri": f"gs://{bucket.name}/{clip_blob(prefix, r['audio_file'])}",
                }
            }
            for r in batch
        ]
        parts.append({"text": prompt.format(count=len(batch))})
        requests_.append(
            {
                "request": {
                    "contents": [{"role": "user", "parts": parts}],
                    # Deterministic, so a resumed shard cannot drift from the
                    # one it replaces.
                    "generationConfig": {
                        "temperature": 0.0,
                        "candidateCount": 1,
                        "responseMimeType": "application/json",
                        "responseSchema": RESPONSE_SCHEMA,
                    },
                }
            }
        )

    log.info("%d clip(s) -> %d request(s)", len(rows), len(requests_))
    jobs = []
    for index, start in enumerate(range(0, len(requests_), args.requests_per_job), start=1):
        shard = requests_[start : start + args.requests_per_job]
        name = f"{prefix}/shards/shard-{index:04d}.jsonl"
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in shard)
        bucket.blob(name).upload_from_string(body, content_type="application/jsonl")

        payload = {
            "displayName": f"{dataset.name}-vertex-{index:04d}",
            "model": f"publishers/google/models/{args.model}",
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": [f"gs://{bucket.name}/{name}"]},
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": f"gs://{bucket.name}/{prefix}/out/shard-{index:04d}/"},
            },
        }
        url = f"{HOST}/v1/projects/{project}/locations/{LOCATION}/batchPredictionJobs"
        response = session.post(url, json=payload, timeout=120)
        response.raise_for_status()
        job = response.json()
        log.info("shard %d: %d request(s) -> %s", index, len(shard), job["name"])
        jobs.append({"shard": index, "name": job["name"], "requests": len(shard)})
    return jobs


def phase_poll(jobs: list[dict], session, interval: float) -> list[dict]:
    """Block until every job reaches a terminal state."""
    pending = {j["name"] for j in jobs if j.get("state") not in TERMINAL}
    while pending:
        for job in jobs:
            if job["name"] not in pending:
                continue
            response = session.get(f"{HOST}/v1/{job['name']}", timeout=120)
            response.raise_for_status()
            body = response.json()
            job["state"] = body.get("state")
            job["output"] = (body.get("outputInfo") or {}).get("gcsOutputDirectory")
            if job["state"] in TERMINAL:
                pending.discard(job["name"])
                log.info("shard %d: %s", job["shard"], job["state"])
                if job["state"] != "JOB_STATE_SUCCEEDED":
                    log.error("shard %d failed: %s", job["shard"], json.dumps(body.get("error"))[:300])
        if pending:
            log.info("%d/%d shard(s) still running", len(pending), len(jobs))
            time.sleep(interval)
    return jobs


def phase_merge(dataset: Path, jobs: list[dict], bucket, prefix: str, client) -> int:
    """Read every prediction back into the shared per-clip cache.

    Each prediction echoes its request, so the clips it covers are recovered
    from the ``fileUri`` list rather than from any stored ordering.
    """
    cache_path = dataset / ".work" / "gemini_transcribe_cache.jsonl"
    written = 0
    with cache_path.open("a", encoding="utf-8") as fh:
        for job in jobs:
            output = job.get("output")
            if job.get("state") != "JOB_STATE_SUCCEEDED" or not output:
                continue
            blob_prefix = output.split(f"gs://{bucket.name}/", 1)[1]
            for blob in client.list_blobs(bucket, prefix=blob_prefix):
                if not blob.name.endswith(".jsonl"):
                    continue
                for line in blob.download_as_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    for entry in _prediction_entries(json.loads(line), prefix):
                        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        written += 1
    log.info("merged %d cache entr(ies)", written)
    return written


def _prediction_entries(record: dict, prefix: str) -> list[dict]:
    """One prediction line -> one cache entry per clip it covered."""
    request = record.get("request") or {}
    clips = [
        blob_clip(prefix, part["fileData"]["fileUri"].split("/", 3)[3])
        for content in request.get("contents", [])
        for part in content.get("parts", [])
        if part.get("fileData")
    ]
    batch = [{"audio_file": c} for c in clips]
    if not batch:
        return []

    if record.get("status"):
        return _entries(batch, None, f"batch:{str(record['status'])[:60]}")

    response = record.get("response") or {}
    candidates = response.get("candidates") or []
    if not candidates:
        return _entries(batch, None, "no_response")
    try:
        raw = "".join(p.get("text", "") for p in candidates[0]["content"]["parts"])
        texts = {int(item["i"]): item.get("text") or "" for item in json.loads(raw)}
    except (KeyError, IndexError, ValueError, TypeError):
        return _entries(batch, None, "unparseable")
    return _entries(batch, texts, None)


def _widen_pool(session, size: int) -> None:
    """Match the session's connection pool to the number of upload threads."""
    import requests as requests_lib

    adapter = requests_lib.adapters.HTTPAdapter(
        pool_connections=max(size, 10), pool_maxsize=max(size, 10)
    )
    session.mount("https://", adapter)


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset", type=Path)
    p.add_argument("--bucket", required=True, help="GCS bucket for audio, shards and predictions")
    p.add_argument("--project", help="GCP project (default: the client's own)")
    p.add_argument("--prefix", default="vertex-asr", help="object prefix inside the bucket")
    p.add_argument("--model", default="gemini-3.7-flash")
    p.add_argument("--dialect", help="najdi | hijazi | gulf | saudi | egyptian | levantine")
    p.add_argument("--clips-per-request", type=int, default=4)
    p.add_argument("--requests-per-job", type=int, default=5000)
    p.add_argument("--upload-workers", type=int, default=32)
    p.add_argument("--poll-interval", type=float, default=60.0)
    p.add_argument(
        "--limit",
        type=int,
        help="only handle the first N outstanding clips -- for a cheap end-to-end "
        "rehearsal of upload/submit/poll/merge before committing the whole corpus",
    )
    p.add_argument(
        "--phases",
        default="upload,submit,poll,merge",
        help="comma-separated subset to run; each is resumable on its own",
    )
    p.add_argument("--config", type=Path)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "google.auth", "google.api_core", "grpc", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    phases = {p.strip() for p in args.phases.split(",") if p.strip()}
    dataset = args.dataset
    state_path = dataset / ".work" / "vertex_batch_jobs.json"

    import google.auth
    import google.auth.transport.requests
    import requests as requests_lib
    from google.cloud import storage

    credentials, default_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project = args.project or default_project
    if not project:
        raise SystemExit("no project: pass --project or set one in your ADC")

    # AuthorizedSession re-mints the access token as it expires. A plain
    # Session with a token pasted into the header would 401 partway through:
    # tokens last about an hour and the upload alone runs longer than that,
    # never mind waiting on jobs.
    session = google.auth.transport.requests.AuthorizedSession(credentials)
    session.headers["x-goog-user-project"] = project

    client = storage.Client(project=project, credentials=credentials)
    # The storage client's session pools ten connections by default. With more
    # upload threads than that they spend their time discarding and reopening
    # sockets instead of sending bytes, which costs roughly half the throughput.
    _widen_pool(client._http, args.upload_workers)
    bucket = client.bucket(args.bucket)

    rows = outstanding(dataset)
    log.info("%s: %d clip(s) still need a transcript", dataset, len(rows))
    if args.limit:
        rows = rows[: args.limit]
        log.info("--limit %d: handling %d clip(s) this run", args.limit, len(rows))
    if not rows and phases & {"upload", "submit"}:
        log.info("nothing outstanding; run --phases merge if predictions are waiting")

    jobs: list[dict] = []
    if state_path.exists():
        jobs = json.loads(state_path.read_text(encoding="utf-8"))

    if "upload" in phases and rows:
        phase_upload(dataset, rows, bucket, args.prefix, args.upload_workers)

    if "submit" in phases and rows:
        jobs = phase_submit(dataset, rows, bucket, args.prefix, args, session, project)
        state_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    if "poll" in phases and jobs:
        jobs = phase_poll(jobs, session, args.poll_interval)
        state_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    if "merge" in phases and jobs:
        phase_merge(dataset, jobs, bucket, args.prefix, client)
        cfg = Config.load(args.config)
        cache = _load_cache(dataset / ".work" / "gemini_transcribe_cache.jsonl")
        metadata_rows = list(read_jsonl(dataset / "metadata.jsonl"))
        _write(dataset, metadata_rows, cache, cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
