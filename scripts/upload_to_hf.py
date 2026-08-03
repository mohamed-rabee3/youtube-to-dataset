#!/usr/bin/env python
"""Push a yt2ds dataset to the Hugging Face Hub as sharded parquet.

    export HF_TOKEN=hf_...
    scripts/upload_to_hf.py dataset/ --repo Rabe3/youtube-300
    scripts/upload_to_hf.py dataset/ --repo Rabe3/youtube-300 --rows 77304

Each shard is built, uploaded, then deleted before the next one starts, so the
extra disk this needs is one shard rather than a second copy of the dataset --
which matters, because a run that fills the disk is the normal failure mode here.

Audio is embedded in the parquet under a ``datasets.Audio`` feature, so the repo
loads with ``load_dataset(repo)`` and the Hub viewer works. The alternative --
77k loose WAVs in one folder -- uploads far slower and defeats the viewer.

Resumable: shards already present in the repo are skipped, so an interrupted run
is restarted by re-running the same command. That only holds if the shard plan
is unchanged, and the plan is derived from ``metadata.jsonl``, which grows while
a pipeline run is still going. Pass ``--rows`` with the count you started from to
pin the plan to that snapshot; without it, a grown metadata file reshards into a
different number of files and orphans everything already uploaded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Features, Value
from huggingface_hub import HfApi

# Every column yt2ds writes, typed explicitly. Inferring the schema from the
# rows does not work: `global_speaker` is null until speaker linking runs, and
# the fields that come from alignment are null on videos that skipped it.
FEATURES = Features(
    {
        "audio": Audio(sampling_rate=24000),
        "audio_file": Value("string"),
        "video_id": Value("string"),
        "video_url": Value("string"),
        "title": Value("string"),
        "channel": Value("string"),
        "upload_date": Value("string"),
        "chunk_index": Value("int32"),
        "start": Value("float64"),
        "end": Value("float64"),
        "duration": Value("float64"),
        "speaker": Value("string"),
        "global_speaker": Value("string"),
        "speaker_conf": Value("float64"),
        "text": Value("string"),
        "text_source": Value("string"),
        "text_yt": Value("string"),
        "text_cohere": Value("string"),
        "cer_yt_vs_cohere": Value("float64"),
        "asr_language": Value("string"),
        "asr_confidence": Value("float64"),
        # Flattened to a fixed struct: the raw dict is keyed by whichever
        # languages lost the decode, which is not a stable parquet schema.
        "asr_confidence_alt": {"ar": Value("float64"), "en": Value("float64")},
        "align_score": Value("float64"),
        "aligned_words": Value("int32"),
        "boundary_clipped": Value("bool"),
        "music_score": Value("float64"),
        "music_label": Value("string"),
        "speech_score": Value("float64"),
        "vocal_ratio": Value("float64"),
        "accompaniment_ratio": Value("float64"),
        "squim_mos": Value("float64"),
        "squim_stoi": Value("float64"),
        "squim_pesq": Value("float64"),
        "squim_si_sdr": Value("float64"),
        "snr_db": Value("float64"),
        "peak_dbfs": Value("float64"),
        "rms_dbfs": Value("float64"),
        "clipping_ratio": Value("float64"),
        "arabic_ratio": Value("float64"),
        "script_ratio": Value("float64"),
        "chars_per_sec": Value("float64"),
        "lufs": Value("float64"),
        "sample_rate": Value("int32"),
        "language": Value("string"),
    }
)
SCHEMA = FEATURES.arrow_schema
COLUMNS = [name for name in SCHEMA.names if name != "audio"]


def read_rows(root: Path, limit: int | None):
    with open(root / "metadata.jsonl", encoding="utf-8") as fh:
        for line in islice(fh, limit):
            yield json.loads(line)


def plan_shards(root: Path, limit: int | None, target_bytes: int):
    """Assign every row to a shard up front, so shard names can carry the total."""
    wavs = root / "wavs"
    shards: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    missing: list[str] = []

    for row in read_rows(root, limit):
        try:
            size = (wavs / row["audio_file"]).stat().st_size
        except FileNotFoundError:
            missing.append(row["audio_file"])
            continue
        current.append(row)
        current_bytes += size
        if current_bytes >= target_bytes:
            shards.append(current)
            current, current_bytes = [], 0

    if current:
        shards.append(current)
    return shards, missing


def build_shard(shard: list[dict], wavs: Path, path: Path) -> None:
    audio = []
    columns: dict[str, list] = {name: [] for name in COLUMNS}

    for row in shard:
        name = row["audio_file"]
        audio.append({"bytes": (wavs / name).read_bytes(), "path": name})
        alt = row.get("asr_confidence_alt") or {}
        for column in COLUMNS:
            if column == "asr_confidence_alt":
                columns[column].append({"ar": alt.get("ar"), "en": alt.get("en")})
            else:
                columns[column].append(row.get(column))

    table = pa.Table.from_pydict({"audio": audio, **columns}, schema=SCHEMA)
    pq.write_table(table, path, compression="zstd", compression_level=3)


def upload(api: HfApi, local: Path, remote: str, repo: str, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=repo,
                repo_type="dataset",
                commit_message=f"Add {remote}",
            )
            return
        except Exception as exc:  # a multi-hour push meets flaky networks
            print(f"  attempt {attempt + 1}/{attempts} failed: {exc}", flush=True)
            time.sleep(20 * (attempt + 1))
    raise SystemExit(f"giving up on {remote}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="yt2ds output directory")
    parser.add_argument("--repo", required=True, help="target repo, e.g. Rabe3/youtube-300")
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="only the first N rows of metadata.jsonl; pin this to keep a resumed run's shard plan stable",
    )
    parser.add_argument("--shard-mb", type=int, default=450, help="target shard size before compression")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/yt2ds-shards"))
    parser.add_argument("--private", action="store_true", help="create the repo private if it does not exist")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN")

    wavs = args.dataset / "wavs"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    shards, missing = plan_shards(args.dataset, args.rows, args.shard_mb * 1024 * 1024)
    total = len(shards)
    print(f"{sum(len(s) for s in shards)} rows -> {total} shards; {len(missing)} missing wavs", flush=True)
    if missing:
        print(f"  missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}", flush=True)

    done = set(api.list_repo_files(args.repo, repo_type="dataset"))
    for index, shard in enumerate(shards):
        remote = f"data/train-{index:05d}-of-{total:05d}.parquet"
        if remote in done:
            print(f"[{index + 1}/{total}] {remote} already on the hub", flush=True)
            continue

        local = args.work_dir / Path(remote).name
        started = time.time()
        build_shard(shard, wavs, local)
        size_mb = local.stat().st_size / 1e6
        upload(api, local, remote, args.repo)
        local.unlink()
        print(
            f"[{index + 1}/{total}] {remote} {size_mb:.0f} MB {len(shard)} rows in {time.time() - started:.0f}s",
            flush=True,
        )

    print("done", flush=True)


if __name__ == "__main__":
    main()
