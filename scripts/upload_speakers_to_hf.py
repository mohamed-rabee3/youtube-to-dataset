#!/usr/bin/env python
"""Push a yt2ds dataset to the Hub with one folder per global speaker.

    export HF_TOKEN=hf_...
    scripts/upload_speakers_to_hf.py dataset-socrates-1/ --repo Rabe3/socrates-youtube-006 --private

Unlike scripts/upload_to_hf.py, which flattens the whole dataset into one
``data/train-*.parquet`` series, this reads ``speakers/<GLOBAL_SPEAKER_NN>/``
-- the folders speaker linking writes -- and keeps that split on the Hub as
``data/<GLOBAL_SPEAKER_NN>/train-*.parquet``. The README it writes declares one
config per speaker, so ``load_dataset(repo, "GLOBAL_SPEAKER_07")`` pulls a
single voice without downloading the rest, plus an ``all`` config that globs
every folder for the whole thing.

Audio is re-encoded to FLAC before being embedded under a ``datasets.Audio``
feature: bit-identical to the source PCM_16 WAVs at a bit over half the bytes,
which halves a multi-hour push.

Each shard is built, uploaded, then deleted before the next one starts, so the
extra disk this needs is one shard rather than a second copy of the dataset.

Shards are committed in batches rather than one at a time: the Hub caps a
repo at 256 commits an hour, and a file-per-commit push of this dataset wants
387 of them, so it stalls for an hour partway through. A batch is staged on
disk, committed as one operation, then deleted, which puts the ceiling on extra
disk at ``--commit-gb`` rather than one shard.

Resumable: shards already in the repo are skipped, so an interrupted run is
restarted by re-running the same command. The shard plan comes from each
speaker's metadata.jsonl, so that only holds while those files are not growing
-- do not run this against a dataset a pipeline is still writing to.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Features, Value
from huggingface_hub import CommitOperationAdd, HfApi

# Every column yt2ds writes, typed explicitly. Inferring the schema from the
# rows does not work: the fields that come from alignment are null on videos
# that skipped it, and a shard where a column is null throughout would type
# itself differently from its neighbours and break the loader.
FEATURES = Features(
    {
        "audio": Audio(sampling_rate=24000),
        "audio_file": Value("string"),
        # Where the clip sits in the dataset's own wavs/ tree, i.e. which
        # per-episode speaker it came from before linking pooled it here.
        "source_audio_file": Value("string"),
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
        "vocals_isolated": Value("bool"),
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
        # Written by scripts/diacritize.py; null on rows it could not diacritize.
        "tashkeel_ratio": Value("float64"),
        "tashkeel_source": Value("string"),
    }
)
SCHEMA = FEATURES.arrow_schema
COLUMNS = [name for name in SCHEMA.names if name != "audio"]

README = """---
license: cc-by-nc-4.0
language:
- ar
task_categories:
- text-to-speech
- automatic-speech-recognition
pretty_name: {pretty}
size_categories:
- 100K<n<1M
configs:
{configs}---

# {pretty}

Arabic (Saudi) speech from the {source} podcast, segmented and transcribed by
[yt2ds](https://github.com/mohamed-rabee3/youtube-to-dataset). {clips:,} clips /
{hours:,.1f} hours across {speakers} speakers, 24 kHz mono, FLAC in parquet.

Speaker linking pools each voice across every episode it appears in, and each
speaker keeps its own folder under `data/`, so a voice can be pulled on its own:

```python
from datasets import load_dataset

one = load_dataset("{repo}", "GLOBAL_SPEAKER_00", split="train")  # the host
everything = load_dataset("{repo}", "all", split="train")
```

`GLOBAL_SPEAKER_00` is the host and by far the largest voice; the numbering runs
roughly from most to least speech. Each folder also carries `speaker.json` with
that speaker's duration, clip count, episode count, mean MOS, and the
per-episode diarization labels that were merged into it.

## Columns

`audio` (24 kHz mono), `text` (diacritized Arabic), and the per-clip quality
measurements the pipeline gates on -- `squim_mos`, `squim_stoi`, `squim_pesq`,
`snr_db`, `music_score`, `speech_score`, `lufs`, `clipping_ratio` -- plus the
provenance of each clip: `video_id`, `title`, `start`, `end`, `speaker` (the
per-episode diarization label) and `global_speaker` (the linked identity).

## Speakers

| speaker | clips | hours |
|---|---:|---:|
{table}
"""


def read_rows(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def plan_shards(speaker: Path, target_bytes: int):
    """Assign every row to a shard up front, so shard names can carry the total."""
    wavs = speaker / "wavs"
    shards: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    missing: list[str] = []

    for row in read_rows(speaker / "metadata.jsonl"):
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


def to_flac(path: Path) -> bytes:
    """Lossless: the sources are uniformly PCM_16, so int16 in, PCM_16 out."""
    data, rate = sf.read(str(path), dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def build_shard(shard: list[dict], wavs: Path, path: Path, flac: bool) -> None:
    audio = []
    columns: dict[str, list] = {name: [] for name in COLUMNS}

    for row in shard:
        name = row["audio_file"]
        source = wavs / name
        if flac:
            audio.append({"bytes": to_flac(source), "path": f"{Path(name).stem}.flac"})
        else:
            audio.append({"bytes": source.read_bytes(), "path": name})
        alt = row.get("asr_confidence_alt") or {}
        for column in COLUMNS:
            if column == "asr_confidence_alt":
                columns[column].append({"ar": alt.get("ar"), "en": alt.get("en")})
            else:
                columns[column].append(row.get(column))

    table = pa.Table.from_pydict({"audio": audio, **columns}, schema=SCHEMA)
    pq.write_table(table, path, compression="zstd", compression_level=3)


def commit(api: HfApi, repo: str, operations: list, message: str, attempts: int = 5) -> None:
    """One commit for many files. Backoff is in minutes: the failure this meets
    is the Hub's 256-commits-an-hour cap, which seconds of sleep cannot outwait."""
    delays = [60, 300, 900, 1800, 3600]
    for attempt in range(attempts):
        try:
            api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                operations=operations,
                commit_message=message,
            )
            return
        except Exception as exc:  # a multi-hour push meets flaky networks
            delay = delays[min(attempt, len(delays) - 1)]
            print(f"  attempt {attempt + 1}/{attempts} failed, retrying in {delay}s: {exc}", flush=True)
            time.sleep(delay)
    raise SystemExit(f"giving up on {message}")


class Batch:
    """Collects files until one of the caps trips, then commits them together."""

    def __init__(self, api: HfApi, repo: str, max_files: int, max_bytes: int):
        self.api = api
        self.repo = repo
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.operations: list[CommitOperationAdd] = []
        self.temporary: list[Path] = []
        self.bytes = 0

    def add(self, local: Path, remote: str, temporary: bool = False) -> None:
        self.operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
        if temporary:
            self.temporary.append(local)
        self.bytes += local.stat().st_size
        if len(self.operations) >= self.max_files or self.bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.operations:
            return
        names = [op.path_in_repo for op in self.operations]
        started = time.time()
        message = f"Add {len(names)} files ({Path(names[0]).parent.name} ...)"
        commit(self.api, self.repo, self.operations, message)
        print(
            f"  committed {len(names)} files, {self.bytes / 1e6:.0f} MB in {time.time() - started:.0f}s",
            flush=True,
        )
        for path in self.temporary:
            path.unlink(missing_ok=True)
        self.operations, self.temporary, self.bytes = [], [], 0


def write_readme(api: HfApi, repo: str, dataset: Path, stats: list[tuple[str, int, float]], work_dir: Path) -> None:
    configs = ['- config_name: all\n  default: true\n  data_files:\n  - split: train\n    path: data/*/train-*.parquet\n']
    for name, _, _ in stats:
        configs.append(f"- config_name: {name}\n  data_files:\n  - split: train\n    path: data/{name}/train-*.parquet\n")
    table = "\n".join(f"| `{name}` | {clips:,} | {hours:,.1f} |" for name, clips, hours in stats)
    pretty = dataset.name.replace("dataset-", "").replace("-", " ").title()
    text = README.format(
        pretty=pretty,
        source=pretty,
        repo=repo,
        configs="".join(configs),
        clips=sum(c for _, c, _ in stats),
        hours=sum(h for _, _, h in stats),
        speakers=len(stats),
        table=table,
    )
    local = work_dir / "README.md"
    local.write_text(text, encoding="utf-8")
    commit(api, repo, [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(local))], "Add README")
    local.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="yt2ds output directory, the one holding speakers/")
    parser.add_argument("--repo", required=True, help="target repo, e.g. Rabe3/socrates-youtube-006")
    parser.add_argument("--only", nargs="*", help="push just these speaker folders")
    parser.add_argument(
        "--shard-mb",
        type=int,
        default=850,
        help="target shard size in source WAV bytes; FLAC lands at a bit over half this",
    )
    parser.add_argument("--wav", action="store_true", help="embed the WAVs untouched instead of re-encoding to FLAC")
    parser.add_argument(
        "--commit-files",
        type=int,
        default=24,
        help="most files to put in one commit; the Hub allows 256 commits an hour",
    )
    parser.add_argument(
        "--commit-gb",
        type=float,
        default=3.0,
        help="most staged bytes to hold before committing; this is the extra disk the run needs",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/yt2ds-speaker-shards"))
    parser.add_argument("--private", action="store_true", help="create the repo private if it does not exist")
    parser.add_argument("--no-readme", action="store_true", help="skip the README/config push at the end")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN")

    speakers = sorted(d for d in (args.dataset / "speakers").iterdir() if (d / "metadata.jsonl").exists())
    if args.only:
        wanted = set(args.only)
        speakers = [d for d in speakers if d.name in wanted]
    if not speakers:
        raise SystemExit("no speaker folders found")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    done = set(api.list_repo_files(args.repo, repo_type="dataset"))

    plans = []
    stats = []
    for speaker in speakers:
        shards, missing = plan_shards(speaker, args.shard_mb * 1024 * 1024)
        plans.append((speaker, shards))
        rows = [row for shard in shards for row in shard]
        stats.append((speaker.name, len(rows), sum(row["duration"] for row in rows) / 3600))
        if missing:
            print(f"{speaker.name}: {len(missing)} missing wavs, first {missing[:3]}", flush=True)

    total_shards = sum(len(s) for _, s in plans)
    print(
        f"{len(plans)} speakers, {sum(c for _, c, _ in stats):,} rows, "
        f"{sum(h for _, _, h in stats):,.1f} hours -> {total_shards} shards",
        flush=True,
    )

    index = 0
    started_all = time.time()
    batch = Batch(api, args.repo, args.commit_files, int(args.commit_gb * 1e9))
    for speaker, shards in plans:
        remote_json = f"data/{speaker.name}/speaker.json"
        if (speaker / "speaker.json").exists() and remote_json not in done:
            batch.add(speaker / "speaker.json", remote_json)

        for position, shard in enumerate(shards):
            index += 1
            remote = f"data/{speaker.name}/train-{position:05d}-of-{len(shards):05d}.parquet"
            if remote in done:
                print(f"[{index}/{total_shards}] {remote} already on the hub", flush=True)
                continue

            local = args.work_dir / f"{speaker.name}-{position:05d}.parquet"
            started = time.time()
            build_shard(shard, speaker / "wavs", local, flac=not args.wav)
            size_mb = local.stat().st_size / 1e6
            print(
                f"[{index}/{total_shards}] {remote} {size_mb:.0f} MB {len(shard)} rows "
                f"(build {time.time() - started:.0f}s, elapsed {(time.time() - started_all) / 60:.0f}m)",
                flush=True,
            )
            batch.add(local, remote, temporary=True)

    manifest = args.dataset / "manifest.json"
    if manifest.exists() and "manifest.json" not in done:
        batch.add(manifest, "manifest.json")
    batch.flush()
    if not args.no_readme:
        write_readme(api, args.repo, args.dataset, stats, args.work_dir)
    print("done", flush=True)


if __name__ == "__main__":
    main()
