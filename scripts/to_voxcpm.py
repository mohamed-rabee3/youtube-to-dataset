#!/usr/bin/env python
"""Convert a yt2ds dataset into the manifest format VoxCPM fine-tuning wants.

    scripts/to_voxcpm.py dataset/ --out voxcpm_data/
    scripts/to_voxcpm.py dataset/ --out voxcpm_data/ --push-to-hub Rabe3/saudi-clean-vox

VoxCPM's trainer reads a JSONL manifest -- one object per line, ``audio`` and
``text`` required, ``ref_audio`` / ``duration`` / ``dataset_id`` optional -- and
loads the audio through ``datasets.Audio(sampling_rate=...)``. So the work here
is: pick the right transcript, clean it, resample the WAVs to the rate the
AudioVAE encoder expects, and write the manifest.

Two choices worth knowing about:

*Transcript.* yt2ds stores YouTube's auto-caption as the canonical ``text``,
which is right for ASR provenance and wrong for TTS: YouTube's Arabic captions
drop ta marbuta, hamza seats and nearly all punctuation. A TTS model trained on
that spelling never learns the mapping from ordinary written Arabic. Cohere's
transcript keeps the real orthography, and the pipeline already gated every row
on the two transcripts agreeing, so it is the default here.

*Reference audio.* Each row gets a second clip of the same speaker as
``ref_audio``, which trains VoxCPM in its prompted / voice-cloning mode. Pass
``--no-ref-audio`` for plain text-to-speech fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------

# Bracketed caption annotations YouTube injects: [موسيقى], (تصفيق), [Music].
_ANNOTATION = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]|^>>+\s*", re.MULTILINE)
# Tatweel plus the invisible characters that survive a copy-paste chain:
# zero-width space/non-joiner/joiner, the bidi marks and isolates, soft hyphen,
# BOM. None of them are pronounceable; all of them cost tokens.
_INVISIBLE = re.compile(r"[ـ­​-‏‪-‮⁦-⁩﻿]")
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff☀-➿⬀-⯿️⃣]",
)
_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"[0-9٠-٩۰-۹]")
_ARABIC_LETTER = re.compile(r"[ء-ي]")


def clean_text(text: str) -> str:
    """Whitespace, caption artefacts and invisibles out; orthography untouched."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _ANNOTATION.sub(" ", text)
    text = _EMOJI.sub(" ", text)
    text = _INVISIBLE.sub("", text)
    text = _WS.sub(" ", text).strip()
    # Punctuation left stranded by a removed annotation.
    return text.strip(" -–—،,;:")


def pick_text(row: dict, mode: str) -> str:
    if mode == "cohere":
        return row.get("text_cohere") or row.get("text") or ""
    if mode == "youtube":
        return row.get("text_yt") or row.get("text") or ""
    return row.get("text") or ""


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------


def _trim_edges(x: np.ndarray, sr: int, floor_db: float, keep_s: float) -> np.ndarray:
    """Trim leading/trailing silence, leaving ``keep_s`` of it behind.

    VoxCPM's guide asks for under half a second of trailing silence. yt2ds cuts
    on VAD boundaries so this is usually a no-op, but a clip that ends on a
    breath still trains the model to emit one.
    """
    frame = max(int(0.02 * sr), 1)
    n = len(x) // frame
    if n < 3:
        return x
    frames = x[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    loud = np.flatnonzero(db > db.max() + floor_db)
    if loud.size == 0:
        return x
    keep = int(keep_s / 0.02)
    start = max(int(loud[0]) - keep, 0) * frame
    end = min(int(loud[-1]) + 1 + keep, n) * frame
    return x[start:end]


def convert_clip(job: tuple) -> tuple:
    """Read one source WAV, resample, and write it. Runs in a worker process."""
    src, dst, target_sr, floor_db, keep_s, peak_ceiling = job
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    x = x.mean(axis=1)

    x = _trim_edges(x, sr, floor_db, keep_s)

    if sr != target_sr:
        import librosa  # imported here so the parent process stays light

        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")

    # Resampling overshoots; without this a clip that peaked at -0.7 dBFS comes
    # out clipped.
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    ceiling = 10.0 ** (peak_ceiling / 20.0)
    if peak > ceiling:
        x = x * (ceiling / peak)

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), x, target_sr, subtype="PCM_16")
    return dst.name, round(len(x) / target_sr, 3)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def choose_rows(rows: list[dict], args) -> tuple[list[dict], dict]:
    """Apply the VoxCPM-side filters. Returns kept rows and a drop tally."""
    dropped: dict[str, int] = defaultdict(int)
    kept = []
    for row in rows:
        text = clean_text(pick_text(row, args.text))
        duration = float(row.get("duration") or 0.0)

        if not text or len(text.split()) < args.min_words:
            dropped["text_too_short"] += 1
            continue
        if not _ARABIC_LETTER.search(text) and args.text != "canonical":
            dropped["no_arabic_script"] += 1
            continue
        if duration < args.min_duration:
            dropped["too_short"] += 1
            continue
        if duration > args.max_duration:
            dropped["too_long"] += 1
            continue
        if args.drop_digits and _DIGIT.search(text):
            dropped["has_digits"] += 1
            continue

        kept.append({**row, "_text": text})
    return kept, dict(dropped)


def assign_ref_audio(kept: list[dict], rng: random.Random) -> None:
    """Give every row a ``_ref`` pointing at another clip of the same speaker.

    The pool is the speaker's cleanest mid-length clips rather than a random
    draw: ``ref_audio`` is what the model conditions the voice on, so a noisy
    or two-second reference teaches it to clone noise.
    """
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in kept:
        by_speaker[row.get("speaker") or row["video_id"]].append(row)

    for speaker, rows in by_speaker.items():
        pool = [r for r in rows if 3.0 <= float(r["duration"]) <= 10.0]
        if not pool:
            pool = list(rows)
        pool.sort(key=lambda r: float(r.get("squim_mos") or 0.0), reverse=True)
        pool = pool[:12]

        for i, row in enumerate(rows):
            candidates = [r for r in pool if r["audio_file"] != row["audio_file"]]
            if not candidates:
                row["_ref"] = None
                continue
            row["_ref"] = candidates[i % len(candidates)]


def split_train_val(kept: list[dict], fraction: float, rng: random.Random) -> tuple[list, list]:
    """Hold out a per-speaker slice so validation covers every voice."""
    if fraction <= 0:
        return kept, []
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in kept:
        by_speaker[row.get("speaker") or row["video_id"]].append(row)

    val_keys = set()
    for rows in by_speaker.values():
        n = min(int(round(len(rows) * fraction)), max(len(rows) - 1, 0))
        for row in rng.sample(rows, n):
            val_keys.add(row["audio_file"])

    train = [r for r in kept if r["audio_file"] not in val_keys]
    val = [r for r in kept if r["audio_file"] in val_keys]
    return train, val


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def manifest_entry(row: dict, durations: dict, audio_prefix: str, with_ref: bool) -> dict:
    name = row["audio_file"]
    entry = {
        "audio": f"{audio_prefix}{name}",
        "text": row["_text"],
        "duration": durations[name],
    }
    if with_ref and row.get("_ref") is not None:
        ref = row["_ref"]["audio_file"]
        entry["ref_audio"] = f"{audio_prefix}{ref}"
        # Without this VoxCPM's length estimator decodes every reference clip
        # just to measure it.
        entry["ref_duration"] = durations[ref]
    entry["speaker"] = row.get("speaker")
    entry["video_id"] = row.get("video_id")
    return entry


def write_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def dataset_card(args, stats: dict) -> str:
    hours = stats["hours"]
    return f"""---
license: cc-by-nc-4.0
language:
- ar
task_categories:
- text-to-speech
tags:
- voxcpm
- arabic
- saudi
- dialectal-arabic
- voice-cloning
- tts
pretty_name: Saudi Clean Vox
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
---

# saudi-clean-vox

Saudi dialectal Arabic speech, cut and filtered for **VoxCPM fine-tuning**.
{stats['clips']} single-speaker clips ({hours:.2f} hours) from {stats['videos']}
YouTube videos, mono {args.sample_rate // 1000} kHz 16-bit PCM WAV. Every clip
carries a `speaker` label; there are {stats['speakers']} of them, one per source
video, and they were not linked across videos -- treat two labels as two voices
only within the video that produced them.

Built with [youtube-to-dataset](https://github.com/mohamed-rabee3/youtube-to-dataset):
diarization, VAD segmentation, music and overlap rejection, speaker-purity
checks, SQUIM audio quality gates, and a two-transcript agreement check.

## Layout

```
train.jsonl          # VoxCPM manifest, {stats['train']} rows
val.jsonl            # VoxCPM manifest, {stats['val']} rows
wavs/*.wav           # {args.sample_rate} Hz mono 16-bit PCM
wavs/metadata.jsonl  # HuggingFace audiofolder metadata
dataset_stats.json   # what was kept, what was dropped, and why
```

Clips run {stats['min_duration']}-{stats['max_duration']} s (mean
{stats['mean_duration']} s), inside VoxCPM's usable range and well clear of the
1-second floor below which it becomes unstable.

Manifest rows look like this:

```json
{{"audio": "wavs/xxx_0000.wav", "text": "...", "duration": 6.1, "ref_audio": "wavs/xxx_0007.wav", "ref_duration": 5.2, "speaker": "xxx_SPK1", "video_id": "xxx"}}
```

`audio` / `text` are what VoxCPM requires; `duration` and `ref_duration` let it
skip decoding during length filtering; `ref_audio` is another clip of the same
speaker, which trains the prompted (voice-cloning) path. Drop the `ref_audio`
keys for plain text-to-speech fine-tuning.

## Fine-tuning VoxCPM

Audio paths are **relative to the dataset root**, so either run training from
that directory or rewrite them once:

```bash
huggingface-cli download Rabe3/saudi-clean-vox --repo-type dataset --local-dir saudi-clean-vox
cd saudi-clean-vox
python -c "
import json,pathlib
root=pathlib.Path.cwd()
for name in ('train.jsonl','val.jsonl'):
    rows=[json.loads(l) for l in open(name,encoding='utf-8')]
    for r in rows:
        for k in ('audio','ref_audio'):
            if k in r: r[k]=str(root/r[k])
    open(name,'w',encoding='utf-8').write(''.join(json.dumps(r,ensure_ascii=False)+chr(10) for r in rows))
"
```

Then point [VoxCPM](https://github.com/OpenBMB/VoxCPM)'s config at them:

```yaml
train_manifest: /path/to/saudi-clean-vox/train.jsonl
val_manifest:   /path/to/saudi-clean-vox/val.jsonl
sample_rate: {args.sample_rate}
```

```bash
python scripts/train_voxcpm_finetune.py --args.load conf/voxcpm_v2/voxcpm_finetune_lora.yaml
```

`sample_rate` must match the AudioVAE encoder input rate of the checkpoint you
are fine-tuning ({args.sample_rate} Hz for VoxCPM 1.0 and VoxCPM 2; VoxCPM 1.5
wants 44.1 kHz, which this dataset cannot supply -- the source audio is 24 kHz).

Two things that bite: `datasets>=4` needs `torchcodec` installed before it will
decode an `Audio` column, and the longest sample here is about 550 audio frames
plus text, so the stock `max_batch_tokens: 8192` filters nothing out.

## Loading it directly

```python
from datasets import load_dataset
ds = load_dataset("Rabe3/saudi-clean-vox")                    # manifests, audio as paths
ds = load_dataset("audiofolder", data_dir="saudi-clean-vox/wavs")  # decoded audio + text
```

## Transcripts

Every clip was transcribed twice: YouTube's auto-captions and Cohere Transcribe
Arabic. Clips where the two disagreed by more than 35% CER were dropped. The
`text` shipped here is the **{args.text}** transcript.

YouTube's Arabic auto-captions drop ta marbuta, hamza seats and almost all
punctuation, which is fine for ASR and harmful for TTS -- a model trained on
that spelling never learns to read ordinary written Arabic. Cohere's transcript
preserves the orthography, and the agreement gate already established that the
two say the same thing.

Transcripts are unvocalized (no harakat), as ordinary written Arabic is.

## Quality gates

Every clip passed all of: single speaker, no overlapped speech, no music or
applause (Demucs accompaniment energy + an AST AudioSet classifier), ECAPA
speaker-purity against the speaker's own centroid, SQUIM MOS/STOI/PESQ, clipping
and SNR bounds, and characters-per-second bounds calibrated for Arabic script.

Loudness is already normalized (about -16.4 LUFS across the set).

## Licence and provenance

Audio is derived from YouTube videos and remains subject to YouTube's Terms of
Service and the underlying copyright; `video_id` on each row keeps provenance
attached. Parts of the pipeline that produced it (DiariZen, the MMS forced
aligner) are **CC BY-NC 4.0**, so this dataset is released non-commercially.
"""


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Convert a yt2ds dataset to VoxCPM fine-tuning format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset", type=Path, help="yt2ds dataset directory (holds metadata.jsonl)")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument(
        "--text",
        choices=["cohere", "youtube", "canonical"],
        default="cohere",
        help="which transcript to train on",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="output rate; must match the AudioVAE encoder (16000 for VoxCPM 1.0/2)",
    )
    p.add_argument("--min-duration", type=float, default=2.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--drop-digits", action="store_true", help="drop rows whose text contains digits")
    p.add_argument("--no-ref-audio", action="store_true", help="omit the ref_audio column")
    p.add_argument("--val-fraction", type=float, default=0.03)
    p.add_argument("--silence-floor-db", type=float, default=-35.0, help="edge-trim threshold below peak")
    p.add_argument("--keep-silence", type=float, default=0.1, help="seconds of edge silence to keep")
    p.add_argument("--peak-ceiling-db", type=float, default=-1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--push-to-hub", metavar="REPO_ID", help="upload to a HuggingFace dataset repo")
    p.add_argument("--private", action="store_true", help="create the hub repo private")
    args = p.parse_args(argv)

    meta = args.dataset / "metadata.jsonl"
    if not meta.exists():
        p.error(f"no metadata.jsonl in {args.dataset}")

    rows = [json.loads(line) for line in meta.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"read {len(rows)} rows from {meta}")

    kept, dropped = choose_rows(rows, args)
    if not kept:
        p.error("every row was filtered out")
    for reason, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n:>5}  {reason}")
    print(f"keeping {len(kept)} clips")

    rng = random.Random(args.seed)
    if not args.no_ref_audio:
        assign_ref_audio(kept, rng)

    # --- audio ---------------------------------------------------------
    wav_dir = args.out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            args.dataset / "wavs" / row["audio_file"],
            wav_dir / row["audio_file"],
            args.sample_rate,
            args.silence_floor_db,
            args.keep_silence,
            args.peak_ceiling_db,
        )
        for row in kept
    ]
    missing = [j[0] for j in jobs if not j[0].exists()]
    if missing:
        p.error(f"{len(missing)} source WAVs missing, first: {missing[0]}")

    print(f"converting {len(jobs)} clips to {args.sample_rate} Hz mono with {args.workers} workers...")
    durations: dict[str, float] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, dur) in enumerate(pool.map(convert_clip, jobs, chunksize=16), 1):
            durations[name] = dur
            if i % 250 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")

    # Edge trimming can push a clip under the floor; drop those rather than
    # ship a manifest whose durations disagree with its audio.
    short = [r for r in kept if durations[r["audio_file"]] < max(args.min_duration - 0.25, 1.0)]
    if short:
        names = {r["audio_file"] for r in short}
        for r in short:
            (wav_dir / r["audio_file"]).unlink(missing_ok=True)
        kept = [r for r in kept if r["audio_file"] not in names]
        for r in kept:
            if r.get("_ref") is not None and r["_ref"]["audio_file"] in names:
                r["_ref"] = None
        print(f"  dropped {len(short)} clips that fell under {args.min_duration}s after trimming")

    # --- manifests -----------------------------------------------------
    train, val = split_train_val(kept, args.val_fraction, rng)
    with_ref = not args.no_ref_audio
    write_jsonl(args.out / "train.jsonl", [manifest_entry(r, durations, "wavs/", with_ref) for r in train])
    write_jsonl(args.out / "val.jsonl", [manifest_entry(r, durations, "wavs/", with_ref) for r in val])

    # audiofolder metadata, so `load_dataset("<repo>")` works on the Hub too
    write_jsonl(
        wav_dir / "metadata.jsonl",
        [
            {
                "file_name": r["audio_file"],
                "text": r["_text"],
                "speaker": r.get("speaker"),
                "duration": durations[r["audio_file"]],
                "video_id": r.get("video_id"),
            }
            for r in kept
        ],
    )

    total = sum(durations[r["audio_file"]] for r in kept)
    stats = {
        "clips": len(kept),
        "train": len(train),
        "val": len(val),
        "hours": round(total / 3600, 3),
        "speakers": len({r.get("speaker") for r in kept}),
        "videos": len({r.get("video_id") for r in kept}),
        "sample_rate": args.sample_rate,
        "text_source": args.text,
        "ref_audio": with_ref,
        "min_duration": round(min(durations[r["audio_file"]] for r in kept), 2),
        "max_duration": round(max(durations[r["audio_file"]] for r in kept), 2),
        "mean_duration": round(total / len(kept), 2),
        "dropped": dropped,
    }
    (args.out / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(dataset_card(args, stats), encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")

    if args.push_to_hub:
        push(args.out, args.push_to_hub, args.private)
    return 0


def push(folder: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"uploading to {repo_id} as {who.get('name')}")
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add VoxCPM fine-tuning manifests and 16 kHz clips",
    )
    print(f"https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    sys.exit(main())
