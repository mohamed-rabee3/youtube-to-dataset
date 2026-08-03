# youtube-to-dataset

Turn YouTube links into an Arabic **TTS / voice-cloning dataset**: a `wavs/`
folder of clean, single-speaker clips and a `metadata.jsonl` with verified
transcripts. Built for dialectal Arabic — Saudi, Egyptian, and everything else
— not just Modern Standard.

```bash
yt2ds run "https://youtu.be/VIDEO_ID" --out dataset/
yt2ds run URL1 URL2 URL3 --out dataset/
yt2ds run --urls-file links.txt --out dataset/ --workers 4
yt2ds run URL --out dataset/ --languages ar      # Arabic only, ~2x faster ASR
yt2ds report dataset/ --link-speakers
```

Playlist and channel URLs are expanded automatically, so a whole podcast series
is one argument:

```bash
yt2ds run "https://www.youtube.com/playlist?list=PL..." --out dataset/
yt2ds run "https://www.youtube.com/@channel"            --out dataset/
yt2ds run "https://www.youtube.com/@channel/playlists"  --out dataset/
```

Videos the listing already reports as private, deleted, live or upcoming are
dropped during expansion rather than becoming guaranteed download failures
later, and a listing that contains listings — a channel's `/playlists` tab — is
expanded in turn. Duplicates across several links are collapsed, so overlapping
playlists cost one download each.

## Many videos, one dataset

Videos are processed **strictly one at a time**, and each one is written into
the dataset the moment it finishes. Point a later run at the same `--out` and it
adds to what is already there:

```bash
yt2ds run URL1 URL2 --out dataset/     # builds the dataset
yt2ds run URL3     --out dataset/      # appends; URL1 and URL2 are not touched
```

Finished videos are recognised **from the URL alone**, so re-running a
200-link file after adding one new link costs one download, not 201. Because
each video is emitted as it completes, killing a long run keeps everything
processed so far — `metadata.jsonl`, `manifest.json` and the WAVs are all
consistent at every point. Restart it and it picks up at the first unfinished
video.

Downloading runs a few videos ahead of processing so the GPU is never waiting
on the network, but never further: `--workers 1` keeps exactly one download in
flight.

**Each video is deleted as soon as its clips are written.** The download, the
two decoded masters, the MP3, the caption file and the info JSON are all spent
once the chunks are on disk and their transcripts are in `metadata.jsonl` —
about 0.5 GB per source hour, plus ~27 MB of MP3 per video. A thousand-video
run would otherwise need far more space for its scratch than for its output.
`--keep-intermediates` keeps the lot; `--keep-mp3` keeps just the archival MP3.

## When downloads fail

YouTube answers each of its own player clients differently. The same video that
returns *"Sign in to confirm you're not a bot"* or a 403 on its media URL to
one client often downloads cleanly as another, which is why plain retries — the
same request, to the same client, again — do not help.

So every extraction is retried under a **different player client** each time,
with a growing pause between attempts:

```yaml
download:
  player_clients: ["default", "tv_simply", "web_safari", "android_vr", "ios", "mweb"]
  attempt_backoff: 4.0
```

Failures that no client can fix — private, members-only, removed, geo-blocked —
are recognised and not retried, so a dead link costs one attempt rather than
six. If the audio comes down but the subtitle track will not, the video is
re-fetched without subtitles: the captions are a second opinion, never a reason
to lose a video.

Everything that still fails is written to `dataset/failed.txt` with its reason,
in `--urls-file` format. A failed video saves no state, so feeding that file
back retries exactly those and nothing else:

```bash
yt2ds run --urls-file dataset/failed.txt --out dataset/
```

**Cookies are the real fix for bot checks.** A datacentre IP trips them
eventually no matter which client asks. Export a `cookies.txt` from a
logged-in session and pass it:

```bash
yt2ds run --urls-file links.txt --out dataset/ --cookies cookies.txt
yt2ds run --urls-file links.txt --out dataset/ --cookies-from-browser chrome
```

If a whole run starts failing, lower `--workers` before anything else — several
parallel downloads from one IP is what triggers throttling in the first place.

## What it does

```
download ──> decode ──> diarize ──> VAD ──> segment ──> music/noise ──>
             (48k+16k)  (DiariZen)          (chunks)    (Demucs + AST)

  ──> speaker purity ──> transcribe ──> align YT subs ──> verify ──> emit
      (ECAPA)            (Cohere)       (MMS CTC)         (CER gate)
```

Each chunk that survives becomes a `wavs/*.wav` at 24 kHz mono plus one
`metadata.jsonl` line. Everything rejected goes to `rejected.jsonl` with the
reason and every score, so nothing is silently thrown away.

### The design decision that matters

**Cohere Transcribe Arabic emits no timestamps.** Rather than working around
that, the pipeline is built on it: segment the audio *first*, then transcribe
each chunk. A chunk's transcript is then simply that chunk's text — there is no
alignment step that can drift, and no word can bleed across a boundary.

Forced alignment is used only for the other path: mapping YouTube's subtitle
*text* onto chunk boundaries so the two transcripts can be compared.

### Text fidelity

YouTube's auto-captions for heavy Saudi or Egyptian dialect are frequently
wrong. For an ASR dataset that is tolerable noise; for TTS it is not — a wrong
transcript teaches the model to pronounce a word as something else.

So every chunk gets **two** transcripts: YouTube's (force-aligned onto the
chunk) and Cohere's. YouTube's text is canonical, but the chunk is dropped when
the character error rate between them exceeds `filters.max_cer_yt_vs_cohere`
(default 0.35). Both texts stay in the metadata, so the threshold can be
re-tuned against `rejected.jsonl` without recomputing anything.

Videos with no Arabic subtitles fall back to Cohere alone; `text_source`
records which path produced each row. On that path there is no second opinion,
so the model's own confidence (mean per-token log-probability) is gated
instead — see `filters.min_asr_confidence`.

### Arabic *and* English

The model handles both languages and Arabic-English code-switching, but its
processor has no auto-detect: it decodes under a language prompt. Forcing every
chunk through the Arabic prompt makes it invent Arabic for speech that is
actually English.

So each chunk is decoded once per language in `asr.languages` (default
`["ar", "en"]`) and the higher-confidence result wins. Each row records
`asr_language`, `asr_confidence`, and the runner-up's score in
`asr_confidence_alt`, so the decision is inspectable. Code-switching *within* a
chunk is handled by the model under whichever prompt scores better, and the
script check (`filters.min_script_ratio`) is a proportion rather than a purity
test, so an Arabic sentence containing an English brand name passes.

Set `asr.languages: ["ar"]` to halve ASR time on Arabic-only material.

## Setup

```bash
scripts/setup.sh            # both venvs + model prefetch
scripts/setup.sh main       # main venv only
scripts/setup.sh diarize    # diarization venv only
```

Requires `uv`, `git`, `ffmpeg`/`ffprobe`, and an NVIDIA GPU. Set `HF_TOKEN` if
the diarization weights need it.

### Why two virtualenvs

`.venv` runs everything except diarization: torch 2.9 and `transformers>=5.4`,
which Cohere Transcribe Arabic requires. `.venv-diarize` runs DiariZen, which
vendors `pyannote-audio` against torch 2.1.1. The two cannot coexist, so they
talk over a subprocess boundary (`scripts/diarize_worker.py`) with a small JSON
contract. Point `YT2DS_DIARIZE_PYTHON` at any interpreter that has `diarizen`
installed to override.

## Models

| Stage | Model | Licence |
|---|---|---|
| ASR | `CohereLabs/cohere-transcribe-arabic-07-2026` | Apache-2.0 |
| Diarization | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` | **CC BY-NC 4.0** |
| Forced alignment | `MahmoudAshraf/mms-300m-1130-forced-aligner` | CC-BY-NC-4.0 |
| Music/event detection | `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD |
| Source separation | Demucs `htdemucs` | MIT |
| Speaker embedding | `speechbrain/spkrec-ecapa-voxceleb` | Apache-2.0 |
| VAD | Silero VAD | MIT |
| Quality | TorchAudio-SQUIM | BSD |

> **Licence note.** DiariZen's weights are **non-commercial**. It was chosen for
> accuracy — DER 9.1 on VoxConverse and 14.5 on DIHARD 3, better than pyannote
> community-1 across the board — with that trade understood. `stages/diarize.py`
> talks to a generic worker contract, so swapping in a permissively licensed
> diarizer is a configuration change rather than a rewrite.

## Output

```
dataset/
├── wavs/                 # 24 kHz mono 16-bit PCM
├── metadata.jsonl        # one line per kept clip
├── rejected.jsonl        # every dropped clip + reason + scores
├── speakers.json         # per-speaker hours, global speaker links
├── manifest.json         # run config, per-video summary
├── failed.txt            # URLs that failed + why; feed back with --urls-file
├── mp3/                  # archival full-length audio, only with --keep-mp3
└── .work/                # state + speaker centroids; safe to delete
```

Each `metadata.jsonl` row carries the text (`text`, `text_yt`, `text_cohere`,
`text_source`, `cer_yt_vs_cohere`), the language it was decoded in
(`language`, `asr_language`, `asr_confidence`, `asr_confidence_alt`), the
speaker (`speaker`, `global_speaker`, `speaker_conf`), and every quality score
behind the keep/drop decision (`squim_mos`, `squim_stoi`, `squim_pesq`,
`music_score`, `vocal_ratio`, `snr_db`, `chars_per_sec`, `script_ratio`, …).

Runs are resumable: finished videos are skipped on a re-run, and adding a URL
to an existing dataset only processes the new video. A video interrupted
mid-write has its partial rows cleared before it is reprocessed, so
`metadata.jsonl` never accumulates duplicates. `.work/` holds only the state
and speaker centroids once a video is done — the bulky decodes are cleaned up
as the run goes.

## Quality gates

All configurable in `configs/default.yaml` or via CLI flags:

- **Music / noise** — Demucs accompaniment-energy ratio *and* an AST AudioSet
  score. Two detectors because each catches what the other misses: Demucs sees
  speech-over-background-music that a classifier reads as ordinary speech; AST
  catches applause and crowd noise that Demucs files under "vocals".
- **Overlapped speech** — discarded outright. Two voices in one clip is poison
  for voice cloning. `--keep-overlap` disables this.
- **Speaker purity** — every chunk is embedded with ECAPA and compared to its
  own speaker's centroid; outliers are dropped. Diarization error rate is
  measured over whole recordings, but what matters here is whether *this clip*
  contains *that voice*.
- **Audio quality** — SQUIM estimated MOS / STOI / PESQ, plus clipping, peak
  level and SNR.
- **Text** — the CER gate above, plus the script check, word count, ASR
  confidence, and characters-per-second bounds that catch both bled-in and
  missing text. Those bounds are calibrated for Arabic script (measured speech
  runs 5–13 chars/sec), which is far more compact than Latin.

## MP3 note

Chunks are never cut from the MP3 by default. Converting to MP3 and then
slicing training clips out of the decoded result bakes compression artefacts
into the dataset, so the pipeline decodes the download straight to lossless WAV.
Set `audio.chunk_from_mp3: true` to cut from the MP3 anyway — it is encoded on
demand for that, then deleted with the rest.

`--keep-mp3` (or `audio.keep_mp3: true`) keeps a full-length MP3 per video in
`mp3/` as an archive. It is off by default: nothing downstream reads it, and at
~27 MB per video it outgrows the dataset it accompanies.

## Development

```bash
.venv/bin/python -m pytest tests/ -q
```

Tests cover Arabic normalization and CER, subtitle parsing (including
YouTube's rolling auto-caption repetition), and the chunker's boundary rules.

## Provenance

Downloading YouTube content is subject to YouTube's Terms of Service and the
underlying copyright. Every row records `video_url`, `channel` and
`upload_date` so provenance stays attached to each sample.
