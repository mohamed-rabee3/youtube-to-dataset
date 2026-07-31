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
is one argument.

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
├── mp3/                  # archival full-length audio
└── .work/                # intermediates; safe to delete
```

Each `metadata.jsonl` row carries the text (`text`, `text_yt`, `text_cohere`,
`text_source`, `cer_yt_vs_cohere`), the speaker (`speaker`, `global_speaker`,
`speaker_conf`), and every quality score used to make the keep/drop decision
(`squim_mos`, `squim_stoi`, `squim_pesq`, `music_score`, `vocal_ratio`,
`snr_db`, `chars_per_sec`, …).

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

You may notice chunks are not cut from the MP3. Converting to MP3 and then
slicing training clips out of the decoded result bakes compression artefacts
into the dataset, so the pipeline decodes the download straight to lossless WAV
and writes the MP3 alongside as an archive. Set `audio.chunk_from_mp3: true` to
cut from the MP3 anyway.

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
