# youtube-to-dataset

Turn YouTube links into an Arabic **TTS / voice-cloning dataset**: a `wavs/`
folder of clean, single-speaker clips and a `metadata.jsonl` with verified
transcripts. Built for dialectal Arabic — Saudi, Egyptian, and everything else
— not just Modern Standard.

```bash
yt2ds run "https://youtu.be/VIDEO_ID" --out dataset/
yt2ds run URL1 URL2 URL3 --out dataset/
yt2ds run --urls-file links.txt --out dataset/ --workers 4
yt2ds run mp3-corpus/ --out dataset/             # audio already on disk
yt2ds run URL --out dataset/ --languages ar      # Arabic only
yt2ds run URL --out dataset/ --asr-backend cohere  # transcribe locally, no API
yt2ds report dataset/ --link-speakers
```

Transcription defaults to **Google Cloud Speech-to-Text** (`latest_long`), so
it needs credentials — see [Transcription backends](#transcription-backends).
`--asr-backend cohere` runs the local GPU model instead and needs none.

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

## Audio already on disk

A local file or a directory of them can be given wherever a URL can, and the
two mix freely in one run. A directory expands to the audio files inside it, in
natural order, so `2.mp3` comes before `10.mp3`:

```bash
yt2ds run mp3-corpus/ --out dataset/
yt2ds run corpus/episode-01.mp3 "https://youtu.be/VIDEO_ID" --out dataset/
```

Everything downstream of the download is identical. Two differences follow from
the source rather than the code:

* **No subtitles, so no CER cross-check.** The fidelity gate described under
  [Text fidelity](#text-fidelity) compares YouTube's captions against the ASR
  transcript; a local file has no captions, so the ASR text is canonical and
  the confidence floor is the only guard against a hallucinated transcript.
  `text_source` records the backend that produced it — `google` or `cohere`.
* **The source file is never deleted.** Cleanup removes what the run fetched,
  and a local corpus is not that. Its decoded working copies still go.

The dataset id for a local file is its filename stem, so `1.mp3` emits
`1_0001.wav` and friends — no collision with the 11-character YouTube ids if
both end up in one dataset.

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
download ──> decode ──> isolate voice ──────> diarize ──> VAD ──> segment ──>
             (48k+16k)  (Mel-Band RoFormer)   (DiariZen)          (chunks)

  ──> music/noise ──> speaker purity ──> transcribe ──> align YT subs ──> emit
      (Demucs + AST)  (ECAPA)            (Google STT)   (MMS CTC)
                                         (or Cohere)
```

Each chunk that survives becomes a `wavs/<speaker>/*.wav` at 24 kHz mono plus
one `metadata.jsonl` line. Everything rejected goes to `rejected.jsonl` with the
reason and every score, so nothing is silently thrown away.

### The design decision that matters

**Segment first, transcribe second.** A chunk's transcript is then simply that
chunk's text — there is no alignment step that can drift, and no word can bleed
across a boundary. It also means no transcriber has to produce usable
timestamps, which is what lets the backend be swapped: Cohere Transcribe Arabic
emits none at all, and nothing downstream misses them.

Forced alignment is used only for the other path: mapping YouTube's subtitle
*text* onto chunk boundaries so the two transcripts can be compared.

### Music: removed, not avoided

Drama, interviews and anything else scored under a soundtrack used to lose most
of its speech to the music gate — the voice was fine, it simply had an
orchestra behind it. So the music bed is now taken out instead: a **Mel-Band
RoFormer** vocal isolator runs over the whole video before segmentation, and
every stage after it — VAD, diarization, alignment, ASR, and the clips that are
written — sees the isolated voice.

Measured over a 32-minute Arabic drama episode:

| | before | after |
|---|---|---|
| AST music score, median | 0.447 | 0.007 |
| Demucs accompaniment ratio, median | 0.005 | 0.000 |
| chunks the old music gate would drop | 96 / 137 | 0 / 137 |

Why a second separator when the music *detector* already loads Demucs: they
answer different questions. Demucs gives a cheap four-stem energy ratio, which
is enough to score a chunk. Mel-Band RoFormer is about 2.3 dB better on vocal
SDR (11.53 against 9.24 on MUSDB), which is what matters when the separated
voice is the thing being kept. It runs at roughly 40x realtime on an Ada-class
card — a 32-minute episode is isolated in under a minute.

It runs over the whole video rather than per chunk because these models
reconstruct the voice from spectral context, and context reaching across a
chunk boundary is context they get to use. Loudness is re-measured afterwards,
since pulling the bed out moves the integrated loudness of a scene.

`--no-separate` turns it off; `separate.model` takes any
[audio-separator](https://github.com/nomadkaraoke/python-audio-separator) model
filename, so a different isolator is a config change.

### Text fidelity

YouTube's auto-captions for heavy Saudi or Egyptian dialect are frequently
wrong. For an ASR dataset that is tolerable noise; for TTS it is not — a wrong
transcript teaches the model to pronounce a word as something else.

So every chunk gets **two** transcripts: YouTube's (force-aligned onto the
chunk) and the ASR backend's. YouTube's text is canonical, but the chunk is
dropped when the character error rate between them exceeds
`filters.max_cer_yt_vs_cohere` (default 0.35). Both texts stay in the metadata,
so the threshold can be re-tuned against `rejected.jsonl` without recomputing
anything.

Videos with no Arabic subtitles fall back to the ASR transcript alone;
`text_source` records which path produced each row. On that path there is no
second opinion, so the transcriber's own confidence is gated instead. The two
backends score on different scales and so have separate floors:
`filters.min_asr_confidence_google` (0-1, default 0.5) and
`filters.min_asr_confidence` (mean per-token log-probability, default -1.0).

> The `text_cohere` and `cer_yt_vs_cohere` column names predate the Google
> backend and are kept as they are, so datasets built before it stay readable
> against one schema. They hold whichever backend ran.

### Transcription backends

`asr.backend` picks the transcriber; `--asr-backend` overrides it per run.

| | `google` (default) | `cohere` | `google_batch` |
|---|---|---|---|
| Model | Cloud Speech-to-Text v1 `latest_long` | `CohereLabs/cohere-transcribe-arabic-07-2026` | Speech-to-Text **V2** `long` |
| Unit of work | one request per chunk | one decode per chunk | one request per **episode** |
| Needs | credentials, billed per audio minute | ~5 GB of GPU, no network | credentials **and a GCS bucket** |
| Language choice | API detects among the configured locales | one decode pass per language | fixed `language_codes` |
| Confidence | its own 0-1 score | mean per-token log-probability | its own 0-1 score, averaged per chunk |

#### `google_batch`: whole episodes, a quarter of the price

Google bills each request **rounded up to 15 seconds**. Chunks here average
under five, so transcribing a corpus clip by clip is billed for roughly three
times the audio it actually contains. `google_batch` sends the whole episode in
one request instead — billed for its true length — and cuts the returned
word-level timestamps into the chunks the diarizer already chose. The
forced-alignment pass goes too: its only job was to place subtitle words in
time, and these words arrive with times.

Combined with V2's `DYNAMIC_BATCHING` tier (about a quarter of the standard
per-minute price, for a latency SLA of up to 24 hours — measured turnaround has
been seconds), a 200-hour corpus costs roughly **$36 instead of ~$800**.

Because batch recognition only reads from Cloud Storage and answers as a
long-running operation, it is a **two-phase** backend:

```bash
# 1. GPU pass: separate, diarize, segment, score. Emits chunks with no text.
yt2ds run socrates/ --out dataset/ --asr-backend google_batch

# 2. Upload, transcribe, cut words into chunks, apply the text gates.
yt2ds transcribe dataset/ --bucket my-asr-bucket
```

Phase 2 prints a cost estimate before spending anything, caches each episode's
words under `.work/asr-words/` so an interrupted pass never re-pays, and takes
`--limit N` for a costed trial. Rows wait with `text_source: "pending"` between
the phases; a partial phase 2 leaves untranscribed episodes pending rather than
deleting audio it has no text for.

> **Location matters.** `ar-SA` is served by `long` only in the `global` and
> `us` locations — `us-central1` rejects the request outright, and no Chirp
> model offers `ar-SA` at all (only `ar-EG`). The defaults are the verified
> working combination.

> **Orthography.** Speech-to-Text returns Arabic without hamza (أ/إ → ا) and
> writes ة as ه, so `الأسئلة` comes back as `الاسئله`. `clean_for_output`
> preserves orthography, so that reaches the dataset verbatim. The local
> `cohere` backend does not have this problem. `scripts/diacritize.py` can
> repair it — its "letters must not change" guard compares normalized
> skeletons, which is exactly the equivalence that lets hamza and ة be restored.

**Credentials** for the default backend come from `asr.google.credentials_file`,
else `GOOGLE_APPLICATION_CREDENTIALS`, else application default credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
yt2ds run URL --out dataset/
# or, per run:
yt2ds run URL --out dataset/ --google-credentials /path/to/service-account.json
```

Requests go out **one per chunk** — that is the price of segmenting first, and
it is what keeps a chunk's transcript exactly that chunk's text. They are sent
concurrently (`asr.google.max_workers`, default 8), so an hour of speech is a
few thousand small requests rather than a few thousand round trips. Transient
5xx and quota errors are retried with backoff; a chunk that never succeeds
comes back empty and is rejected as `text:asr_empty` rather than failing the
video.

`asr.google.model` takes any v1 model name, so `chirp_2` or `latest_short` is a
config change.

### Tashkeel

No ASR backend here emits diacritics — Cloud Speech-to-Text has no flag for it
and returns none. `scripts/diacritize.py` adds them as a separate pass, from
the audio, so the marks match how the speaker actually pronounced the word
rather than what Modern Standard grammar would prescribe:

```bash
scripts/diacritize.py dataset/ --dialect najdi --workers 8
```

Two rules are enforced **in code**, not merely asked of the model
(`yt2ds.arabic`):

* **the letters must not change** — if the bare skeleton moved, the model
  rewrote the line instead of marking it, and the row is left alone. Because
  the comparison normalizes, a model that restores a missing hamza seat or ta
  marbuta *is* accepted, and logged as `ok_orthography_fixed`;
* **no mark on any word's final letter** — every accepted result goes through
  `strip_final_tashkeel`, which clears the trailing letter and also the
  tanwin-fath-plus-alef spelling of the same ending (`مَرْحَبًا` → `مَرْحَبا`).
  Case endings are grammatical and inaudible in ordinary speech, so a TTS model
  should never be taught to pronounce them.

The pass is also the repair for ASR orthography. It is allowed to correct two
things and nothing else — a bare alef whose root is a hamza, and a word-final
ه whose root is ة — which is exactly the damage Speech-to-Text does. Measured
over a 20-clip sample of Google `long` output, every row came back
`ok_orthography_fixed`:

```
before:  انقل اسئلتكم واضعها على طاوله قاده التحول.
after :  أَنْقُل أَسْئِلَتَكُم وَأَضَعُهَا عَلَى طَاوِلَة قَادَة التَّحَوُّل.
```

Tatweel (U+0640) is stripped by `clean_for_output`: a model asked for marks will
occasionally insert one as somewhere to hang a vowel, and being inside the
Arabic letter range it would otherwise pose as a word's final letter and shield
a real case ending from the rule above.

`--dialect` names the expected dialect (`najdi`, `hijazi`, `gulf`, `saudi`,
`egyptian`, `levantine`, or any string you write). It is deliberately a
**prior, not an override**: it resolves vowels the audio leaves ambiguous, and
the prompt says explicitly that heard pronunciation wins whenever the two
disagree. Interview speech code-switches constantly toward MSA and a guest need
not share the host's dialect, so a model told flatly "this is Najdi" would
impose Najdi vowels on Modern Standard pronunciation — the exact failure mode
that makes text-only diacritizers (Farasa, Mishkal, CATT) unusable here.
Omitting the flag keeps the earlier dialect-agnostic prompt.

`text` becomes the diacritized string and `text_cohere` keeps the undiacritized
transcript it was built from, so nothing is lost. Results are cached per clip in
`.work/diacritize_cache.jsonl`, so an interrupted pass resumes without re-paying.

### Arabic *and* English

Both backends handle both languages and Arabic-English code-switching; they
find the language differently.

**google** sends the first entry of `asr.languages` as the primary locale and
the rest as alternatives for the API to detect among, so a chunk costs one
request no matter how many languages are listed. `asr.google.locales` maps the
short codes onto BCP-47 (`ar` → `ar-SA`, `en` → `en-US`); an entry that is
already a locale passes through, so `--languages ar-EG,en` works. Not every
Speech-to-Text model accepts alternatives — `latest_long` does not — so the
first rejection drops them for the rest of the run and logs that it did.

**cohere** has no auto-detect: it decodes under a language prompt, and forcing
every chunk through the Arabic prompt makes it invent Arabic for speech that is
actually English. So each chunk is decoded once per language in `asr.languages`
and the higher-confidence result wins, with the runner-up's score kept in
`asr_confidence_alt`. `asr.languages: ["ar"]` halves ASR time on this backend.

Either way each row records `asr_language` and `asr_confidence`, and the script
check (`filters.min_script_ratio`) is a proportion rather than a purity test,
so an Arabic sentence containing an English brand name passes.

## Setup

```bash
scripts/setup.sh            # both venvs + model prefetch
scripts/setup.sh main       # main venv only
scripts/setup.sh diarize    # diarization venv only
```

Requires `uv`, `git`, `ffmpeg`/`ffprobe`, and an NVIDIA GPU. Set `HF_TOKEN` if
the diarization weights need it, and `GOOGLE_APPLICATION_CREDENTIALS` for the
default ASR backend (not needed with `--asr-backend cohere`).

### Why two virtualenvs

`.venv` runs everything except diarization: torch 2.9 and `transformers>=5.4`,
which the `cohere` ASR backend requires. `.venv-diarize` runs DiariZen, which
vendors `pyannote-audio` against torch 2.1.1. The two cannot coexist, so they
talk over a subprocess boundary (`scripts/diarize_worker.py`) with a small JSON
contract. Point `YT2DS_DIARIZE_PYTHON` at any interpreter that has `diarizen`
installed to override.

## Models

| Stage | Model | Licence |
|---|---|---|
| ASR (default) | Google Cloud Speech-to-Text `latest_long` | hosted API |
| ASR (`--asr-backend cohere`) | `CohereLabs/cohere-transcribe-arabic-07-2026` | Apache-2.0 |
| Diarization | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` | **CC BY-NC 4.0** |
| Forced alignment | `MahmoudAshraf/mms-300m-1130-forced-aligner` | CC-BY-NC-4.0 |
| Music/event detection | `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD |
| Vocal isolation | Mel-Band RoFormer `mel_band_roformer_kim_ft_unwa` | MIT |
| Music scoring | Demucs `htdemucs` | MIT |
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
├── wavs/                 # 24 kHz mono 16-bit PCM, one folder per speaker
│   ├── <speaker>/        # <video_id>_SPK<n> during a run; GLOBAL_SPEAKER_nn
│   │                     # after `report --link-speakers` pools voices
│   └── _unassigned/      # clips whose speaker pooled under
│                         # speakers.min_speaker_seconds -- diarization debris
├── metadata.jsonl        # one line per kept clip
├── rejected.jsonl        # every dropped clip + reason + scores
├── speakers.json         # per-speaker hours, global speaker links
├── manifest.json         # run config, per-video summary
├── failed.txt            # URLs that failed + why; feed back with --urls-file
├── mp3/                  # archival full-length audio, only with --keep-mp3
└── .work/                # state + speaker centroids; safe to delete
```

Each `metadata.jsonl` row carries the text (`text`, `text_yt`, `text_cohere` —
the ASR transcript whichever backend produced it, `text_source`,
`cer_yt_vs_cohere`), the language it was decoded in
(`language`, `asr_language`, `asr_confidence`, `asr_confidence_alt` on
`cohere`), the
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

- **Music / noise** — measured, not fatal. The music bed is *removed* before
  anything else runs (see below), so the Demucs accompaniment-energy ratio and
  the AST AudioSet score describe the residue the isolator left behind. They
  are recorded per clip; `--reject-music` restores the old behaviour of
  dropping the chunk instead. Two detectors because each catches what the other
  misses: Demucs sees speech-over-background-music that a classifier reads as
  ordinary speech; AST catches applause and crowd noise that Demucs files under
  "vocals".
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
YouTube's rolling auto-caption repetition), the chunker's boundary rules, and
the ASR backends' request/response mapping (no credentials needed — the API
call itself is stubbed).

## Provenance

Downloading YouTube content is subject to YouTube's Terms of Service and the
underlying copyright. Every row records `video_url`, `channel` and
`upload_date` so provenance stays attached to each sample.
