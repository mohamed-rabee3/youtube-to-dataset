# youtube-to-dataset

Turn YouTube links or local audio into an Arabic **TTS / voice-cloning
dataset**: a `wavs/` folder of clean, single-speaker clips that start and end in
real pauses, and a `metadata.jsonl` with fully diacritized, cross-checked
transcripts. Built for dialectal Arabic — Saudi, Egyptian, and everything else
— not just Modern Standard.

```bash
yt2ds run "https://youtu.be/VIDEO_ID" --out dataset/
yt2ds run URL1 URL2 URL3 --out dataset/
yt2ds run --urls-file links.txt --out dataset/ --workers 4
yt2ds run mp3-corpus/ --out dataset/             # audio already on disk
yt2ds run URL --out dataset/ --asr-backend google  # Cloud Speech-to-Text instead
yt2ds speakers dataset/                            # one folder per voice
yt2ds report dataset/
```

Transcription runs locally on the GPU by default:
**[Cohere-Speech-Tashkeel-2B](https://huggingface.co/NAMAA-Space/Cohere-Speech-Tashkeel-2B)**
writes fully diacritized Arabic straight from the audio, and
**Cohere Transcribe Arabic** transcribes every clip a second time as a check.
No API key, no per-minute cost — see [Transcription backends](#transcription-backends).

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
download ──> decode ──> isolate voice ──────> diarize ──> VAD ──> segment ──────>
             (48k+16k)  (Mel-Band RoFormer)   (DiariZen)  (Silero) (edges in pauses)

  ──> music/noise ──> speaker purity ──> transcribe ─────────> verify ─────> emit
      (Demucs + AST)  (ResNet293)        (Tashkeel-2B,         (Cohere        (loudness,
                                          diacritized)          Transcribe)    no clipping)

  then:  yt2ds speakers  ──>  speakers/GLOBAL_SPEAKER_nn/   (one folder per voice)
```

Each chunk that survives becomes a `wavs/<speaker>/*.wav` at 24 kHz mono plus
one `metadata.jsonl` line. Everything rejected goes to `rejected.jsonl` with the
reason and every score, so nothing is silently thrown away.

### The design decision that matters

**Segment first, transcribe second — one clip at a time.** A chunk's transcript
is then simply that chunk's text: there is no alignment step that can drift, and
no transcriber is ever shown two clips at once, so no clip's words can end up in
its neighbour's label. (An audit of a corpus transcribed four neighbouring clips
per request found ~13% of labels shifted onto the wrong clip; see
[Text fidelity](#text-fidelity).)

Forced alignment is used only for the other path: mapping YouTube's subtitle
*text* onto chunk boundaries so the two transcripts can be compared.

### Clip boundaries

The rule the segmenter enforces: **every clip starts and ends in a real
pause.** A clip cut mid-word teaches a TTS model to clip its own words, and a
clip whose edge holds a sliver of the next sentence teaches it to babble one.

* A clip edge must sit at a VAD region boundary with at least
  `segment.min_cut_pause` (0.15 s) of silence to the nearest speech of *any*
  speaker — VAD's, or a diarized turn VAD missed.
* An edge that lands mid-phrase — a speaker change or overlapped speech with no
  pause — is pulled back to the previous valid pause. The partial phrase is
  dropped, never cut.
* Long speech is split only at its longest valid pause (sentence ends pause
  longest). A stretch with no valid pause is dropped; there is no fixed-length
  fallback cut.
* Each clip gets up to `segment.pad` (0.25 s) of silence on each side, but
  never more than half the gap to the neighbouring speech — so no two clips
  share audio, and no clip's padding contains someone's words.
* Same-speaker speech separated by up to `segment.merge_gap` (0.8 s) stays in
  one clip, so sentences are not chopped at every breath.

Each row records where its speech sits (`speech_start`, `speech_end`,
`voiced_seconds`, `max_internal_pause`), so the padding is visible.

These rules replace an earlier design that padded every chunk blindly by 0.2 s
and fell back to a fixed time grid when a stretch had no pause. An audit of a
14-hour dataset built that way found 17% of clips overlapping their
neighbour by up to 0.4 s, and 758 clips cut on the grid mid-word.
`tests/test_segment.py` checks the guarantees above on 200 randomly generated
conversations.

Written clips are loudness-normalized (`audio.loudness: clip`, −23 LUFS by
default, or one gain per video with `video`), with the gain capped so no sample
exceeds `audio.peak_ceiling_dbfs` (−1 dBFS) *after* resampling — nothing is ever
hard-clipped by the 16-bit writer — and a 10 ms fade on each edge.

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

For TTS a wrong transcript is not noise — it teaches the model to say a word as
something else. So no transcript is trusted on its own:

* **Two independent transcriptions of every clip.** The canonical text comes
  from `asr.model` (Cohere-Speech-Tashkeel-2B); `asr.verify_model` (Cohere
  Transcribe Arabic) transcribes the same clip again. A clip is dropped when the
  undiacritized skeletons disagree by more than `filters.max_cer_verify` (0.35).
  On Saudi podcast speech that rejects ~2% of clips, almost all genuinely
  mis-transcribed. Clips containing English or digits are not gated: the two
  models write those differently (`social anxiety` vs `سوشيل أنزايتي`).
* **The model's own confidence**, gated at `filters.min_asr_confidence`
  (−0.10 mean per-token log-probability; the tashkeel model's median is about
  −0.03, and every clip measured below −0.10 was garbled).
* **Subtitles, when a video has them**, force-aligned onto the clip and
  compared with the transcript (`filters.max_cer_yt_vs_asr`). By default the
  diacritized ASR text stays canonical (`filters.prefer_asr_text`), because
  YouTube's captions carry neither diacritics nor reliable spelling;
  `text_source` then reads e.g. `cohere+auto`.
* **Speaking rate**, counted on undiacritized letters per second of *speech*
  (not of padding), catches a transcript far too long or short for its audio.

Every transcript stays in the metadata (`text`, `text_asr`, `text_verify`,
`text_yt`, with `cer_verify` and `cer_yt_vs_asr`), so thresholds can be re-tuned
against `rejected.jsonl` without recomputing anything.

The Google backends have no second model and gate on their own 0–1 confidence
(`filters.min_asr_confidence_google`). The optional Gemini transcription
scripts send one clip per request by default; with `--max-batch` above 1 they
still never put two clips of one video in the same request.

### Transcription backends

`asr.backend` picks the transcriber; `--asr-backend` overrides it per run.

| | `cohere` (default) | `google` | `google_batch` |
|---|---|---|---|
| Model | `NAMAA-Space/Cohere-Speech-Tashkeel-2B`, checked by `CohereLabs/cohere-transcribe-arabic-07-2026` | Cloud Speech-to-Text v1 `latest_long` | Speech-to-Text **V2** `long` |
| Output | fully diacritized Arabic | undiacritized | undiacritized |
| Unit of work | one decode per chunk, per model | one request per chunk | one request per **episode** |
| Needs | ~10 GB of GPU, no network | credentials, billed per audio minute | credentials **and a GCS bucket** |
| Language choice | one decode pass per language in `asr.languages` | API detects among the configured locales | fixed `language_codes` |
| Confidence | mean per-token log-probability | its own 0-1 score | its own 0-1 score, averaged per chunk |

`--asr-model` and `--verify-model` (or `none`) swap the two Cohere models.

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

The default transcriber writes the diacritics itself, **from the audio**:
Cohere-Speech-Tashkeel-2B is Cohere Transcribe Arabic fine-tuned to emit fully
diacritized text, so the marks follow what the speaker actually pronounced —
dialectal vowels, sukun at pauses — rather than what a text-only diacritizer
infers from spelling and Modern Standard grammar. Its model card reports a
diacritic error rate of 5.7% (6.6% with case endings) on natural speech.

```
تَحِسّ نَفْسَكْ أَنَّكْ أَنْتَ مُسْتَنْزِفْ
هَذَا اللِّي الْيَوْمْ بَنِتْكَلَّمْ عَنَّهْ وَبَشَارِكْكُمْ فِي نِهَايَةِ الْحَلَقَةْ
```

`scripts/diacritize.py` (Gemini, audio + existing text) remains for datasets
built before this default, or for text from another source. It enforces two
rules in code: the letters must not change, and no mark goes on a word's final
letter. `scripts/retranscribe.py` re-runs the default transcriber over an
existing dataset's clips instead.

Tatweel (U+0640) is stripped by `clean_for_output`: it carries no sound, and
inside the Arabic letter range it would otherwise pose as a word's final letter.

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

**cohere** has no auto-detect: it decodes under a language prompt. With several
entries in `asr.languages` each chunk is decoded once per language and the
higher-confidence result wins, with the runner-up's score kept in
`asr_confidence_alt`. The tashkeel model is trained for Arabic, so the default
is `["ar"]`; English words inside Arabic speech come out transliterated
(`سوشيل أنزايتي`). Use `--languages ar,en` with a model that also writes English.

Either way each row records `asr_language` and `asr_confidence`, and the script
check (`filters.min_script_ratio`) is a proportion rather than a purity test,
so an Arabic sentence containing an English brand name passes.

## Setup

```bash
scripts/setup.sh            # both venvs + model prefetch
scripts/setup.sh main       # main venv only
scripts/setup.sh diarize    # diarization venv only
```

Requires `uv`, `git`, `ffmpeg`/`ffprobe`, and an NVIDIA GPU (the default set of
models needs about 20 GB of GPU memory). Set `HF_TOKEN` if the diarization
weights need it, and `GOOGLE_APPLICATION_CREDENTIALS` only for the Google
backends.

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
| ASR (default) | `NAMAA-Space/Cohere-Speech-Tashkeel-2B` | Apache-2.0 |
| ASR cross-check | `CohereLabs/cohere-transcribe-arabic-07-2026` | Apache-2.0 |
| ASR (`--asr-backend google`) | Google Cloud Speech-to-Text `latest_long` | hosted API |
| Diarization | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` | **CC BY-NC 4.0** |
| Speaker embedding | `Wespeaker/wespeaker-voxceleb-resnet293-LM` | CC BY 4.0 |
| Forced alignment | `MahmoudAshraf/mms-300m-1130-forced-aligner` | CC-BY-NC-4.0 |
| Music/event detection | `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD |
| Vocal isolation | Mel-Band RoFormer `mel_band_roformer_kim_ft_unwa` | MIT |
| Music scoring | Demucs `htdemucs` | MIT |
| VAD | Silero VAD | MIT |
| Quality | TorchAudio-SQUIM | BSD |

Why these, as of late 2026:

* **Diarization — DiariZen-Large-s80-v2** is still the strongest open
  diarizer. Against pyannote's open `community-1` pipeline on the benchmarks
  both publish (DER, no collar): AMI-SDM 13.9 vs 19.9, AliMeeting 10.8 vs 20.3,
  MSDWild 15.8 vs 22.8, DIHARD3 14.5 vs 20.2, VoxConverse 9.1 vs 11.2.
  pyannote's `precision-2` is better again but runs only on pyannoteAI's cloud;
  NVIDIA Sortformer is capped at four speakers. The weights are
  **non-commercial**; `community-1` (CC BY 4.0) is the permissive alternative,
  and `stages/diarize.py` talks to a generic worker contract, so swapping is a
  configuration change rather than a rewrite.
* **Speaker embedding — WeSpeaker ResNet293** replaced ECAPA-TDNN. Measured on
  this project's Saudi podcast audio (2,185 clips, 18 episodes, DiariZen's
  within-episode labels as ground truth): equal error rate 2.94% → 2.32% within
  episodes, and 2.72% → 1.58% host-versus-guest across episodes — the comparison
  speaker linking depends on. It runs on the GPU from the official checkpoint
  through a small local port of the network (`yt2ds.embedders`), verified
  identical to WeSpeaker's ONNX export. The distance thresholds in
  `speakers.*` are calibrated for it.
* **ASR — Cohere-Speech-Tashkeel-2B** gives diacritized text directly from the
  audio; its parent, Cohere Transcribe Arabic, serves as the independent check.
* **VAD — Silero** stays: no independent benchmark shows a successor (TEN VAD's
  claims are its own) clearly better at speech *boundaries*, which is what
  matters here, and boundary precision is enforced by the segmenter anyway.
* **Vocal isolation — Mel-Band RoFormer.** BS/Mel-RoFormer variants lead the
  music-separation leaderboards; none is benchmarked on speech. For audio with
  no music at all, `--no-separate` avoids separator artefacts entirely.

## Output

```
dataset/
├── wavs/                 # 24 kHz mono 16-bit PCM
│   └── <video_id>_SPK<n>/  # one folder per per-video speaker
├── speakers/             # written by `yt2ds speakers`: one folder per voice
│   ├── report.json
│   ├── GLOBAL_SPEAKER_00/  # wavs/ (hardlinks) + metadata.jsonl + speaker.json
│   └── _unassigned/
├── metadata.jsonl        # one line per kept clip
├── rejected.jsonl        # every dropped clip + reason + scores
├── speakers.json         # per-speaker hours (from `yt2ds report`)
├── manifest.json         # run config, per-video summary
├── failed.txt            # URLs that failed + why; feed back with --urls-file
├── mp3/                  # archival full-length audio, only with --keep-mp3
└── .work/                # state + speaker centroids; safe to delete
```

Each `metadata.jsonl` row carries:

* **text** — `text` (canonical), `text_source`, `text_asr`, `text_verify`,
  `cer_verify`, `text_yt`, `cer_yt_vs_asr`, `asr_language`, `asr_confidence`,
  `verify_confidence`;
* **timing** — `start`/`end` of the clip in the source, `speech_start`/
  `speech_end` of the speech inside it, `voiced_seconds`, `max_internal_pause`;
* **speaker** — `speaker` (per video), `global_speaker`, `speaker_conf`,
  `speaker_distance`, `global_speaker_distance`;
* **level** — `lufs` and `out_peak_dbfs` of the written clip, `gain_db`,
  `gain_limited_db`, and `source_lufs` of the whole episode;
* every quality score behind the keep/drop decision (`squim_mos`,
  `squim_stoi`, `squim_pesq`, `music_score`, `snr_db`, `chars_per_sec`,
  `script_ratio`, …).

Datasets written before these names keep `text_cohere` / `cer_yt_vs_cohere`;
the scripts read either.

Runs are resumable: finished videos are skipped on a re-run, and adding a URL
to an existing dataset only processes the new video. A video interrupted
mid-write has its partial rows cleared before it is reprocessed, so
`metadata.jsonl` never accumulates duplicates. `.work/` holds only the state
and speaker centroids once a video is done — the bulky decodes are cleaned up
as the run goes.

## Speaker folders

```bash
yt2ds speakers dataset/                     # link voices, write speakers/
yt2ds speakers dataset/ --report-only       # just print the ranking
yt2ds speakers dataset/ --only GLOBAL_SPEAKER_00 --threshold 0.38
```

Diarization names speakers per episode, so a host across a hundred episodes
arrives as a hundred `SPKn` labels. `yt2ds speakers` pools them:

1. Per-episode speaker centroids are clustered across the corpus (average
   linkage, `speakers.link_threshold`, 0.40 for ResNet293). `GLOBAL_SPEAKER_00`
   is the voice with the most audio.
2. An identity pooling under `speakers.min_speaker_seconds` (30 s) is
   diarization debris — an intro voice-over, crosstalk.
3. **Every clip is checked against its voice.** A clip further than
   `speakers.max_clip_distance` (0.45) from the voice's centroid was mislabelled
   by the diarizer and goes to `_unassigned/`. A centroid can be right while
   one of its clips is wrong, and one wrong clip is what poisons a cloned voice.

Each `speakers/GLOBAL_SPEAKER_nn/` is a complete dataset root (`wavs/` +
`metadata.jsonl`), ready to train on or upload
(`scripts/upload_speakers_to_hf.py`). Clips are hardlinks, so the split costs no
disk; re-running replaces the previous split. The chosen voice is also written
back into the dataset's own `metadata.jsonl` as `global_speaker`.

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
- **Speaker purity** — every chunk is embedded with WeSpeaker ResNet293 and
  compared to its own speaker's centroid; outliers are dropped. Diarization error rate is
  measured over whole recordings, but what matters here is whether *this clip*
  contains *that voice*.
- **Audio quality** — SQUIM estimated MOS / STOI / PESQ, plus clipping, peak
  level and SNR.
- **Text** — the cross-check against the verify model, the subtitle CER gate,
  the script check, word count, ASR confidence, and characters-per-second
  bounds that catch both bled-in and missing text. Rates are counted on
  undiacritized letters per second of speech (measured Arabic speech runs 5–13).

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
YouTube's rolling auto-caption repetition), the chunker's boundary guarantees
(on 200 random conversations), clip rendering (loudness, peak ceiling, fades),
the ASR backends' request/response mapping including long-audio row
regrouping, the speaker-embedding network, and the speaker split (no
credentials or model downloads needed — models and APIs are stubbed).

## Provenance

Downloading YouTube content is subject to YouTube's Terms of Service and the
underlying copyright. Every row records `video_url`, `channel` and
`upload_date` so provenance stays attached to each sample.
