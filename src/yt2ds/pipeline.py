"""Per-video orchestration.

Stage order is not arbitrary. The audio is segmented *before* it is
transcribed -- then each chunk's transcript is simply that chunk's text, with
no alignment step able to go wrong. Forced alignment is needed only to map the
YouTube subtitle text onto chunk boundaries. That order is also what lets the
ASR backend be swapped freely: neither Cohere nor Cloud Speech-to-Text has to
produce usable timestamps, because nothing downstream reads them.

The expensive detectors run in an order chosen so each one only sees what
survived the last: music rejection before ASR means no musical interlude is
ever transcribed, and speaker purity before ASR means neither the GPU nor the
API bill is spent on a chunk that is about to be dropped anyway.

Chunk audio is read from disk in batches rather than all at once; a two-hour
video can produce a thousand chunks, and holding them all as float arrays would
not fit.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import Config
from .io import JsonlWriter, Workspace, drop_video_records, write_json
from .models import ModelRegistry
from .stages import align, asr, audio, diarize, download, filters, music, quality, segment, separate, speakers, subtitles, vad
from .stages.segment import Chunk

log = logging.getLogger(__name__)

# How many chunks are held in memory at once.
BATCH = 8


@dataclass
class VideoResult:
    video_id: str
    url: str
    kept: int = 0
    rejected: int = 0
    seconds_kept: float = 0.0
    speakers: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    skipped: bool = False


class Pipeline:
    def __init__(self, ws: Workspace, cfg: Config) -> None:
        self.ws = ws.create()
        self.cfg = cfg
        self.registry = ModelRegistry(cfg)

    @property
    def defers_asr(self) -> bool:
        """True when transcription is a separate pass over the whole corpus.

        The google_batch backend transcribes whole episodes through a
        long-running cloud operation, so the GPU pass emits chunks with no text
        and `yt2ds transcribe` fills them in afterwards.
        """
        return (self.cfg.asr.backend or "").lower() == "google_batch"

    # ------------------------------------------------------------------
    def run(self, urls: list[str], resume: bool = True) -> list[VideoResult]:
        video_urls = download.expand_urls(urls, self.cfg)
        if not video_urls:
            log.info("no videos to process")
            return []

        results: list[VideoResult] = []
        pending: list[str] = []
        for url in video_urls:
            video_id = download.source_id(url)
            if resume and video_id and self.ws.is_complete(video_id):
                results.append(self._already_done(video_id, url))
            else:
                pending.append(url)

        done = len(results)
        if done:
            log.info("%d video(s): %d already complete, %d to process", len(video_urls), done, len(pending))
        else:
            log.info("%d video(s) to process", len(video_urls))

        # One video is processed start to finish before the next begins, so
        # each one lands in the dataset as it completes rather than at the end
        # of the run. Downloads run a little ahead so the GPU is not left
        # waiting on the network.
        for position, (url, assets) in enumerate(self._stream_downloads(pending), start=1):
            if assets is None or assets.error:
                results.append(
                    VideoResult(
                        video_id=(assets.video_id if assets else None) or download.source_id(url) or "unknown",
                        url=url,
                        error=assets.error if assets else "download failed",
                    )
                )
                self._write_failed(results)
                self._write_manifest(results)
                continue

            log.info("[%d/%d] %s -- %s", position, len(pending), assets.video_id, assets.title or url)
            try:
                results.append(self._process(assets, resume))
            except Exception as exc:  # noqa: BLE001 - one bad video must not stop the run
                log.exception("failed processing %s", assets.video_id)
                results.append(VideoResult(video_id=assets.video_id, url=assets.url, error=str(exc)))
            # The manifest is rewritten after every video, so a run that is
            # interrupted still describes what it managed to finish.
            self._write_manifest(results)
            self._write_failed(results)

        self._write_manifest(results)
        self._write_failed(results)
        return results

    def _write_failed(self, results: list[VideoResult]) -> None:
        """List the URLs that did not make it, ready to be fed back in.

        A failed video saves no state, so `--urls-file failed.txt` on the next
        run retries exactly those and nothing else. Removed when the list is
        empty, so its presence is the signal.
        """
        failed = [r for r in results if r.error]
        if not failed:
            self.ws.failed.unlink(missing_ok=True)
            return
        lines = [f"# {r.video_id}: {r.error}\n{r.url}" for r in failed]
        self.ws.failed.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _already_done(self, video_id: str, url: str) -> VideoResult:
        """Rebuild a result from saved state, without downloading anything."""
        state = self.ws.load_state(video_id)
        log.info("%s: already complete, skipping", video_id)
        return VideoResult(
            video_id=video_id,
            url=state.get("url") or url,
            kept=state.get("kept", 0),
            rejected=state.get("rejected", 0),
            seconds_kept=state.get("seconds_kept", 0.0),
            speakers=state.get("speakers", {}),
            reasons=state.get("reasons", {}),
            skipped=True,
        )

    # ------------------------------------------------------------------
    def _stream_downloads(self, urls: list[str]) -> Iterator[tuple[str, download.VideoAssets | None]]:
        """Yield each video as its download finishes, in the order given.

        Downloading the whole list up front would mean a long silence before
        any video is processed, every raw file on disk at once, and nothing to
        show for it if the run is interrupted. Instead only a bounded
        look-ahead is in flight: the network stays busy during GPU work without
        running arbitrarily far ahead of it.
        """
        if not urls:
            return

        lookahead = max(1, self.cfg.runtime.download_workers)
        with ThreadPoolExecutor(max_workers=lookahead) as pool:
            queued: deque[tuple[str, Future[download.VideoAssets | None]]] = deque()
            next_index = 0
            while next_index < len(urls) and len(queued) < lookahead:
                queued.append((urls[next_index], pool.submit(self._download_one, urls[next_index])))
                next_index += 1

            while queued:
                url, future = queued.popleft()
                if next_index < len(urls):
                    queued.append((urls[next_index], pool.submit(self._download_one, urls[next_index])))
                    next_index += 1
                yield url, future.result()

    def _download_one(self, url: str) -> download.VideoAssets | None:
        # A file already on disk skips the whole download stage; everything
        # after it is identical, minus the subtitles no local file has.
        local = download.local_source(url)
        if local is not None:
            return download.local_assets(local)
        try:
            assets = download.download(url, self.ws, self.cfg)
        except Exception as exc:  # noqa: BLE001
            log.error("download error for %s: %s", url, exc)
            return download.VideoAssets(
                video_id=download.source_id(url) or "unknown",
                url=url,
                error=str(exc),
            )
        if assets.error:
            log.error("%s: %s", url, assets.error)
        return assets

    # ------------------------------------------------------------------
    def _process(self, assets: download.VideoAssets, resume: bool) -> VideoResult:
        result = VideoResult(video_id=assets.video_id, url=assets.url)

        # Normally caught before the download, but a URL whose id could not be
        # parsed only reveals it here.
        if resume and self.ws.is_complete(assets.video_id):
            return self._already_done(assets.video_id, assets.url)

        # A previous partial run may have written rows for this video already.
        dropped = drop_video_records(self.ws.metadata, assets.video_id)
        dropped += drop_video_records(self.ws.rejected, assets.video_id)
        if dropped:
            log.info("%s: cleared %d rows from an incomplete previous run", assets.video_id, dropped)

        started = time.time()
        prepared = audio.prepare(assets, self.ws, self.cfg)
        log.info("%s: %.1fs of audio, %.1f LUFS", assets.video_id, prepared.duration, prepared.lufs)

        # Music comes out of the speech here, before anything measures or
        # transcribes it. Everything below this line sees the isolated voice.
        prepared = separate.apply(prepared, self.ws, self.cfg, self.registry)

        subs = subtitles.Subtitles()
        if assets.sub_path is not None:
            subs = subtitles.parse(assets.sub_path, kind=assets.sub_kind or "", lang=assets.sub_lang or "")
            log.info(
                "%s: %d subtitle words (%s, %s)",
                assets.video_id,
                len(subs.words),
                assets.sub_kind,
                assets.sub_lang,
            )

        annotation = self._diarize(prepared, assets.video_id)
        regions = vad.detect(prepared.work_path, self.cfg, self.registry)
        chunks = segment.build(regions, annotation, self.cfg, prepared.duration)
        log.info("%s: %d candidate chunks", assets.video_id, len(chunks))
        if not chunks:
            self._finish(assets, result, prepared, started)
            return result

        # Namespace the diarizer's local labels so they stay distinct across
        # videos before cross-video linking runs.
        for chunk in chunks:
            chunk.speaker = f"{assets.video_id}_SPK{chunk.speaker}"

        self._score_chunks(chunks, prepared, assets.video_id)

        if subs:
            aligned = align.align(prepared.work_path, subs, self.cfg, self.registry)
            log.info("%s: aligned %d/%d subtitle words", assets.video_id, len(aligned), len(subs.words))
            align.assign_to_chunks(chunks, aligned, self.cfg)

        if self.defers_asr:
            # No text yet, so the text gates cannot run and would reject every
            # chunk as empty. Only the audio-side gates apply here; `yt2ds
            # transcribe` runs the text gates once the words arrive.
            quality.apply_gates(chunks, self.cfg)
            self._export_for_batch(prepared, assets.video_id)
        else:
            filters.resolve_text(chunks, assets.sub_kind, self.cfg)
            quality.apply_gates(chunks, self.cfg)

        self._emit(chunks, assets, prepared, result)
        self._finish(assets, result, prepared, started)
        return result

    # ------------------------------------------------------------------
    def _diarize(self, prepared: audio.PreparedAudio, video_id: str) -> diarize.Diarization:
        try:
            annotation = diarize.run(prepared.work_path, self.ws, self.cfg, video_id, self.registry)
        except diarize.DiarizerUnavailable as exc:
            log.error("%s -- continuing with a single-speaker assumption", exc)
            return diarize.single_speaker(prepared.duration)
        except Exception as exc:  # noqa: BLE001
            log.error("diarization failed for %s (%s); assuming one speaker", video_id, exc)
            return diarize.single_speaker(prepared.duration)

        log.info("%s: %d speakers, %d turns", video_id, len(annotation.speakers), len(annotation.turns))
        return diarize.collapse_if_dominant(annotation, self.cfg)

    # ------------------------------------------------------------------
    def _score_chunks(self, chunks: list[Chunk], prepared: audio.PreparedAudio, video_id: str) -> None:
        """Run the GPU detectors over every chunk, batch by batch."""
        sr = prepared.master_sr
        embeddings: list[np.ndarray] = []
        reference: np.ndarray | None = None

        for batch in _batched(chunks, BATCH):
            items = [(c, audio.read_segment(prepared.master_path, sr, c.start, c.end)) for c in batch]

            music.score_batch(items, sr, self.cfg, self.registry)
            quality.score_batch(items, sr, self.cfg, self.registry, reference=reference)
            if reference is None:
                reference = quality.pick_reference(items, sr)

            embeddings.append(speakers.embed_batch(items, sr, self.cfg, self.registry))

            # Only transcribe what is still alive: no point running a 2B model
            # -- or paying for an API call -- over a chunk already rejected as
            # music.
            alive = [(c, a) for c, a in items if c.reject_reason is None]
            if alive and not self.defers_asr:
                texts = asr.transcribe_batch(alive, sr, self.cfg, self.registry)
                for (chunk, _), text in zip(alive, texts):
                    chunk.text_cohere = text

        # The SQUIM MOS reference is only available once the first batch has
        # been scored, so that batch comes back unscored. Backfill it now.
        self._backfill_mos(chunks, prepared, reference)

        if embeddings:
            matrix = np.concatenate(embeddings, axis=0)
            centroids = speakers.verify_purity(chunks, matrix, self.cfg)
            self._save_centroids(video_id, centroids)

    def _backfill_mos(
        self,
        chunks: list[Chunk],
        prepared: audio.PreparedAudio,
        reference: np.ndarray | None,
    ) -> None:
        if reference is None:
            return
        missing = [c for c in chunks if c.reject_reason is None and c.scores.get("squim_mos") is None]
        if not missing:
            return
        sr = prepared.master_sr
        for batch in _batched(missing, BATCH):
            items = [(c, audio.read_segment(prepared.master_path, sr, c.start, c.end)) for c in batch]
            quality.score_batch(items, sr, self.cfg, self.registry, reference=reference)

    def _export_for_batch(self, prepared: audio.PreparedAudio, video_id: str) -> None:
        """Stage this episode's audio for the later batch transcription pass.

        The isolated voice is what gets uploaded, not the original mix: chunks
        are cut from it, so the transcript should describe the same signal. It
        is written now because `_cleanup` is about to delete the working copy.
        """
        from .stages import asr_google_batch

        target = self.ws.asr_audio / f"{video_id}.flac"
        if target.exists():
            return
        try:
            asr_google_batch.export_audio(prepared.work_path, target)
        except Exception as exc:  # noqa: BLE001 - a failed export is retryable
            log.error("%s: could not export audio for batch ASR: %s", video_id, exc)
            return
        log.info("%s: staged %.0f MB for batch transcription", video_id, target.stat().st_size / 1e6)

    def _save_centroids(self, video_id: str, centroids: dict[str, np.ndarray]) -> None:
        """Persist per-speaker centroids for `report --link-speakers`.

        Keys are the already-namespaced speaker labels, so centroids from
        different videos never collide when they are pooled for clustering.
        """
        if not centroids:
            return
        np.savez(self.ws.embeddings / f"{video_id}.npz", **centroids)

    # ------------------------------------------------------------------
    def _emit(
        self,
        chunks: list[Chunk],
        assets: download.VideoAssets,
        prepared: audio.PreparedAudio,
        result: VideoResult,
    ) -> None:
        out_sr = self.cfg.audio.out_sample_rate
        # The MP3 is the untouched mix, so cutting from it would undo the
        # separation. The isolated master wins whenever both are asked for.
        from_mp3 = self.cfg.audio.chunk_from_mp3 and prepared.mp3_path is not None and not prepared.separated
        if self.cfg.audio.chunk_from_mp3 and prepared.separated:
            log.warning("audio.chunk_from_mp3 ignored: chunks are cut from the isolated voice, not the mix")
        source = prepared.mp3_path if from_mp3 else prepared.master_path
        source_sr = 44100 if from_mp3 else prepared.master_sr

        kept_records: list[dict[str, Any]] = []
        rejected_records: list[dict[str, Any]] = []

        for chunk in chunks:
            record = self._record(chunk, assets, prepared, out_sr)
            if chunk.reject_reason is not None:
                record["reject_reason"] = chunk.reject_reason
                rejected_records.append(record)
                continue

            samples = audio.read_segment(source, source_sr, chunk.start, chunk.end)
            samples = audio.apply_gain(samples, prepared.gain_db)
            samples = audio.resample(samples, source_sr, out_sr)
            if samples.size == 0:
                record["reject_reason"] = "audio:empty"
                rejected_records.append(record)
                continue

            audio.write_wav(self.ws.wavs / record["audio_file"], samples, out_sr)
            kept_records.append(record)
            result.seconds_kept += chunk.duration
            result.speakers[chunk.speaker] = result.speakers.get(chunk.speaker, 0.0) + chunk.duration

        with JsonlWriter(self.ws.metadata) as fh:
            fh.write_all(kept_records)
        with JsonlWriter(self.ws.rejected) as fh:
            fh.write_all(rejected_records)

        result.kept = len(kept_records)
        result.rejected = len(rejected_records)
        result.reasons = filters.summarize(chunks)

    def _record(
        self,
        chunk: Chunk,
        assets: download.VideoAssets,
        prepared: audio.PreparedAudio,
        out_sr: int,
    ) -> dict[str, Any]:
        s = chunk.scores
        return {
            # One directory per speaker. The label is already namespaced by
            # video, so a multi-speaker recording lands as one folder per voice
            # in it; `report --link-speakers` then regroups those folders into
            # one per global identity.
            "audio_file": f"{chunk.speaker}/{assets.video_id}_{chunk.index:04d}.wav",
            "video_id": assets.video_id,
            "video_url": assets.url,
            "title": assets.title,
            "channel": assets.channel,
            "upload_date": assets.upload_date,
            "chunk_index": chunk.index,
            "start": round(chunk.start, 3),
            "end": round(chunk.end, 3),
            "duration": round(chunk.duration, 3),
            "speaker": chunk.speaker,
            "global_speaker": None,
            "speaker_conf": s.get("speaker_conf"),
            "text": chunk.text,
            # "pending" marks a row the GPU pass emitted without text, for
            # `yt2ds transcribe` to fill in. It is not a text_source anyone
            # reads as provenance -- that is overwritten when the words land.
            "text_source": "pending" if self.defers_asr else chunk.text_source,
            "text_yt": chunk.text_yt,
            "text_cohere": chunk.text_cohere,
            "cer_yt_vs_cohere": s.get("cer_yt_vs_cohere"),
            "asr_language": s.get("asr_language"),
            "asr_confidence": s.get("asr_confidence"),
            "asr_confidence_alt": s.get("asr_confidence_alt"),
            "align_score": s.get("align_score"),
            "aligned_words": s.get("aligned_words"),
            "boundary_clipped": s.get("boundary_clipped"),
            # Music scores are measured *after* separation, so they describe
            # what the isolator left behind rather than what was in the mix.
            "vocals_isolated": prepared.separated,
            "music_score": s.get("music_score"),
            "music_label": s.get("music_label"),
            "speech_score": s.get("speech_score"),
            "vocal_ratio": s.get("vocal_ratio"),
            "accompaniment_ratio": s.get("accompaniment_ratio"),
            "squim_mos": s.get("squim_mos"),
            "squim_stoi": s.get("squim_stoi"),
            "squim_pesq": s.get("squim_pesq"),
            "squim_si_sdr": s.get("squim_si_sdr"),
            "snr_db": s.get("snr_db"),
            "peak_dbfs": s.get("peak_dbfs"),
            "rms_dbfs": s.get("rms_dbfs"),
            "clipping_ratio": s.get("clipping_ratio"),
            "arabic_ratio": s.get("arabic_ratio"),
            "script_ratio": s.get("script_ratio"),
            "chars_per_sec": s.get("chars_per_sec"),
            "lufs": round(prepared.lufs, 3) if prepared.lufs == prepared.lufs else None,
            "sample_rate": out_sr,
            # The language the transcript is actually in, not an assumption.
            "language": s.get("asr_language") or "ar",
        }

    # ------------------------------------------------------------------
    def _finish(
        self,
        assets: download.VideoAssets,
        result: VideoResult,
        prepared: audio.PreparedAudio,
        started: float,
    ) -> None:
        elapsed = time.time() - started
        log.info(
            "%s: kept %d, rejected %d, %.1f min of audio in %.0fs",
            assets.video_id,
            result.kept,
            result.rejected,
            result.seconds_kept / 60,
            elapsed,
        )
        self.ws.save_state(
            assets.video_id,
            {
                "complete": True,
                "url": assets.url,
                "kept": result.kept,
                "rejected": result.rejected,
                "seconds_kept": result.seconds_kept,
                "speakers": result.speakers,
                "reasons": result.reasons,
                "duration": prepared.duration,
                "elapsed": elapsed,
            },
        )
        # Only after the state file says "complete": a video that failed keeps
        # its download so a retry does not start from the network again.
        self._cleanup(assets, prepared)

    def _cleanup(self, assets: download.VideoAssets, prepared: audio.PreparedAudio) -> None:
        """Delete everything this video needed but the dataset does not.

        Once the clips are written and their transcripts are in
        `metadata.jsonl`, nothing reads the source material again: the download,
        the two decoded masters, the caption file and the info JSON are all
        spent. They dominate disk use -- roughly 0.5 GB per source hour for the
        masters alone -- so a thousand-video run that kept them would need more
        space for its scratch than for its output.

        The MP3 goes with them unless `audio.keep_mp3` asks for the archive.
        What survives is the deliverable plus the speaker centroids and the
        state file, so `report --link-speakers` and resume still work on a
        dataset built over many runs.

        A local source file is never touched: it is the user's own corpus, not
        something this run fetched and can fetch again.
        """
        if self.cfg.runtime.keep_intermediates:
            return

        spent: list[Path | None] = [
            None if assets.is_local else assets.audio_path,
            prepared.master_path,
            prepared.work_path,
            # Present only when the pre-separation decode was kept.
            prepared.source_master_path,
            prepared.source_work_path,
            assets.sub_path,
            self.ws.raw / f"{assets.video_id}.info.json",
        ]
        if not self.cfg.audio.keep_mp3:
            # `prepared.mp3_path` is None when no MP3 was written at all; name
            # the file anyway so an archive left by an earlier run also goes.
            spent.append(prepared.mp3_path or self.ws.mp3 / f"{assets.video_id}.mp3")

        freed = 0
        for path in spent:
            if path is None:
                continue
            try:
                freed += path.stat().st_size
                path.unlink()
            except OSError:  # already gone, or held open -- not worth failing over
                continue
        if freed:
            log.info("%s: freed %.0f MB", assets.video_id, freed / 1e6)

    def _write_manifest(self, results: list[VideoResult]) -> None:
        write_json(
            self.ws.manifest,
            {
                "config": self.cfg.to_dict(),
                "videos": [
                    {
                        "video_id": r.video_id,
                        "url": r.url,
                        "kept": r.kept,
                        "rejected": r.rejected,
                        "seconds_kept": round(r.seconds_kept, 2),
                        "reasons": r.reasons,
                        "error": r.error,
                        "skipped": r.skipped,
                    }
                    for r in results
                ],
                "totals": {
                    "videos": len(results),
                    "kept": sum(r.kept for r in results),
                    "rejected": sum(r.rejected for r in results),
                    "hours_kept": round(sum(r.seconds_kept for r in results) / 3600, 3),
                },
            },
        )


def _batched(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
