"""Multi-video run behaviour: ordering, resume, and accumulation."""

from __future__ import annotations

import json

import pytest

from yt2ds.config import Config
from yt2ds.io import Workspace, drop_video_records, read_jsonl
from yt2ds.pipeline import Pipeline
from yt2ds.stages.download import VideoAssets, video_id_from_url


@pytest.fixture
def cfg():
    return Config.load()


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "dataset").create()


class TestVideoIdFromUrl:
    """Resume has to recognise a finished video before paying for a download."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.youtube.com/watch?v=1i7qMjp63jw", "1i7qMjp63jw"),
            ("https://youtube.com/watch?v=abc123&t=90s", "abc123"),
            ("https://m.youtube.com/watch?v=abc123", "abc123"),
            ("https://youtu.be/1i7qMjp63jw", "1i7qMjp63jw"),
            ("https://youtu.be/1i7qMjp63jw?t=42", "1i7qMjp63jw"),
            ("https://www.youtube.com/shorts/xyz789", "xyz789"),
            ("https://www.youtube.com/embed/xyz789", "xyz789"),
            ("https://www.youtube.com/live/xyz789", "xyz789"),
        ],
    )
    def test_recognised_forms(self, url, expected):
        assert video_id_from_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/@somechannel",
            "https://www.youtube.com/playlist?list=PL123",
            "https://vimeo.com/12345",
            "not a url",
        ],
    )
    def test_unrecognised_forms_return_none(self, url):
        # None is not a failure: the caller downloads and asks yt-dlp instead.
        assert video_id_from_url(url) is None


class TestStreamDownloads:
    def test_videos_are_yielded_in_order(self, ws, cfg):
        pipeline = Pipeline(ws, cfg)
        urls = [f"https://www.youtube.com/watch?v=vid{i}" for i in range(7)]
        pipeline._download_one = lambda url: VideoAssets(video_id=url[-4:], url=url)

        seen = [assets.video_id for _, assets in pipeline._stream_downloads(urls)]
        assert seen == [f"vid{i}" for i in range(7)]

    def test_download_runs_no_further_ahead_than_the_lookahead(self, ws, cfg):
        """The whole list must not be fetched before the first video is handed back.

        This is what makes the run one-by-one: raw files on disk stay bounded,
        and an interrupted run has not wasted bandwidth on videos it never
        reached.
        """
        cfg.runtime.download_workers = 2
        pipeline = Pipeline(ws, cfg)
        urls = [f"https://www.youtube.com/watch?v=vid{i}" for i in range(10)]

        started: list[str] = []

        def fake_download(url):
            started.append(url)
            return VideoAssets(video_id=url[-4:], url=url)

        pipeline._download_one = fake_download

        stream = pipeline._stream_downloads(urls)
        next(stream)
        # Two in flight, plus the one submitted when the first was consumed.
        assert len(started) <= 3, f"downloaded {len(started)} of 10 before processing one"

    def test_failed_download_is_reported_not_skipped(self, ws, cfg):
        pipeline = Pipeline(ws, cfg)
        urls = ["https://www.youtube.com/watch?v=good", "https://www.youtube.com/watch?v=bad0"]
        pipeline._download_one = lambda url: None if "bad0" in url else VideoAssets(video_id="good", url=url)

        out = list(pipeline._stream_downloads(urls))
        assert out[0][1] is not None
        assert out[1][1] is None  # surfaced, so run() can record the error

    def test_empty_list(self, ws, cfg):
        assert list(Pipeline(ws, cfg)._stream_downloads([])) == []


class TestResumeAcrossRuns:
    def test_finished_video_is_not_downloaded_again(self, ws, cfg, monkeypatch):
        from yt2ds.pipeline import VideoResult
        from yt2ds.stages import download as dl

        ws.save_state("done0", {"complete": True, "kept": 4, "rejected": 1, "seconds_kept": 20.0})
        monkeypatch.setattr(dl, "expand_urls", lambda urls, cfg: urls)

        pipeline = Pipeline(ws, cfg)
        downloaded: list[str] = []

        def fake_download(url):
            downloaded.append(url)
            return VideoAssets(video_id=video_id_from_url(url) or "?", url=url)

        monkeypatch.setattr(pipeline, "_download_one", fake_download)
        monkeypatch.setattr(
            pipeline,
            "_process",
            lambda assets, resume: VideoResult(video_id=assets.video_id, url=assets.url),
        )

        results = pipeline.run(
            [
                "https://www.youtube.com/watch?v=done0",
                "https://www.youtube.com/watch?v=new01",
            ]
        )

        assert downloaded == ["https://www.youtube.com/watch?v=new01"]
        assert [r.video_id for r in results] == ["done0", "new01"]
        assert results[0].skipped and results[0].kept == 4
        assert not results[1].skipped

    def test_state_carries_the_counts_forward(self, ws, cfg):
        ws.save_state("v0", {"complete": True, "kept": 7, "rejected": 3, "seconds_kept": 42.5})
        result = Pipeline(ws, cfg)._already_done("v0", "https://www.youtube.com/watch?v=v0")
        assert (result.kept, result.rejected, result.seconds_kept) == (7, 3, 42.5)
        assert result.skipped


class TestAccumulation:
    """Rows from many videos pile up in one metadata.jsonl without duplicates."""

    def test_rows_from_later_videos_append_to_earlier_ones(self, ws):
        from yt2ds.io import JsonlWriter

        with JsonlWriter(ws.metadata) as fh:
            fh.write_all([{"video_id": "v1", "audio_file": "v1_0000.wav"}])
        with JsonlWriter(ws.metadata) as fh:
            fh.write_all([{"video_id": "v2", "audio_file": "v2_0000.wav"}])

        rows = list(read_jsonl(ws.metadata))
        assert [r["video_id"] for r in rows] == ["v1", "v2"]

    def test_reprocessing_one_video_leaves_the_others_untouched(self, ws):
        from yt2ds.io import JsonlWriter

        with JsonlWriter(ws.metadata) as fh:
            fh.write_all(
                [
                    {"video_id": "v1", "audio_file": "v1_0000.wav"},
                    {"video_id": "v2", "audio_file": "v2_0000.wav"},
                    {"video_id": "v1", "audio_file": "v1_0001.wav"},
                ]
            )

        removed = drop_video_records(ws.metadata, "v1")
        assert removed == 2
        rows = list(read_jsonl(ws.metadata))
        assert [r["video_id"] for r in rows] == ["v2"]


class TestCleanup:
    def test_intermediates_are_deleted_once_a_video_is_complete(self, ws, cfg, tmp_path):
        from yt2ds.stages.audio import PreparedAudio

        raw = tmp_path / "raw.webm"
        master = tmp_path / "v.48k.wav"
        work = tmp_path / "v.16k.wav"
        mp3 = tmp_path / "v.mp3"
        for path in (raw, master, work, mp3):
            path.write_bytes(b"x" * 16)

        assets = VideoAssets(video_id="v", url="u", audio_path=raw)
        prepared = PreparedAudio("v", master, work, mp3, 10.0, 48000, 16000, -20.0, 3.0)

        Pipeline(ws, cfg)._cleanup(assets, prepared)

        assert not raw.exists() and not master.exists() and not work.exists()
        assert mp3.exists(), "the archival MP3 is a deliverable, not an intermediate"

    def test_keep_intermediates_preserves_everything(self, ws, cfg, tmp_path):
        from yt2ds.stages.audio import PreparedAudio

        cfg.runtime.keep_intermediates = True
        raw = tmp_path / "raw.webm"
        raw.write_bytes(b"x")
        master = tmp_path / "v.48k.wav"
        master.write_bytes(b"x")
        work = tmp_path / "v.16k.wav"
        work.write_bytes(b"x")

        assets = VideoAssets(video_id="v", url="u", audio_path=raw)
        prepared = PreparedAudio("v", master, work, None, 10.0, 48000, 16000, -20.0, 3.0)
        Pipeline(ws, cfg)._cleanup(assets, prepared)

        assert raw.exists() and master.exists() and work.exists()

    def test_missing_files_do_not_raise(self, ws, cfg, tmp_path):
        from yt2ds.stages.audio import PreparedAudio

        assets = VideoAssets(video_id="v", url="u", audio_path=None)
        prepared = PreparedAudio(
            "v", tmp_path / "gone.wav", tmp_path / "gone2.wav", None, 10.0, 48000, 16000, -20.0, 3.0
        )
        Pipeline(ws, cfg)._cleanup(assets, prepared)  # must not raise


class TestManifest:
    def test_manifest_is_written_after_each_video(self, ws, cfg):
        from yt2ds.pipeline import VideoResult

        pipeline = Pipeline(ws, cfg)
        pipeline._write_manifest([VideoResult(video_id="v1", url="u1", kept=3)])
        data = json.loads(ws.manifest.read_text(encoding="utf-8"))
        assert data["totals"]["kept"] == 3
        assert [v["video_id"] for v in data["videos"]] == ["v1"]
