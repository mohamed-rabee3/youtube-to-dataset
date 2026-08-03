"""URL expansion and download resilience.

Both are network-facing, so the yt-dlp boundary is faked: what is under test
is how this code reads a flat playlist listing and how it reacts to a failure,
not yt-dlp itself.
"""

from __future__ import annotations

import pytest

from yt2ds.config import Config
from yt2ds.stages import download as dl


@pytest.fixture
def cfg():
    cfg = Config.load()
    cfg.download.attempt_backoff = 0.0  # no real sleeping in tests
    return cfg


def _flat_entry(video_id: str, **extra):
    """What `extract_flat` hands back for one playlist member."""
    return {
        "_type": "url",
        "ie_key": "Youtube",
        "id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": f"video {video_id}",
        **extra,
    }


class TestIterEntries:
    def test_single_video(self):
        info = {"_type": "video", "id": "abc123"}
        assert list(dl._iter_entries(info)) == ["https://www.youtube.com/watch?v=abc123"]

    def test_playlist_of_flat_entries(self):
        """A flat listing is *all* `url` entries -- dropping them expands to nothing."""
        info = {"_type": "playlist", "entries": [_flat_entry("a1"), _flat_entry("b2")]}
        assert list(dl._iter_entries(info)) == [
            "https://www.youtube.com/watch?v=a1",
            "https://www.youtube.com/watch?v=b2",
        ]

    def test_channel_tabs_are_walked(self):
        """A channel is a playlist of tabs, each a playlist of videos."""
        info = {
            "_type": "playlist",
            "entries": [
                {"_type": "playlist", "id": "UC1", "entries": [_flat_entry("a1")]},
                {"_type": "playlist", "id": "UC1", "entries": [_flat_entry("b2")]},
            ],
        }
        assert [u[-2:] for u in dl._iter_entries(info)] == ["a1", "b2"]

    def test_dead_and_live_entries_are_skipped(self):
        info = {
            "_type": "playlist",
            "entries": [
                _flat_entry("a1"),
                _flat_entry("p1", title="[Private video]"),
                _flat_entry("d1", title="[Deleted video]"),
                _flat_entry("l1", live_status="is_upcoming"),
                _flat_entry("b2"),
            ],
        }
        assert [u[-2:] for u in dl._iter_entries(info)] == ["a1", "b2"]

    def test_nested_listing_is_yielded_for_the_caller_to_expand(self):
        """A channel's /playlists tab is a page of playlists, not of videos."""
        info = {
            "_type": "playlist",
            "entries": [
                {
                    "_type": "url",
                    "ie_key": "YoutubeTab",
                    "id": "PL1",
                    "url": "https://www.youtube.com/playlist?list=PL1",
                },
                _flat_entry("a1"),
            ],
        }
        assert list(dl._iter_entries(info)) == [
            "https://www.youtube.com/playlist?list=PL1",
            "https://www.youtube.com/watch?v=a1",
        ]

    def test_none_entries_do_not_stop_the_walk(self):
        # ignoreerrors turns an unavailable member into a None entry.
        info = {"_type": "playlist", "entries": [None, _flat_entry("a1"), None]}
        assert [u[-2:] for u in dl._iter_entries(info)] == ["a1"]


class TestExpandUrls:
    def test_playlist_expands_and_dedupes(self, cfg, monkeypatch):
        listing = {"_type": "playlist", "entries": [_flat_entry("a1"), _flat_entry("b2"), _flat_entry("a1")]}
        monkeypatch.setattr(dl, "_extract", lambda *a, **k: (listing, None))

        assert [u[-2:] for u in dl.expand_urls(["https://youtube.com/playlist?list=PL1"], cfg)] == ["a1", "b2"]

    def test_nested_listings_are_expanded_in_turn(self, cfg, monkeypatch):
        listings = {
            "https://www.youtube.com/@ch/playlists": {
                "_type": "playlist",
                "entries": [
                    {
                        "_type": "url",
                        "ie_key": "YoutubeTab",
                        "id": "PL1",
                        "url": "https://www.youtube.com/playlist?list=PL1",
                    }
                ],
            },
            "https://www.youtube.com/playlist?list=PL1": {
                "_type": "playlist",
                "entries": [_flat_entry("a1")],
            },
        }
        monkeypatch.setattr(dl, "_extract", lambda url, *a, **k: (listings.get(url), None))

        got = dl.expand_urls(["https://www.youtube.com/@ch/playlists"], cfg)
        assert got == ["https://www.youtube.com/watch?v=a1"]

    def test_recursion_is_depth_bounded(self, cfg, monkeypatch):
        """A listing that links to itself must not spin forever."""
        loop = {
            "_type": "playlist",
            "entries": [
                {"_type": "url", "ie_key": "YoutubeTab", "id": "PL2", "url": "https://x/list=PL2"},
                {"_type": "url", "ie_key": "YoutubeTab", "id": "PL3", "url": "https://x/list=PL3"},
            ],
        }
        calls: list[str] = []

        def fake_extract(url, *a, **k):
            calls.append(url)
            return loop, None

        monkeypatch.setattr(dl, "_extract", fake_extract)
        assert dl.expand_urls(["https://x/list=PL1"], cfg, max_depth=2) == []
        # Each distinct listing is visited once, and only to the given depth.
        assert calls == ["https://x/list=PL1", "https://x/list=PL2", "https://x/list=PL3"]

    def test_one_unreachable_url_does_not_lose_the_others(self, cfg, monkeypatch):
        def fake_extract(url, *a, **k):
            if "bad" in url:
                return None, "HTTP Error 404"
            return {"_type": "video", "id": "ok1"}, None

        monkeypatch.setattr(dl, "_extract", fake_extract)
        urls = ["https://youtube.com/playlist?list=bad", "https://youtu.be/ok1"]
        assert dl.expand_urls(urls, cfg) == ["https://www.youtube.com/watch?v=ok1"]


class TestExtractRetries:
    def test_second_client_is_tried_after_a_failure(self, cfg, monkeypatch):
        cfg.download.player_clients = ["default", "tv_simply", "ios"]
        clients: list[object] = []

        class FakeYDL:
            def __init__(self, opts):
                clients.append(opts.get("extractor_args", {}).get("youtube", {}).get("player_client"))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False, process=True):
                if len(clients) < 2:
                    raise RuntimeError("HTTP Error 403: Forbidden")
                return {"id": "ok"}

        monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)
        info, error = dl._extract("u", cfg, {}, download=False)

        assert (info, error) == ({"id": "ok"}, None)
        assert clients == [None, ["tv_simply"]]

    def test_permanent_failure_is_not_retried(self, cfg, monkeypatch):
        cfg.download.player_clients = ["default", "tv_simply", "ios"]
        attempts = []

        class FakeYDL:
            def __init__(self, opts):
                attempts.append(opts)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False, process=True):
                raise RuntimeError("ERROR: Private video. Sign in if you've been granted access")

        monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)
        info, error = dl._extract("u", cfg, {}, download=False)

        assert info is None and "Private video" in error
        assert len(attempts) == 1, "a private video is not going to become public on retry"

    def test_error_is_returned_after_every_client_fails(self, cfg, monkeypatch):
        cfg.download.player_clients = ["default", "ios"]

        class FakeYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False, process=True):
                raise RuntimeError("Sign in to confirm you're not a bot")

        monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)
        info, error = dl._extract("u", cfg, {}, download=False)
        assert info is None and "not a bot" in error


class TestClientSequence:
    def test_duplicates_are_dropped_and_order_kept(self, cfg):
        cfg.download.player_clients = ["default", "ios", "default", "tv"]
        assert dl._client_sequence(cfg) == ["default", "ios", "tv"]

    def test_empty_falls_back_to_default(self, cfg):
        cfg.download.player_clients = []
        assert dl._client_sequence(cfg) == ["default"]
