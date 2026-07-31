"""Download audio, subtitles, and metadata with yt-dlp.

Subtitle selection is a two-pass process: probe first to see which Arabic
tracks a video actually has, then download exactly one. That way the caller
knows with certainty whether it got a human-written track or an
auto-generated one -- which matters, because auto-generated Arabic captions
on dialectal speech are frequently wrong, and the pipeline treats them with
more suspicion downstream.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..io import Workspace

log = logging.getLogger(__name__)


@dataclass
class VideoAssets:
    """Everything the download stage produces for one video."""

    video_id: str
    url: str
    title: str = ""
    channel: str = ""
    upload_date: str = ""
    duration: float = 0.0
    audio_path: Path | None = None
    sub_path: Path | None = None
    sub_lang: str | None = None
    # "yt_manual" | "yt_auto" | None
    sub_kind: str | None = None
    info: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _match_lang(available: list[str], patterns: list[str]) -> str | None:
    """Pick the first available language matching the configured patterns.

    Patterns are tried in order, so ``["ar", "ar-SA", "ar.*"]`` prefers plain
    ``ar`` over a regional variant. Auto-translated tracks (``ar-en`` style
    derived captions) are skipped -- they are machine translations of another
    language, not a transcript of the Arabic audio.
    """
    candidates = [c for c in available if not c.endswith("-orig")]
    for pattern in patterns:
        for lang in candidates:
            if lang == pattern or fnmatch.fnmatch(lang, pattern):
                return lang
    return None


def probe(url: str, cfg: Config) -> dict[str, Any] | None:
    """Fetch metadata for a single video without downloading media."""
    import yt_dlp

    opts = _base_opts(cfg)
    opts.update({"skip_download": True, "quiet": True})
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            return ydl.extract_info(url, download=False)
        except Exception as exc:  # noqa: BLE001 - yt-dlp raises many types
            log.error("probe failed for %s: %s", url, exc)
            return None


def expand_urls(urls: list[str], cfg: Config) -> list[str]:
    """Expand playlist and channel URLs into individual video URLs.

    Plain video URLs pass through untouched, so callers can mix single links
    and playlists in one invocation.
    """
    import yt_dlp

    opts = _base_opts(cfg)
    opts.update({"extract_flat": "in_playlist", "skip_download": True, "quiet": True})

    expanded: list[str] = []
    seen: set[str] = set()
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            try:
                info = ydl.extract_info(url, download=False, process=False)
            except Exception as exc:  # noqa: BLE001
                log.error("could not expand %s: %s", url, exc)
                continue
            if info is None:
                continue
            for video_url in _iter_entries(info):
                if video_url not in seen:
                    seen.add(video_url)
                    expanded.append(video_url)
    return expanded


def _iter_entries(info: dict[str, Any]):
    """Walk a possibly nested playlist structure and yield video URLs."""
    if info.get("_type") in (None, "video"):
        vid = info.get("id")
        if vid:
            yield f"https://www.youtube.com/watch?v={vid}"
        return
    for entry in info.get("entries") or []:
        if entry is None:
            continue
        if entry.get("_type") == "url" and entry.get("ie_key") not in (None, "Youtube"):
            # A channel's tab links to sub-playlists; recurse into them lazily.
            continue
        yield from _iter_entries(entry)


def _base_opts(cfg: Config) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": cfg.download.retries,
        "fragment_retries": cfg.download.retries,
        "sleep_interval_requests": cfg.download.sleep_interval,
        "ignoreerrors": False,
    }
    if cfg.download.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.download.cookies_from_browser,)
    return opts


def download(url: str, ws: Workspace, cfg: Config) -> VideoAssets:
    """Download one video's audio plus its best Arabic subtitle track."""
    import yt_dlp

    info = probe(url, cfg)
    if info is None:
        return VideoAssets(video_id="unknown", url=url, error="probe failed")

    video_id = info.get("id") or "unknown"
    assets = VideoAssets(
        video_id=video_id,
        url=info.get("webpage_url") or url,
        title=info.get("title") or "",
        channel=info.get("channel") or info.get("uploader") or "",
        upload_date=info.get("upload_date") or "",
        duration=float(info.get("duration") or 0.0),
        info={k: info.get(k) for k in ("id", "title", "channel", "uploader", "upload_date", "duration", "webpage_url")},
    )

    manual = list((info.get("subtitles") or {}).keys())
    automatic = list((info.get("automatic_captions") or {}).keys())

    sub_lang: str | None = None
    sub_kind: str | None = None
    if cfg.download.prefer_manual_subs:
        sub_lang = _match_lang(manual, cfg.download.sub_langs)
        sub_kind = "yt_manual" if sub_lang else None
    if sub_lang is None:
        sub_lang = _match_lang(automatic, cfg.download.sub_langs)
        sub_kind = "yt_auto" if sub_lang else None
    if sub_lang is None and not cfg.download.prefer_manual_subs:
        sub_lang = _match_lang(manual, cfg.download.sub_langs)
        sub_kind = "yt_manual" if sub_lang else None

    opts = _base_opts(cfg)
    opts.update(
        {
            "format": "bestaudio/best",
            # `paths` is ignored when an outtmpl is absolute, so the subtitle
            # destination has to be spelled out here or the .json3 lands in
            # raw/ next to the audio.
            "outtmpl": {
                "default": str(ws.raw / "%(id)s.%(ext)s"),
                "subtitle": str(ws.subs / "%(id)s.%(ext)s"),
                "infojson": str(ws.raw / "%(id)s.%(ext)s"),
            },
            "writeinfojson": True,
            "skip_download": False,
        }
    )
    if sub_lang:
        opts.update(
            {
                # Request exactly one track so there is no ambiguity about
                # which file on disk is which.
                "writesubtitles": sub_kind == "yt_manual",
                "writeautomaticsub": sub_kind == "yt_auto",
                "subtitleslangs": [sub_lang],
                "subtitlesformat": cfg.download.sub_format,
            }
        )

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            downloaded = ydl.extract_info(assets.url, download=True)
    except Exception as exc:  # noqa: BLE001
        assets.error = f"download failed: {exc}"
        log.error("download failed for %s: %s", url, exc)
        return assets

    audio = _downloaded_path(downloaded) or _find_audio(ws.raw, video_id)
    if audio is None:
        assets.error = "no audio file produced"
        return assets
    assets.audio_path = audio

    if sub_lang:
        sub = _find_subtitle(ws.subs, video_id, sub_lang)
        if sub is None:
            # The track was advertised but did not materialize; fall through
            # with no subtitles rather than failing the video.
            log.warning("subtitle track %s advertised but missing for %s", sub_lang, video_id)
        else:
            assets.sub_path = sub
            assets.sub_lang = sub_lang
            assets.sub_kind = sub_kind

    return assets


# Containers yt-dlp can hand back for a bestaudio selection.
_AUDIO_SUFFIXES = {".m4a", ".webm", ".opus", ".ogg", ".oga", ".mp3", ".aac", ".flac", ".wav", ".mp4", ".mka"}
_SUBTITLE_SUFFIXES = {".json3", ".srv3", ".srv2", ".srv1", ".vtt", ".srt", ".ttml"}


def _downloaded_path(info: dict[str, Any] | None) -> Path | None:
    """Take the media path straight from yt-dlp rather than guessing.

    Globbing the directory is fragile: subtitle and info-json files share the
    video id prefix, and handing a .json3 to ffmpeg fails in a confusing way.
    """
    if not info:
        return None
    for entry in info.get("requested_downloads") or []:
        path = entry.get("filepath") or entry.get("_filename")
        if path and Path(path).exists():
            return Path(path)
    path = info.get("filepath") or info.get("_filename")
    if path and Path(path).exists():
        return Path(path)
    return None


def _find_audio(directory: Path, video_id: str) -> Path | None:
    """Fallback: the newest file with the video id and an audio extension."""
    matches = [
        p
        for p in directory.glob(f"{video_id}.*")
        if p.is_file() and p.suffix.lower() in _AUDIO_SUFFIXES
    ]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def _find_subtitle(directory: Path, video_id: str, lang: str) -> Path | None:
    matches = [
        p
        for p in sorted(directory.glob(f"{video_id}*"))
        if p.is_file() and p.suffix.lower() in _SUBTITLE_SUFFIXES
    ]
    # Prefer the exact language, then any subtitle we ended up with.
    exact = [p for p in matches if f".{lang}." in p.name]
    for candidate in (exact, matches):
        if candidate:
            # json3 carries word-level timings; prefer it over vtt.
            json3 = [p for p in candidate if p.suffix.lower() in (".json3", ".srv3")]
            return (json3 or candidate)[0]
    return None
