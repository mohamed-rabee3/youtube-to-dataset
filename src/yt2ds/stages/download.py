"""Download audio, subtitles, and metadata with yt-dlp.

Subtitle selection is a two-pass process: probe first to see which Arabic
tracks a video actually has, then download exactly one. That way the caller
knows with certainty whether it got a human-written track or an
auto-generated one -- which matters, because auto-generated Arabic captions
on dialectal speech are frequently wrong, and the pipeline treats them with
more suspicion downstream.

Every network call goes through `_extract`, which retries the *whole*
extraction under a different YouTube player client each time. yt-dlp's own
`retries` only re-issues the same request to the same client, which does
nothing for the failures that actually matter here -- a bot check or a 403 on
the media URL is a property of the client YouTube answered, so the fix is to
ask as something else.
"""

from __future__ import annotations

import fnmatch
import logging
import time
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
    # A file that was already on disk rather than something we fetched. The
    # pipeline must never delete it during cleanup.
    is_local: bool = False


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


def video_id_from_url(url: str) -> str | None:
    """Extract a video id from a URL without touching the network.

    Used to answer "have I already done this one?" before paying for a probe
    and a download, so re-running a long list only works on what is new.
    Returns None for anything unrecognised, in which case the caller falls
    back to downloading and checking the id yt-dlp reports.
    """
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate or None
    if host not in ("youtube.com", "music.youtube.com", "youtube-nocookie.com"):
        return None
    if parsed.path == "/watch":
        values = parse_qs(parsed.query).get("v")
        return values[0] if values else None
    for prefix in ("/shorts/", "/embed/", "/live/", "/v/"):
        if parsed.path.startswith(prefix):
            candidate = parsed.path[len(prefix) :].split("/")[0]
            return candidate or None
    return None


def source_id(spec: str) -> str | None:
    """Id for a source that may be a URL or a local path.

    The resume check runs before anything is fetched, so it needs an id for
    both kinds of input.
    """
    path = local_source(spec)
    return local_id(path) if path is not None else video_id_from_url(spec)


def probe(url: str, cfg: Config) -> dict[str, Any] | None:
    """Fetch metadata for a single video without downloading media."""
    info, error = _extract(url, cfg, {"skip_download": True}, download=False)
    if info is None:
        log.error("probe failed for %s: %s", url, error)
    return info


def expand_urls(urls: list[str], cfg: Config, max_depth: int = 2) -> list[str]:
    """Expand playlist and channel URLs into individual video URLs.

    Plain video URLs pass through untouched, so callers can mix single links
    and playlists in one invocation. Entries a flat listing already reports as
    private, deleted or upcoming are dropped here rather than becoming
    guaranteed download failures later.

    A listing can contain listings -- a channel's `/playlists` tab is a page of
    playlist links, not videos -- so anything that comes back still pointing at
    a container is queued and expanded in turn, up to ``max_depth``.

    Local paths are the other kind of container: a directory expands to the
    audio files inside it, a file stands for itself, and neither touches the
    network.
    """
    from collections import deque

    expanded: list[str] = []
    seen: set[str] = set()
    remote: list[str] = []

    for spec in urls:
        path = local_source(spec)
        if path is None:
            remote.append(spec)
            continue
        files = expand_local(path)
        if not files:
            log.error("no audio files found in %s", path)
        for file in files:
            item = str(file)
            if item not in seen:
                seen.add(item)
                expanded.append(item)
        if path.is_dir():
            log.info("%s -> %d local file(s)", path, len(files))

    extra = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        # One dead entry must not abort a 200-video playlist.
        "ignoreerrors": True,
        # `_base_opts` pins this on so the *download* of one video can never
        # drag in a playlist. Expansion is the one place that has to see it.
        "noplaylist": not cfg.download.follow_playlist,
    }

    queue: deque[tuple[str, int]] = deque((url, 0) for url in remote)
    visited: set[str] = set(remote)

    while queue:
        url, depth = queue.popleft()
        info, error = _extract(url, cfg, extra, download=False, process=False)
        if info is None:
            log.error("could not expand %s: %s", url, error)
            continue

        before = len(expanded)
        for item in _iter_entries(info):
            if video_id_from_url(item) is None:
                # A nested container. Depth is bounded so a channel cannot walk
                # into an unbounded tree of related listings.
                if depth < max_depth and item not in visited:
                    visited.add(item)
                    queue.append((item, depth + 1))
                continue
            if item not in seen:
                seen.add(item)
                expanded.append(item)

        found = len(expanded) - before
        if found != 1 or video_id_from_url(url) is None:
            log.info("%s -> %d video(s)", url, found)
    return expanded


# Placeholder titles YouTube uses for entries that cannot be downloaded.
_DEAD_ENTRY_TITLES = ("[private video]", "[deleted video]", "[unavailable video]")


def _iter_entries(info: dict[str, Any]):
    """Walk a flat listing and yield video URLs, plus any nested listing URLs.

    Flat extraction (``process=False``) reports playlist members as ``url``
    entries carrying only an id -- that is the whole point of it, and those
    entries are what a playlist expands to. A ``url`` entry whose extractor is
    ``YoutubeTab`` is another listing rather than a video; it is yielded as-is
    so the caller can expand it separately.
    """
    kind = info.get("_type")

    if kind in (None, "video", "url", "url_transparent"):
        if kind in ("url", "url_transparent") and info.get("ie_key") not in (None, "Youtube"):
            nested = info.get("url")
            if nested:
                yield nested
            return
        if str(info.get("title") or "").strip().lower() in _DEAD_ENTRY_TITLES:
            return
        if info.get("live_status") in ("is_upcoming", "is_live"):
            log.info("skipping live/upcoming video %s", info.get("id"))
            return
        vid = info.get("id")
        if vid:
            yield f"https://www.youtube.com/watch?v={vid}"
        return

    for entry in info.get("entries") or []:
        if entry is not None:
            yield from _iter_entries(entry)


def _base_opts(cfg: Config) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": cfg.download.retries,
        "fragment_retries": cfg.download.retries,
        "extractor_retries": cfg.download.retries,
        "file_access_retries": cfg.download.retries,
        # Back off rather than hammering: a burst of immediate retries is what
        # turns a transient throttle into a sustained block.
        "retry_sleep_functions": {
            key: (lambda n: min(2**n, 60)) for key in ("http", "fragment", "file_access", "extractor")
        },
        "socket_timeout": cfg.download.socket_timeout,
        "sleep_interval_requests": cfg.download.sleep_interval,
        # A watch URL that carries &list= must not drag in the playlist.
        "noplaylist": True,
        # Resume a half-finished .part instead of starting the file again.
        "continuedl": True,
        "ignoreerrors": False,
    }
    if cfg.download.cookies_file:
        opts["cookiefile"] = str(cfg.download.cookies_file)
    if cfg.download.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.download.cookies_from_browser,)
    return opts


# Failures no player client and no amount of waiting will fix. Retrying these
# just multiplies the wait before the run moves on.
_PERMANENT = (
    "private video",
    "members-only",
    "members only",
    "join this channel",
    "removed by the user",
    "removed by the uploader",
    "has been terminated",
    "video has been removed",
    "violat",
    "not available in your country",
    "blocked it on copyright grounds",
    "this live event will begin",
    "premieres in",
    "unable to extract video id",
    "is not a valid url",
    "unsupported url",
)


def _is_permanent(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _PERMANENT)


def _client_sequence(cfg: Config) -> list[str]:
    """Player clients to try, in order, deduplicated."""
    seen: set[str] = set()
    clients = [c for c in cfg.download.player_clients if c and not (c in seen or seen.add(c))]
    return clients or ["default"]


def _extract(
    url: str,
    cfg: Config,
    extra: dict[str, Any],
    *,
    download: bool,
    process: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run one yt-dlp extraction, retrying under successive player clients.

    Returns ``(info, None)`` on success or ``(None, message)`` once every
    client has been tried -- the caller decides whether that is fatal.
    """
    import yt_dlp

    clients = _client_sequence(cfg)
    last_error = "no attempt made"

    for attempt, client in enumerate(clients, start=1):
        opts = _base_opts(cfg)
        opts.update(extra)
        if client != "default":
            opts["extractor_args"] = {"youtube": {"player_client": [client]}}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=download, process=process)
            if info is not None:
                if attempt > 1:
                    log.info("%s: succeeded on attempt %d (client=%s)", url, attempt, client)
                return info, None
            # ignoreerrors turns a failed extraction into a None return.
            last_error = "extraction returned nothing"
        except Exception as exc:  # noqa: BLE001 - yt-dlp raises many types
            last_error = str(exc).strip() or exc.__class__.__name__
            if _is_permanent(last_error):
                log.warning("%s: %s (not retrying)", url, last_error)
                return None, last_error

        if attempt < len(clients):
            delay = cfg.download.attempt_backoff * attempt
            log.warning(
                "%s: attempt %d/%d failed (client=%s): %s -- retrying in %.0fs",
                url,
                attempt,
                len(clients),
                client,
                last_error,
                delay,
            )
            time.sleep(delay)

    return None, last_error


def download(url: str, ws: Workspace, cfg: Config) -> VideoAssets:
    """Download one video's audio plus its best Arabic subtitle track."""
    info, probe_error = _extract(url, cfg, {"skip_download": True}, download=False)
    if info is None:
        return VideoAssets(
            video_id=video_id_from_url(url) or "unknown",
            url=url,
            error=f"probe failed: {probe_error}",
        )

    video_id = info.get("id") or "unknown"
    if cfg.download.skip_live and (info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming")):
        return VideoAssets(video_id=video_id, url=url, error="live or upcoming stream; skipped")

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

    extra: dict[str, Any] = {
        # Prefer a plain audio-only stream, but fall back all the way to a
        # progressive video file rather than failing: some videos expose no
        # DASH audio to the client that answered.
        "format": "bestaudio[acodec!=none]/bestaudio/best[acodec!=none]/best",
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
    if sub_lang:
        extra.update(
            {
                # Request exactly one track so there is no ambiguity about
                # which file on disk is which.
                "writesubtitles": sub_kind == "yt_manual",
                "writeautomaticsub": sub_kind == "yt_auto",
                "subtitleslangs": [sub_lang],
                "subtitlesformat": cfg.download.sub_format,
            }
        )

    downloaded, error = _extract(assets.url, cfg, extra, download=True)
    if downloaded is None and sub_lang:
        # The audio is the dataset; the captions are a second opinion. Never
        # lose a video because its subtitle track would not come down.
        log.warning("%s: retrying without subtitles after: %s", video_id, error)
        for key in ("writesubtitles", "writeautomaticsub", "subtitleslangs", "subtitlesformat"):
            extra.pop(key, None)
        sub_lang = sub_kind = None
        downloaded, error = _extract(assets.url, cfg, extra, download=True)
    if downloaded is None:
        assets.error = f"download failed: {error}"
        log.error("download failed for %s: %s", url, error)
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


def local_source(spec: str) -> Path | None:
    """Return the path if ``spec`` names an existing local file or directory.

    This is what lets a corpus already on disk be fed to the same pipeline as
    a YouTube link. Anything carrying a scheme other than ``file://`` is a URL
    and is left to yt-dlp, so a link is never mistaken for a path.
    """
    if spec.startswith("file://"):
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        spec = url2pathname(urlparse(spec).path)
    elif "://" in spec:
        return None

    path = Path(spec).expanduser()
    return path if path.exists() else None


def _natural_key(path: Path) -> list[Any]:
    """Sort ``2.mp3`` before ``10.mp3``, which plain lexical order does not."""
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def expand_local(path: Path) -> list[Path]:
    """A directory expands to the audio files in it; a file is itself."""
    if not path.is_dir():
        return [path.resolve()]
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _AUDIO_SUFFIXES]
    return [p.resolve() for p in sorted(files, key=_natural_key)]


def local_id(path: Path) -> str:
    """Dataset id for a local file: its stem, reduced to id-safe characters."""
    import re

    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", path.stem).strip("_")
    return stem or "local"


def local_assets(path: Path) -> VideoAssets:
    """Build the assets for a file already on disk -- no network, no subtitles.

    ``url`` is the path itself so the record in ``metadata.jsonl`` says where
    the audio came from, and so a line in ``failed.txt`` can be fed straight
    back in on a retry.
    """
    path = path.resolve()
    return VideoAssets(
        video_id=local_id(path),
        url=str(path),
        title=path.stem,
        audio_path=path,
        info={"id": local_id(path), "title": path.stem, "source": "local"},
        is_local=True,
    )


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
