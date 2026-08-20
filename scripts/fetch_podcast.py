"""Download every episode of a podcast RSS feed into a folder.

Apple Podcasts show pages are a directory, not a source: yt-dlp's extractor
handles a single ``?i=`` episode URL and nothing else. The show's real audio
lives in the RSS feed Apple itself points at, which the iTunes lookup API
returns for a numeric show id.

Files are named ``<slug>-NNN.mp3`` by feed position rather than by title,
because ``yt2ds`` derives a local file's dataset id from its stem with
``[^0-9A-Za-z_-] -> _``: an Arabic title reduces to a run of underscores and
101 episodes would collide into one id. The real titles are kept beside the
audio in ``episodes.json`` so nothing is lost.

Downloads are resumable -- a file already on disk with the length the feed
advertises is skipped, so a re-run costs nothing.

    scripts/fetch_podcast.py --apple-id 1585550025 --out nafas --slug nafas
    scripts/fetch_podcast.py --feed https://... --out nafas --slug nafas
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

log = logging.getLogger("fetch_podcast")

ITUNES = "https://itunes.apple.com/lookup?id={id}&entity=podcast"
ITUNES_NS = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"


def feed_url_for(apple_id: str) -> str:
    r = requests.get(ITUNES.format(id=apple_id), timeout=30)
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results or not results[0].get("feedUrl"):
        raise SystemExit(f"no RSS feed for Apple id {apple_id}")
    return results[0]["feedUrl"]


def episodes(feed: str) -> list[dict]:
    r = requests.get(feed, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out = []
    for item in root.findall("./channel/item"):
        enc = item.find("enclosure")
        if enc is None or not enc.get("url"):
            continue
        dur = item.findtext(f"{ITUNES_NS}duration") or ""
        out.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "url": enc.get("url"),
                "length": int(enc.get("length") or 0),
                "duration": dur.strip(),
                "published": (item.findtext("pubDate") or "").strip(),
                "guid": (item.findtext("guid") or "").strip(),
            }
        )
    return out


def fetch(ep: dict, dest: Path) -> str:
    """Download one episode unless a complete copy is already there."""
    if dest.exists() and ep["length"] and dest.stat().st_size == ep["length"]:
        return "skip"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(ep["url"], stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
    tmp.rename(dest)
    return "ok"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apple-id", help="numeric id from a podcasts.apple.com URL")
    p.add_argument("--feed", help="RSS feed URL, if known")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--slug", required=True, help="ASCII filename prefix")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    feed = args.feed or (args.apple_id and feed_url_for(args.apple_id))
    if not feed:
        raise SystemExit("give --feed or --apple-id")
    log.info("feed: %s", feed)

    eps = episodes(feed)
    if args.limit:
        eps = eps[: args.limit]
    log.info("%d episodes in feed", len(eps))

    args.out.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(eps))))
    for i, ep in enumerate(eps, start=1):
        ep["seq"] = i
        ep["file"] = f"{args.slug}-{i:0{width}d}.mp3"

    def work(ep: dict) -> tuple[dict, str]:
        try:
            return ep, fetch(ep, args.out / ep["file"])
        except Exception as exc:  # noqa: BLE001 -- one bad episode must not stop the rest
            log.warning("%s failed: %s", ep["file"], exc)
            return ep, f"fail: {exc}"

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ep, status in pool.map(work, eps):
            done += 1
            ep["status"] = status
            log.info("[%d/%d] %-16s %-5s %s", done, len(eps), ep["file"], status, ep["title"][:60])

    (args.out / "episodes.json").write_text(
        json.dumps({"feed": feed, "episodes": eps}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    bad = [e for e in eps if e["status"].startswith("fail")]
    log.info("downloaded %d, failed %d -> %s", len(eps) - len(bad), len(bad), args.out)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
