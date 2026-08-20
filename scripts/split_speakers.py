"""Pool per-episode speakers into global identities, one folder each.

Diarization names speakers per episode, so a podcast host with 209 episodes
arrives as 209 unrelated ``SPKn`` labels. Clustering the speaker centroids the
pipeline already saved collapses those into one identity per voice, which is
what makes it possible to choose *whose* audio to transcribe rather than paying
to transcribe everyone.

Each global speaker gets a directory holding its own ``metadata.jsonl`` and a
``wavs/`` tree. The clips are **hardlinks**, not copies: the dataset is 41 GB
and a speaker split that duplicated it would cost another 41 GB to say nothing
new. A hardlink costs an inode, deleting a speaker folder never touches the
source clip, and anything reading the folder sees an ordinary wav.

Model note. This clusters the ECAPA centroids written during the run, which
were computed over *every* clip of each speaker. Re-embedding with a heavier
model was measured on this corpus and produced identical clusters -- host on
206/209 episodes either way -- so the centroids already on disk are both the
cheapest and the best-supported input.

    scripts/split_speakers.py dataset-socrates --report-only
    scripts/split_speakers.py dataset-socrates --threshold 0.40
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yt2ds.io import read_jsonl, write_json  # noqa: E402
from yt2ds.stages.speakers import link_across_videos  # noqa: E402

log = logging.getLogger("split_speakers")


def load_centroids(dataset: Path) -> dict[str, np.ndarray]:
    """Every per-episode speaker centroid the run saved."""
    centroids: dict[str, np.ndarray] = {}
    for path in sorted(glob.glob(str(dataset / ".work" / "embeddings" / "*.npz"))):
        with np.load(path) as data:
            for key in data.files:
                centroids[key] = data[key]
    return centroids


def summarize(rows: list[dict], mapping: dict[str, str]) -> dict[str, dict]:
    """Per global speaker: duration, clips, episodes, mean MOS, sample clips."""
    acc: dict[str, dict] = collections.defaultdict(
        lambda: {"seconds": 0.0, "clips": 0, "episodes": set(), "mos": [], "samples": [], "members": set()}
    )
    for row in rows:
        speaker = row.get("speaker")
        target = mapping.get(speaker)
        if not target:
            continue
        entry = acc[target]
        entry["seconds"] += float(row.get("duration") or 0.0)
        entry["clips"] += 1
        entry["episodes"].add(row.get("video_id", "?"))
        entry["members"].add(speaker)
        mos = row.get("squim_mos")
        if mos is not None:
            entry["mos"].append(float(mos))
        # A handful of clips to actually listen to before committing money.
        if len(entry["samples"]) < 5:
            entry["samples"].append(row["audio_file"])

    out: dict[str, dict] = {}
    for name, entry in acc.items():
        out[name] = {
            "seconds": round(entry["seconds"], 1),
            "hours": round(entry["seconds"] / 3600, 2),
            "clips": entry["clips"],
            "episodes": len(entry["episodes"]),
            "mean_mos": round(sum(entry["mos"]) / len(entry["mos"]), 3) if entry["mos"] else None,
            "per_video_speakers": sorted(entry["members"]),
            "samples": entry["samples"],
        }
    return out


def report(summary: dict[str, dict], limit: int) -> None:
    ranked = sorted(summary.items(), key=lambda kv: -kv[1]["seconds"])
    print(f"\n{'global speaker':<22} {'hours':>7} {'clips':>7} {'eps':>5} {'MOS':>6}  members")
    print("-" * 72)
    for name, s in ranked[:limit]:
        mos = f"{s['mean_mos']:.2f}" if s["mean_mos"] is not None else "  -  "
        print(f"{name:<22} {s['hours']:>7.2f} {s['clips']:>7} {s['episodes']:>5} {mos:>6}  {len(s['per_video_speakers'])}")
    if len(ranked) > limit:
        tail = sum(s["seconds"] for _, s in ranked[limit:]) / 3600
        print(f"... {len(ranked) - limit} more, {tail:.1f} h in total")


def build(dataset: Path, rows: list[dict], mapping: dict[str, str], summary: dict[str, dict],
          out_root: Path, only: set[str] | None, min_seconds: float) -> None:
    """Write one folder per global speaker: metadata.jsonl plus hardlinked wavs."""
    wanted = {
        name for name, s in summary.items()
        if s["seconds"] >= min_seconds and (only is None or name in only)
    }
    log.info("writing %d speaker folder(s) under %s", len(wanted), out_root)

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        target = mapping.get(row.get("speaker"))
        if target in wanted:
            grouped[target].append(row)

    for name in sorted(wanted):
        speaker_dir = out_root / name
        (speaker_dir / "wavs").mkdir(parents=True, exist_ok=True)
        lines = []
        linked = 0
        for row in grouped[name]:
            source = dataset / "wavs" / row["audio_file"]
            # Flatten: the per-episode speaker folder is already encoded in the
            # filename, and a flat tree is what most trainers expect.
            target = speaker_dir / "wavs" / Path(row["audio_file"]).name
            if not target.exists():
                if not source.exists():
                    log.warning("missing clip, skipped: %s", source)
                    continue
                try:
                    os.link(source, target)
                except OSError:  # different filesystem, or link limit
                    import shutil

                    shutil.copy2(source, target)
                linked += 1
            entry = dict(row)
            entry["global_speaker"] = name
            entry["source_audio_file"] = row["audio_file"]
            # Relative to the folder's wavs/, matching the parent dataset --
            # everything downstream resolves clips as <dataset>/wavs/<audio_file>.
            entry["audio_file"] = Path(row["audio_file"]).name
            lines.append(entry)

        with (speaker_dir / "metadata.jsonl").open("w", encoding="utf-8") as fh:
            for entry in lines:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        write_json(speaker_dir / "speaker.json", summary[name])
        log.info("%s: %d clip(s), %.2f h, %d newly linked", name, len(lines), summary[name]["hours"], linked)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset", type=Path)
    p.add_argument("--threshold", type=float, default=0.40,
                   help="cosine distance for pooling speakers (default 0.40; the host is stable 0.30-0.50)")
    p.add_argument("--out", type=Path, help="output root (default <dataset>/speakers)")
    p.add_argument("--report-only", action="store_true", help="print the ranking and write nothing")
    p.add_argument("--only", help="comma-separated global speaker ids to materialise")
    p.add_argument("--min-seconds", type=float, default=60.0,
                   help="skip speakers with less usable audio than this")
    p.add_argument("--top", type=int, default=30, help="rows to show in the report")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    dataset = args.dataset
    centroids = load_centroids(dataset)
    if not centroids:
        raise SystemExit(f"no speaker centroids in {dataset}/.work/embeddings -- run the pipeline first")
    rows = [r for r in read_jsonl(dataset / "metadata.jsonl") if r.get("audio_file")]

    weights: dict[str, float] = collections.Counter()
    for row in rows:
        weights[row.get("speaker", "?")] += float(row.get("duration") or 0.0)

    mapping = link_across_videos(centroids, args.threshold, weights=dict(weights))
    summary = summarize(rows, mapping)
    log.info("%d per-episode speaker(s) -> %d global identit(ies) at threshold %.2f",
             len(centroids), len(summary), args.threshold)
    report(summary, args.top)

    out_root = args.out or dataset / "speakers"
    write_json(out_root / "report.json", {"threshold": args.threshold, "speakers": summary})
    print(f"\nfull ranking written to {out_root / 'report.json'}")

    if args.report_only:
        print("report-only: no folders written")
        return 0

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    build(dataset, rows, mapping, summary, out_root, only, args.min_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
