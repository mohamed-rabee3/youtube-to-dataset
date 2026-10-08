"""Speaker embeddings: per-chunk purity, and identity linking across videos.

Diarization error rate is measured over whole recordings, but what actually
matters for voice cloning is whether *this clip* contains *that voice*. A
diarizer with an excellent DER can still hand one chunk to the wrong cluster,
and a single contaminated clip is worse than a missing one.

So every surviving chunk is embedded (WeSpeaker ResNet293 by default, see
:mod:`yt2ds.embedders`), a centroid is built per diarization speaker, and chunks far from their own centroid are dropped. The
centroid is computed from the closest-to-median chunks rather than all of them,
so a handful of mislabelled clips cannot drag it toward themselves.

The same embeddings then let ``yt2ds speakers`` merge identities across
videos, so a podcast host appearing in forty episodes becomes one voice with
pooled hours rather than forty separate ones.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..models import ModelRegistry
from .audio import resample
from .segment import Chunk

log = logging.getLogger(__name__)

_EMBED_SR = 16000
# Below this there is too little signal for a stable embedding.
_MIN_SECONDS = 0.8


def embed_batch(
    items: list[tuple[Chunk, np.ndarray]],
    sr: int,
    cfg: Config,
    registry: ModelRegistry,
) -> np.ndarray:
    """Return L2-normalized embeddings, one row per chunk.

    Chunks too short to embed get a zero row, which every consumer treats as
    "no embedding".
    """
    encoder = registry.speaker_encoder
    if not items:
        return np.zeros((0, encoder.dim), dtype=np.float32)

    audio = [resample(samples, sr, _EMBED_SR) for _, samples in items]
    usable = [i for i, a in enumerate(audio) if a.size >= int(_MIN_SECONDS * _EMBED_SR)]
    out = np.zeros((len(items), encoder.dim), dtype=np.float32)
    if usable:
        out[usable] = encoder.embed([audio[i] for i in usable])
    return out


def verify_purity(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
    """Drop chunks that do not match their assigned speaker.

    Returns the per-speaker centroids, which the caller persists for
    cross-video linking.
    """
    centroids: dict[str, np.ndarray] = {}
    if not chunks:
        return centroids

    by_speaker: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        if chunk.reject_reason is None and np.any(embeddings[i]):
            by_speaker.setdefault(chunk.speaker, []).append(i)

    for speaker, indices in by_speaker.items():
        vectors = embeddings[indices]
        centroid = robust_centroid(vectors)
        centroids[speaker] = centroid

        distances = 1.0 - vectors @ centroid
        for idx, distance in zip(indices, distances):
            chunks[idx].scores["speaker_distance"] = round(float(distance), 5)
            chunks[idx].scores["speaker_conf"] = round(float(1.0 - distance), 5)
            if distance > cfg.speakers.max_centroid_distance:
                chunks[idx].reject(f"speaker:distance={distance:.3f}")

        dropped = sum(1 for d in distances if d > cfg.speakers.max_centroid_distance)
        if dropped:
            log.info("%s: dropped %d/%d chunks as off-centroid", speaker, dropped, len(indices))

    return centroids


def robust_centroid(vectors: np.ndarray) -> np.ndarray:
    """Mean of the half of the vectors closest to the provisional mean.

    Trimming before averaging keeps a few contaminated clips from pulling the
    centroid toward themselves and thereby legitimising each other.
    """
    provisional = _l2(vectors.mean(axis=0))
    if len(vectors) < 4:
        return provisional

    distances = 1.0 - vectors @ provisional
    keep = np.argsort(distances)[: max(2, len(vectors) // 2)]
    return _l2(vectors[keep].mean(axis=0))


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def link_across_videos(
    centroids: dict[str, np.ndarray],
    threshold: float,
    weights: dict[str, float] | None = None,
) -> dict[str, str]:
    """Cluster per-video speakers into global identities.

    Agglomerative clustering with average linkage on cosine distance. Keys are
    the namespaced ``"<video_id>_SPK<n>"`` labels; the returned mapping sends
    each to a ``GLOBAL_SPEAKER_xx`` label.

    ``weights`` maps a speaker key to its kept seconds. When given, global
    labels are numbered by pooled duration, so ``GLOBAL_SPEAKER_00`` is the
    voice with the most usable audio -- which is the ordering that matters when
    choosing whom to clone. Without it, numbering falls back to cluster size,
    which is meaningless when every cluster holds one speaker.
    """
    if not centroids:
        return {}

    keys = sorted(centroids)
    if len(keys) == 1:
        return {keys[0]: "GLOBAL_SPEAKER_00"}

    from sklearn.cluster import AgglomerativeClustering

    matrix = np.stack([centroids[k] for k in keys])
    distances = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    ).fit(distances)

    labels = clustering.labels_

    def rank(label: int) -> tuple[float, int]:
        members = [key for key, lbl in zip(keys, labels) if lbl == label]
        if weights:
            return (-sum(weights.get(k, 0.0) for k in members), -len(members))
        return (-float(len(members)), -len(members))

    order = sorted(set(labels), key=rank)
    renumbered = {lbl: f"GLOBAL_SPEAKER_{i:02d}" for i, lbl in enumerate(order)}
    return {key: renumbered[label] for key, label in zip(keys, labels)}
