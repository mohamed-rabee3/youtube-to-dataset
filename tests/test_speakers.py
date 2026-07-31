"""Speaker purity and cross-video identity linking."""

from __future__ import annotations

import numpy as np
import pytest

from yt2ds.config import Config
from yt2ds.stages.segment import Chunk
from yt2ds.stages.speakers import _robust_centroid, link_across_videos, verify_purity


@pytest.fixture
def cfg():
    return Config.load()


def unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    return v / np.linalg.norm(v)


def chunk(index, speaker):
    return Chunk(index=index, start=float(index), end=float(index) + 3.0, speaker=speaker)


class TestVerifyPurity:
    def test_consistent_speaker_keeps_every_chunk(self, cfg):
        chunks = [chunk(i, "A") for i in range(5)]
        base = unit([1.0, 0.0, 0.0])
        embeddings = np.stack([base] * 5)
        centroids = verify_purity(chunks, embeddings, cfg)
        assert all(c.reject_reason is None for c in chunks)
        assert "A" in centroids

    def test_outlier_chunk_is_dropped(self, cfg):
        chunks = [chunk(i, "A") for i in range(6)]
        embeddings = np.stack([unit([1.0, 0.0, 0.0])] * 5 + [unit([0.0, 1.0, 0.0])])
        verify_purity(chunks, embeddings, cfg)
        assert all(c.reject_reason is None for c in chunks[:5])
        assert chunks[5].reject_reason is not None
        assert chunks[5].reject_reason.startswith("speaker:distance")

    def test_speaker_confidence_is_recorded(self, cfg):
        chunks = [chunk(i, "A") for i in range(4)]
        embeddings = np.stack([unit([1.0, 0.0, 0.0])] * 4)
        verify_purity(chunks, embeddings, cfg)
        assert chunks[0].scores["speaker_conf"] == pytest.approx(1.0, abs=1e-4)

    def test_already_rejected_chunks_are_ignored(self, cfg):
        chunks = [chunk(i, "A") for i in range(4)]
        chunks[0].reject("music:test")
        embeddings = np.stack([unit([1.0, 0.0, 0.0])] * 4)
        verify_purity(chunks, embeddings, cfg)
        assert chunks[0].reject_reason == "music:test"

    def test_zero_embeddings_are_skipped(self, cfg):
        chunks = [chunk(0, "A")]
        embeddings = np.zeros((1, 3), dtype=np.float32)
        assert verify_purity(chunks, embeddings, cfg) == {}

    def test_no_chunks(self, cfg):
        assert verify_purity([], np.zeros((0, 3), dtype=np.float32), cfg) == {}


class TestRobustCentroid:
    def test_trims_contaminated_vectors(self):
        # Six clean vectors and two far ones: the centroid must stay near the
        # clean cluster rather than being dragged between the two.
        clean = [unit([1.0, 0.0, 0.0])] * 6
        dirty = [unit([0.0, 1.0, 0.0])] * 2
        centroid = _robust_centroid(np.stack(clean + dirty))
        assert float(centroid @ unit([1.0, 0.0, 0.0])) > 0.99

    def test_small_sample_uses_the_plain_mean(self):
        vectors = np.stack([unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])])
        centroid = _robust_centroid(vectors)
        assert float(np.linalg.norm(centroid)) == pytest.approx(1.0, abs=1e-5)


class TestLinkAcrossVideos:
    def test_same_voice_in_two_videos_merges(self):
        centroids = {
            "vid1_SPK0": unit([1.0, 0.0, 0.0]),
            "vid2_SPK0": unit([0.99, 0.01, 0.0]),
        }
        linked = link_across_videos(centroids, threshold=0.35)
        assert linked["vid1_SPK0"] == linked["vid2_SPK0"]

    def test_different_voices_stay_separate(self):
        centroids = {
            "vid1_SPK0": unit([1.0, 0.0, 0.0]),
            "vid2_SPK0": unit([0.0, 1.0, 0.0]),
        }
        linked = link_across_videos(centroids, threshold=0.35)
        assert linked["vid1_SPK0"] != linked["vid2_SPK0"]

    def test_labels_are_numbered_by_kept_duration(self):
        """GLOBAL_SPEAKER_00 must be the voice with the most usable audio.

        Numbering by cluster size is meaningless when every cluster holds one
        speaker, which is the common case for unrelated videos.
        """
        centroids = {
            "vid1_SPK0": unit([1.0, 0.0, 0.0]),
            "vid2_SPK0": unit([0.0, 1.0, 0.0]),
            "vid3_SPK0": unit([0.0, 0.0, 1.0]),
        }
        weights = {"vid1_SPK0": 10.0, "vid2_SPK0": 300.0, "vid3_SPK0": 50.0}
        linked = link_across_videos(centroids, threshold=0.35, weights=weights)
        assert linked["vid2_SPK0"] == "GLOBAL_SPEAKER_00"
        assert linked["vid3_SPK0"] == "GLOBAL_SPEAKER_01"
        assert linked["vid1_SPK0"] == "GLOBAL_SPEAKER_02"

    def test_pooled_duration_decides_across_merged_clusters(self):
        # Two videos share one voice; a third has a different, longer one.
        centroids = {
            "vid1_SPK0": unit([1.0, 0.0, 0.0]),
            "vid2_SPK0": unit([0.99, 0.01, 0.0]),
            "vid3_SPK0": unit([0.0, 1.0, 0.0]),
        }
        weights = {"vid1_SPK0": 100.0, "vid2_SPK0": 100.0, "vid3_SPK0": 150.0}
        linked = link_across_videos(centroids, threshold=0.35, weights=weights)
        assert linked["vid1_SPK0"] == "GLOBAL_SPEAKER_00"  # pooled 200s
        assert linked["vid3_SPK0"] == "GLOBAL_SPEAKER_01"  # 150s

    def test_single_speaker(self):
        linked = link_across_videos({"vid1_SPK0": unit([1.0, 0.0, 0.0])}, threshold=0.35)
        assert linked == {"vid1_SPK0": "GLOBAL_SPEAKER_00"}

    def test_empty(self):
        assert link_across_videos({}, threshold=0.35) == {}
