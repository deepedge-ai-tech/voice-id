"""Tests for the AS-Norm core module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.wespeaker_deep_edge.asnorm import (
    CohortCache,
    CohortData,
    apply_asnorm,
    top_k_mean_std,
)

# ============================================================================ #
#  Fixtures
# ============================================================================ #


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def cohort_embs(rng: np.random.Generator) -> np.ndarray:
    """Random cohort embeddings: (30, 256)."""
    return rng.uniform(-1.0, 1.0, size=(30, 256)).astype(np.float32)


@pytest.fixture
def enroll_matrix(rng: np.random.Generator) -> np.ndarray:
    """Random enrollment embeddings: (7, 256)."""
    return rng.uniform(-1.0, 1.0, size=(7, 256)).astype(np.float32)


@pytest.fixture
def test_emb(rng: np.random.Generator) -> np.ndarray:
    """Random test embedding: (256,)."""
    return rng.uniform(-1.0, 1.0, size=(256,)).astype(np.float32)


@pytest.fixture
def test_emb_same_speaker(enroll_matrix: np.ndarray) -> np.ndarray:
    """Test embedding identical to the first enrollment template."""
    return enroll_matrix[0].copy()


# ============================================================================ #
#  CohortData
# ============================================================================ #


class TestCohortData:
    """CohortData NamedTuple tests."""

    def test_basic_construction(self, cohort_embs: np.ndarray) -> None:
        data = CohortData(embeddings=cohort_embs)
        assert np.array_equal(data.embeddings, cohort_embs)
        assert data.metadata is None

    def test_with_metadata(self, cohort_embs: np.ndarray) -> None:
        meta = {"name": "test_cohort", "size": 30}
        data = CohortData(embeddings=cohort_embs, metadata=meta)
        assert data.metadata == meta

    def test_is_namedtuple(self, cohort_embs: np.ndarray) -> None:
        data = CohortData(embeddings=cohort_embs)
        emb, meta = data
        assert np.array_equal(emb, cohort_embs)
        assert meta is None


# ============================================================================ #
#  top_k_mean_std
# ============================================================================ #


class TestTopKMeanStd:
    """top_k_mean_std pure function tests."""

    def test_basic_top_k(self) -> None:
        """Top 2 of [0.1, 0.5, 0.3, 0.9, 0.7] → mean ≈ 0.8."""
        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.7], dtype=np.float32)
        mean, std = top_k_mean_std(scores, k=2)
        assert mean == pytest.approx(0.8, abs=1e-6)
        assert std >= 0.0

    def test_k_larger_than_array(self) -> None:
        """k larger than len scores uses all elements."""
        scores = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        mean, std = top_k_mean_std(scores, k=10)
        assert mean == pytest.approx(0.3, abs=1e-6)
        assert std >= 0.0

    def test_single_element(self) -> None:
        """Single element → mean = value, std = 0."""
        scores = np.array([0.42], dtype=np.float32)
        mean, std = top_k_mean_std(scores, k=1)
        assert mean == pytest.approx(0.42, abs=1e-6)
        assert std == 0.0

    def test_empty_array_raises(self) -> None:
        """Empty scores array raises ValueError."""
        scores = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            top_k_mean_std(scores, k=1)

    def test_k_zero_raises(self) -> None:
        """k < 1 raises ValueError."""
        scores = np.array([0.1, 0.2], dtype=np.float32)
        with pytest.raises(ValueError, match="k must be >= 1"):
            top_k_mean_std(scores, k=0)

    def test_std_positive_with_multiple(self) -> None:
        """With varied top-k values, std should be > 0."""
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9], dtype=np.float32)
        mean, std = top_k_mean_std(scores, k=2)
        assert mean == pytest.approx(0.85, abs=1e-6)
        assert std > 0.0


# ============================================================================ #
#  apply_asnorm
# ============================================================================ #


class TestApplyAsnorm:
    """apply_asnorm pure function tests."""

    def test_returns_correct_shape(
        self,
        test_emb: np.ndarray,
        enroll_matrix: np.ndarray,
        cohort_embs: np.ndarray,
    ) -> None:
        """Returns (7,) scores, (7,) enroll_mu, (7,) enroll_sigma."""
        n_enroll = enroll_matrix.shape[0]
        norm_scores, enroll_mu, enroll_sigma = apply_asnorm(
            test_emb, enroll_matrix, cohort_embs, top_k=10
        )
        assert norm_scores.shape == (n_enroll,)
        assert enroll_mu.shape == (n_enroll,)
        assert enroll_sigma.shape == (n_enroll,)

    def test_output_is_finite(
        self,
        test_emb: np.ndarray,
        enroll_matrix: np.ndarray,
        cohort_embs: np.ndarray,
    ) -> None:
        """All output scores are finite numbers."""
        norm_scores, enroll_mu, enroll_sigma = apply_asnorm(
            test_emb, enroll_matrix, cohort_embs, top_k=10
        )
        assert np.all(np.isfinite(norm_scores))
        assert np.all(np.isfinite(enroll_mu))
        assert np.all(np.isfinite(enroll_sigma))

    def test_same_speaker_higher_score(
        self,
        enroll_matrix: np.ndarray,
        cohort_embs: np.ndarray,
    ) -> None:
        """The same-speaker template scores highest after AS-Norm."""
        # Make test_emb identical to the first enroll template.
        test_emb = enroll_matrix[0].copy()

        norm_scores, _, _ = apply_asnorm(
            test_emb, enroll_matrix, cohort_embs, top_k=10
        )
        # Index 0 should have the highest normalized score.
        assert norm_scores[0] == np.max(norm_scores)

    def test_different_k_values_give_different_results(
        self,
        test_emb: np.ndarray,
        enroll_matrix: np.ndarray,
        cohort_embs: np.ndarray,
    ) -> None:
        """Different top-k values produce different normalized scores."""
        scores_k1, _, _ = apply_asnorm(
            test_emb, enroll_matrix, cohort_embs, top_k=1
        )
        scores_k10, _, _ = apply_asnorm(
            test_emb, enroll_matrix, cohort_embs, top_k=10
        )
        # At least one score differs.
        assert not np.allclose(scores_k1, scores_k10)

    def test_invalid_test_emb_shape(
        self,
        enroll_matrix: np.ndarray,
        cohort_embs: np.ndarray,
    ) -> None:
        """Wrong test_emb shape raises ValueError."""
        with pytest.raises(ValueError, match="test_emb must be"):
            apply_asnorm(
                np.zeros(128, dtype=np.float32),
                enroll_matrix,
                cohort_embs,
            )

    def test_invalid_enroll_matrix_shape(
        self,
        test_emb: np.ndarray,
        cohort_embs: np.ndarray,
    ) -> None:
        """Wrong enroll_matrix shape raises ValueError."""
        with pytest.raises(ValueError, match="enroll_matrix must be"):
            apply_asnorm(
                test_emb,
                np.zeros((3, 128), dtype=np.float32),
                cohort_embs,
            )

    def test_invalid_cohort_shape(
        self,
        test_emb: np.ndarray,
        enroll_matrix: np.ndarray,
    ) -> None:
        """Wrong cohort_embeddings shape raises ValueError."""
        with pytest.raises(ValueError, match="cohort_embeddings must be"):
            apply_asnorm(
                test_emb,
                enroll_matrix,
                np.zeros((3, 128), dtype=np.float32),
            )


# ============================================================================ #
#  CohortCache
# ============================================================================ #


class TestCohortCache:
    """CohortCache class tests."""

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def test_valid_construction(self, cohort_embs: np.ndarray) -> None:
        """Valid (30, 256) embeddings construct successfully."""
        cache = CohortCache(cohort_embs)
        assert cache.size == 30

    def test_invalid_shape_raises_value_error(self) -> None:
        """Wrong embedding dimension raises ValueError."""
        bad_embs = np.zeros((10, 128), dtype=np.float32)
        with pytest.raises(ValueError, match="must be.*256"):
            CohortCache(bad_embs)

    def test_3d_array_raises_value_error(self) -> None:
        """3-D array raises ValueError."""
        bad_embs = np.zeros((5, 256, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="must be.*256"):
            CohortCache(bad_embs)

    def test_initial_size_property(self, cohort_embs: np.ndarray) -> None:
        """size property returns correct count."""
        cache = CohortCache(cohort_embs)
        assert cache.size == 30

    def test_initial_metadata_default(self, cohort_embs: np.ndarray) -> None:
        """Default metadata is an empty dict."""
        cache = CohortCache(cohort_embs)
        assert cache.metadata == {}

    def test_metadata_property(self, cohort_embs: np.ndarray) -> None:
        """metadata property returns the stored dict."""
        meta = {"source": "voxceleb"}
        cache = CohortCache(cohort_embs, metadata=meta)
        assert cache.metadata["source"] == "voxceleb"

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def test_save_load_roundtrip(self, cohort_embs: np.ndarray) -> None:
        """Save and load roundtrip preserves embeddings."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cohort.npy")
            cache = CohortCache(cohort_embs)
            cache.save(path)

            loaded = CohortCache.load(path)
            assert loaded.size == cache.size
            assert np.allclose(loaded.embeddings, cache.embeddings)

    def test_save_load_with_metadata(self, cohort_embs: np.ndarray) -> None:
        """Save and load roundtrip preserves metadata."""
        meta = {"name": "test", "size": 30}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cohort.npy")
            cache = CohortCache(cohort_embs, metadata=meta)
            cache.save(path)

            loaded = CohortCache.load(path)
            assert loaded.metadata["name"] == "test"
            assert loaded.metadata["size"] == 30

    def test_load_nonexistent_raises(self) -> None:
        """Loading a nonexistent .npy file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "nonexistent.npy")
            with pytest.raises(FileNotFoundError, match="not found"):
                CohortCache.load(path)

    def test_save_creates_parent_dir(self, cohort_embs: np.ndarray) -> None:
        """save() creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "subdir" / "cohort.npy")
            cache = CohortCache(cohort_embs)
            cache.save(path)
            assert Path(path).exists()

    # ------------------------------------------------------------------ #
    #  Precompute
    # ------------------------------------------------------------------ #

    def test_precompute_enroll_stats_shape(
        self,
        cohort_embs: np.ndarray,
        enroll_matrix: np.ndarray,
    ) -> None:
        """precompute_enroll_stats returns (N,) mu and sigma."""
        cache = CohortCache(cohort_embs)
        mu, sigma = cache.precompute_enroll_stats(
            enroll_matrix, enroll_names=None, top_k=10
        )
        assert mu.shape == (7,)
        assert sigma.shape == (7,)
        assert mu.dtype == np.float32
        assert sigma.dtype == np.float32

    def test_precompute_enroll_stats_finite(
        self,
        cohort_embs: np.ndarray,
        enroll_matrix: np.ndarray,
    ) -> None:
        """Precomputed stats are finite."""
        cache = CohortCache(cohort_embs)
        mu, sigma = cache.precompute_enroll_stats(
            enroll_matrix, top_k=10
        )
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(sigma))

    def test_precompute_enroll_stats_positive_sigma(
        self,
        cohort_embs: np.ndarray,
        enroll_matrix: np.ndarray,
    ) -> None:
        """All sigma values are positive (guarded against zero)."""
        cache = CohortCache(cohort_embs)
        _, sigma = cache.precompute_enroll_stats(
            enroll_matrix, top_k=10
        )
        assert np.all(sigma > 0.0)

    def test_precompute_enroll_stats_returns_cached(
        self,
        cohort_embs: np.ndarray,
        enroll_matrix: np.ndarray,
    ) -> None:
        """Calling precompute twice returns the same results."""
        cache = CohortCache(cohort_embs)
        mu1, sigma1 = cache.precompute_enroll_stats(
            enroll_matrix, top_k=10
        )
        mu2, sigma2 = cache.precompute_enroll_stats(
            enroll_matrix, top_k=10
        )
        assert np.allclose(mu1, mu2)
        assert np.allclose(sigma1, sigma2)

    def test_precompute_invalid_enroll_shape(
        self, cohort_embs: np.ndarray
    ) -> None:
        """Invalid enroll_matrix shape raises ValueError."""
        cache = CohortCache(cohort_embs)
        with pytest.raises(ValueError, match="enroll_matrix must be"):
            cache.precompute_enroll_stats(
                np.zeros((3, 128), dtype=np.float32)
            )

    # ------------------------------------------------------------------ #
    #  Apply
    # ------------------------------------------------------------------ #

    def test_apply_requires_precompute(
        self,
        cohort_embs: np.ndarray,
        test_emb: np.ndarray,
    ) -> None:
        """apply() raises RuntimeError if precompute not called."""
        cache = CohortCache(cohort_embs)
        with pytest.raises(RuntimeError, match="precompute_enroll_stats"):
            cache.apply(test_emb)

    def test_apply_after_precompute(
        self,
        cohort_embs: np.ndarray,
        enroll_matrix: np.ndarray,
        test_emb: np.ndarray,
    ) -> None:
        """apply() returns (N,) scores after precompute."""
        cache = CohortCache(cohort_embs)
        cache.precompute_enroll_stats(enroll_matrix, top_k=10)
        norm_scores, mu, sigma = cache.apply(test_emb, top_k=10)
        assert norm_scores.shape == (7,)
        assert mu.shape == (7,)
        assert sigma.shape == (7,)
        assert np.all(np.isfinite(norm_scores))

    def test_apply_same_speaker_highest(
        self,
        cohort_embs: np.ndarray,
        enroll_matrix: np.ndarray,
    ) -> None:
        """Same speaker scores highest through CohortCache.apply()."""
        test_emb = enroll_matrix[0].copy()

        cache = CohortCache(cohort_embs)
        cache.precompute_enroll_stats(enroll_matrix, top_k=10)
        norm_scores, _, _ = cache.apply(test_emb, top_k=10)
        assert norm_scores[0] == np.max(norm_scores)

    def test_apply_invalid_test_emb_shape(
        self,
        cohort_embs: np.ndarray,
        enroll_matrix: np.ndarray,
    ) -> None:
        """apply() raises ValueError for wrong test_emb shape."""
        cache = CohortCache(cohort_embs)
        cache.precompute_enroll_stats(enroll_matrix, top_k=10)
        with pytest.raises(ValueError, match="test_emb must be"):
            cache.apply(np.zeros(128, dtype=np.float32))

    def test_apply_consistency_with_pure_function(
        self,
        cohort_embs: np.ndarray,
        enroll_matrix: np.ndarray,
        test_emb: np.ndarray,
    ) -> None:
        """CohortCache.apply() matches apply_asnorm() for the same inputs."""
        cache = CohortCache(cohort_embs)
        cache.precompute_enroll_stats(enroll_matrix, top_k=10)

        cache_scores, _, _ = cache.apply(test_emb, top_k=10)
        func_scores, _, _ = apply_asnorm(
            test_emb, enroll_matrix, cohort_embs, top_k=10
        )
        assert np.allclose(cache_scores, func_scores, atol=1e-5)

    # ------------------------------------------------------------------ #
    #  Save-load with precomputed state
    # ------------------------------------------------------------------ #

    def test_save_load_does_not_preserve_precomputed(
        self,
        cohort_embs: np.ndarray,
        enroll_matrix: np.ndarray,
    ) -> None:
        """Save/load loses precomputed stats (they must be recomputed)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cohort.npy")

            cache = CohortCache(cohort_embs)
            cache.precompute_enroll_stats(enroll_matrix, top_k=10)
            cache.save(path)

            loaded = CohortCache.load(path)
            assert loaded.size == 30
            # Precomputed state should be gone after load.
            with pytest.raises(RuntimeError, match="precompute_enroll_stats"):
                loaded.apply(enroll_matrix[0])
