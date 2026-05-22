"""AS-Norm (Adaptive Score Normalization) module for speaker recognition.

Provides pure functions and a caching class for computing AS-Norm
on speaker embedding cosine similarity scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class CohortData(NamedTuple):
    """Container for cohort embeddings and optional metadata.

    Attributes:
        embeddings: Cohort embedding matrix of shape (N, 256).
        metadata: Optional dictionary with cohort information.
    """

    embeddings: np.ndarray
    metadata: dict | None = None


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all pairs of rows in a and b.

    Args:
        a: Matrix of shape (M, D).
        b: Matrix of shape (N, D).

    Returns:
        Cosine similarity matrix of shape (M, N).
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Convert cosine similarity from [-1, 1] to [0, 1].

    Args:
        scores: Cosine similarity values in [-1, 1].

    Returns:
        Scores mapped to [0, 1].
    """
    return (scores + 1.0) / 2.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Squash real-valued scores to (0, 1) via logistic sigmoid.

    Uses a gentle slope (x/3) so that typical AS-Norm z-scores
    spread across the (0, 1) range instead of collapsing near 1::

        z=0  → 0.50
        z=3  → 0.73
        z=6  → 0.88
        z=9  → 0.95
        z=12 → 0.98

    Args:
        x: Input scores (unbounded).

    Returns:
        Scores in (0, 1).
    """
    x_clipped = np.clip(x / 3.0, -100, 100)
    return 1.0 / (1.0 + np.exp(-x_clipped))


# --------------------------------------------------------------------------- #
#  Core functions
# --------------------------------------------------------------------------- #


def top_k_mean_std(scores: np.ndarray, k: int) -> tuple[float, float]:
    """Compute mean and standard deviation of the top-k scores.

    Args:
        scores: 1-D array of scores.
        k: Number of top scores to use.  If k exceeds the array length,
            all scores are used.

    Returns:
        Tuple of (mean, standard_deviation).  Standard deviation is 0.0
        when k <= 1.

    Raises:
        ValueError: If scores is empty or k < 1.
    """
    if scores.size == 0:
        raise ValueError("scores array is empty")
    if k < 1:
        raise ValueError("k must be >= 1")

    k = min(k, scores.size)
    top_k = np.sort(scores)[-k:]
    mean = float(np.mean(top_k))
    std = float(np.std(top_k, ddof=1)) if k > 1 else 0.0
    return mean, std


def apply_asnorm(
    test_emb: np.ndarray,
    enroll_matrix: np.ndarray,
    cohort_embeddings: np.ndarray,
    top_k: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply AS-Norm (Adaptive Score Normalization) to recognition scores.

    For each enrollment template, computes:
        norm_i = 0.5 * (raw_i - mu_enroll_i) / sigma_enroll_i
               + 0.5 * (raw_i - mu_test) / sigma_test

    where all raw scores are cosine similarities converted to [0, 1] range,
    and mu/sigma are computed from the top-k similarity scores against the
    cohort.

    Args:
        test_emb: Test embedding vector of shape (256,).
        enroll_matrix: Enrollment embedding matrix of shape (N, 256)
            where N is the number of enrollment templates.
        cohort_embeddings: Cohort embedding matrix of shape (M, 256)
            where M is the number of cohort utterances.
        top_k: Number of top cohort scores to use for statistics.

    Returns:
        Tuple of (norm_scores, enroll_mu, enroll_sigma):
            - norm_scores: (N,) normalized scores (float32).
            - enroll_mu: (N,) enrollment-side means (float32).
            - enroll_sigma: (N,) enrollment-side standard deviations (float32).

    Raises:
        ValueError: If input shapes are invalid.
    """
    # --- Validate shapes -------------------------------------------------- #
    if test_emb.ndim != 1 or test_emb.shape[0] != 256:
        raise ValueError(f"test_emb must be (256,), got {test_emb.shape}")
    if enroll_matrix.ndim != 2 or enroll_matrix.shape[1] != 256:
        raise ValueError(
            f"enroll_matrix must be (N, 256), got {enroll_matrix.shape}"
        )
    if cohort_embeddings.ndim != 2 or cohort_embeddings.shape[1] != 256:
        raise ValueError(
            f"cohort_embeddings must be (M, 256), got {cohort_embeddings.shape}"
        )

    n_enroll = enroll_matrix.shape[0]

    # --- Test-side statistics --------------------------------------------- #
    test_cohort_sim = _cosine_similarity_matrix(
        test_emb.reshape(1, -1), cohort_embeddings
    ).flatten()  # (M,)
    test_cohort_sim_norm = _normalize_scores(test_cohort_sim)
    mu_test, sigma_test = top_k_mean_std(test_cohort_sim_norm, top_k)
    sigma_test = max(sigma_test, 1e-8)

    # --- Enroll-side statistics (per template) ---------------------------- #
    enroll_cohort_sim = _cosine_similarity_matrix(
        enroll_matrix, cohort_embeddings
    )  # (N, M)
    enroll_cohort_sim_norm = _normalize_scores(enroll_cohort_sim)

    enroll_mu = np.zeros(n_enroll, dtype=np.float32)
    enroll_sigma = np.zeros(n_enroll, dtype=np.float32)

    for i in range(n_enroll):
        mu_i, sigma_i = top_k_mean_std(enroll_cohort_sim_norm[i], top_k)
        enroll_mu[i] = float(mu_i)
        enroll_sigma[i] = float(max(sigma_i, 1e-8))

    # --- Raw test-enroll similarities ------------------------------------- #
    raw_scores = _cosine_similarity_matrix(
        test_emb.reshape(1, -1), enroll_matrix
    ).flatten()  # (N,)
    raw_scores_norm = _normalize_scores(raw_scores)

    # --- Apply AS-Norm formula -------------------------------------------- #
    norm_scores = np.zeros(n_enroll, dtype=np.float32)
    for i in range(n_enroll):
        norm_scores[i] = (
            0.5 * (raw_scores_norm[i] - enroll_mu[i]) / enroll_sigma[i]
            + 0.5 * (raw_scores_norm[i] - mu_test) / sigma_test
        )

    return norm_scores, enroll_mu, enroll_sigma


# --------------------------------------------------------------------------- #
#  CohortCache
# --------------------------------------------------------------------------- #


class CohortCache:
    """Cache for cohort embeddings with precomputed AS-Norm statistics.

    Stores a cohort of speaker embeddings and provides methods for
    computing and caching enrollment-side statistics so that the full
    AS-Norm pipeline can be applied efficiently across multiple test
    utterances.

    Typical usage::

        cache = CohortCache(cohort_embeddings)
        cache.precompute_enroll_stats(enroll_matrix)
        norm_scores, _, _ = cache.apply(test_emb)
    """

    def __init__(
        self, embeddings: np.ndarray, metadata: dict | None = None
    ) -> None:
        """Initialize the cohort cache.

        Args:
            embeddings: Cohort embedding matrix of shape (N, 256).
            metadata: Optional metadata dictionary.

        Raises:
            ValueError: If embeddings shape is not (N, 256).
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != 256:
            raise ValueError(
                f"embeddings must be (N, 256), got {embeddings.shape}"
            )
        self._embeddings = embeddings.astype(np.float32, copy=False)
        self._metadata = metadata or {}

        # Precompute normalized cohort embeddings for efficiency.
        self._norms = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        )

        # Cached enrollment statistics (populated by precompute_enroll_stats).
        self._enroll_norm: np.ndarray | None = None
        self._enroll_mu: np.ndarray | None = None
        self._enroll_sigma: np.ndarray | None = None
        self._enroll_names: list[str] | None = None
        
    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str) -> CohortCache:
        """Load cohort embeddings from a .npy file.

        If a .json file with the same stem exists adjacent to the .npy
        file, it is loaded as metadata.

        Args:
            path: Path to the .npy file containing embeddings.

        Returns:
            A new CohortCache instance.

        Raises:
            FileNotFoundError: If the .npy file does not exist.
        """
        npy_path = Path(path)
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Cohort embeddings file not found: {path}"
            )

        embeddings = np.load(str(npy_path))

        metadata: dict | None = None
        json_path = npy_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r") as f:
                metadata = json.load(f)

        return cls(embeddings, metadata=metadata)

    def save(self, path: str) -> None:
        """Save cohort embeddings and metadata to disk.

        Args:
            path: Path for the .npy file.  Metadata is written to a
                companion .json file with the same stem.
        """
        npy_path = Path(path)
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(npy_path), self._embeddings)

        if self._metadata:
            json_path = npy_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(self._metadata, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        """Return the number of cohort embeddings."""
        return self._embeddings.shape[0]

    @property
    def embeddings(self) -> np.ndarray:
        """Return the cohort embeddings (read-only view)."""
        return self._embeddings

    @property
    def metadata(self) -> dict:
        """Return the metadata dictionary."""
        return self._metadata

    # ------------------------------------------------------------------ #
    #  Precomputation & application
    # ------------------------------------------------------------------ #

    def precompute_enroll_stats(
        self,
        enroll_matrix: np.ndarray,
        enroll_names: list[str] | None = None,
        top_k: int = 300,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute enrollment-side AS-Norm statistics.

        Computes mu_enroll and sigma_enroll for each enrollment template
        by measuring the similarity of each template against all cohort
        embeddings and taking the top-k statistics.

        Args:
            enroll_matrix: Enrollment embedding matrix of shape (N, 256).
            enroll_names: Optional list of N enrollment speaker names.
            top_k: Number of top cohort scores to use for statistics.

        Returns:
            Tuple of (enroll_mu, enroll_sigma), each shape (N,) float32.

        Raises:
            ValueError: If enroll_matrix shape is invalid.
        """
        if enroll_matrix.ndim != 2 or enroll_matrix.shape[1] != 256:
            raise ValueError(
                f"enroll_matrix must be (N, 256), got {enroll_matrix.shape}"
            )

        # Normalize enrollment embeddings and cache them for apply().
        enroll_norm = enroll_matrix.astype(np.float32, copy=False) / (
            np.linalg.norm(enroll_matrix, axis=1, keepdims=True) + 1e-10
        )

        # Similarity: (M, N) where M = cohort size, N = enroll count.
        sim_matrix = np.dot(self._norms, enroll_norm.T)  # (M, N)
        sim_matrix_norm = _normalize_scores(sim_matrix)

        n_enroll = enroll_matrix.shape[0]
        enroll_mu = np.zeros(n_enroll, dtype=np.float32)
        enroll_sigma = np.zeros(n_enroll, dtype=np.float32)

        for i in range(n_enroll):
            mu_i, sigma_i = top_k_mean_std(sim_matrix_norm[:, i], top_k)
            enroll_mu[i] = float(mu_i)
            enroll_sigma[i] = float(max(sigma_i, 1e-8))

        # Cache for later use by apply().
        self._enroll_norm = enroll_norm
        self._enroll_mu = enroll_mu
        self._enroll_sigma = enroll_sigma
        self._enroll_names = enroll_names
        
        return enroll_mu, enroll_sigma

    def apply(
        self, test_emb: np.ndarray, top_k: int = 300,
        norm_type: str = "asnorm",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply full AS-Norm pipeline using precomputed enrollment stats.

        The test-side statistics (mu_test, sigma_test) are computed at
        call time from the given test embedding.  The enrollment-side
        statistics must have been precomputed via
        :meth:`precompute_enroll_stats`.

        Args:
            test_emb: Test embedding vector of shape (256,).
            top_k: Number of top cohort scores to use for test-side
                statistics.  Does **not** override the enrollment-side
                top_k used during precomputation.
            norm_type: Normalization variant — ``"asnorm"`` (both sides),
                ``"snorm"`` (test-side only), or ``"tnorm"`` (enroll-side only).

        Returns:
            Tuple of (norm_scores, enroll_mu, enroll_sigma).

        Raises:
            RuntimeError: If :meth:`precompute_enroll_stats` has not been
                called before this method.
            ValueError: If test_emb shape is invalid.
        """
        if self._enroll_mu is None or self._enroll_sigma is None:
            raise RuntimeError(
                "precompute_enroll_stats must be called before apply"
            )

        if test_emb.ndim != 1 or test_emb.shape[0] != 256:
            raise ValueError(
                f"test_emb must be (256,), got {test_emb.shape}"
            )

        # Normalize test embedding.
        test_norm = test_emb.astype(np.float32) / (
            np.linalg.norm(test_emb) + 1e-10
        )

        # --- Test-side statistics -------------------------------------- #
        test_cohort_sim = np.dot(self._norms, test_norm)
        test_cohort_sim_norm = _normalize_scores(test_cohort_sim)
        mu_test, sigma_test = top_k_mean_std(test_cohort_sim_norm, top_k)
        sigma_test = max(sigma_test, 1e-8)

        # --- Raw test-enroll similarities ------------------------------ #
        
        raw_scores = np.dot(self._enroll_norm, test_norm)  # (N,)
        raw_scores_norm = _normalize_scores(raw_scores)

        # --- Apply AS-Norm formula (vectorised) ------------------------ #
        if norm_type == "tnorm":
            norm_scores = (
                (raw_scores_norm - self._enroll_mu) / self._enroll_sigma
            ).astype(np.float32)
        elif norm_type == "snorm":
            sigma_test_arr = np.full_like(raw_scores_norm, sigma_test)
            norm_scores = (
                (raw_scores_norm - mu_test) / sigma_test_arr
            ).astype(np.float32)
        else:  # "asnorm"
            sigma_test_arr = np.full_like(raw_scores_norm, sigma_test)
            norm_scores = (
                0.5 * (raw_scores_norm - self._enroll_mu) / self._enroll_sigma
                + 0.5 * (raw_scores_norm - mu_test) / sigma_test_arr
            ).astype(np.float32)

        return norm_scores, self._enroll_mu, self._enroll_sigma
