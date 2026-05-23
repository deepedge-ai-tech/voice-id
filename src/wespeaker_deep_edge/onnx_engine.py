"""Minimal speaker recognition — ONNX Runtime, no PyTorch.

Eliminates the ``import torch`` overhead (~200-400 MB) by running
the ResNet34 model via ONNX Runtime and computing FBANK features
in pure numpy + scipy.

Usage::

    from wespeaker_deep_edge.onnx_engine import OnnxEngine

    engine = OnnxEngine()
    engine.load_templates(indices=[0, 1])  # built-in voiceprints

    result = engine.recognize_multi_pcm(pcm_int16_array, sample_rate=16000)
    print(result.name, result.confidence)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import onnxruntime as ort

from .asnorm import CohortCache

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants  (matches training config of vblinkf SimAM_ResNet34_ASP)
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
FFT_SIZE = 512
WINDOW_LENGTH = int(0.025 * SAMPLE_RATE)  # 25 ms  → 400 samples
HOP_LENGTH = int(0.010 * SAMPLE_RATE)     # 10 ms  → 160 samples
LOW_FREQ = 20.0
HIGH_FREQ = 8000.0   # nyquist for 16 kHz (matches Kaldi default)


# --------------------------------------------------------------------------- #
#  OnnxConfig
# --------------------------------------------------------------------------- #


@dataclass
class OnnxConfig:
    """Configuration for OnnxEngine recognition behavior."""
    sim_threshold: float = 0.55
    enable_asnorm: bool = True
    asnorm_threshold: float = 6.0
    asnorm_top_k: int = 300
    asnorm_norm_type: str = "asnorm"


# --------------------------------------------------------------------------- #
#  RecognitionResult
# --------------------------------------------------------------------------- #


class RecognitionResult(NamedTuple):
    is_recognized: bool
    confidence: float
    name: str
    all_scores: dict | None = None


# --------------------------------------------------------------------------- #
#  FBANK features  (pure numpy, matches Kaldi defaults)
# --------------------------------------------------------------------------- #


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to mel (Kaldi/HTK scale)."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert mel to Hz (Kaldi/HTK scale)."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_fft: int,
    sr: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Create a Mel filter bank matrix matching torchaudio/Kaldi's approach.

    For each FFT bin frequency we evaluate the triangular mel filter value,
    rather than snapping to discrete bins.  This avoids degenerate (zero)
    filters at low frequencies.
    """
    # Mel frequency range
    mel_min = _hz_to_mel(np.array(fmin))
    mel_max = _hz_to_mel(np.array(fmax))

    # Center frequencies of each triangular filter (n_mels filters)
    # Equally spaced in mel domain, Kaldi-style: / (n_mels + 1)
    delta = (mel_max - mel_min) / (n_mels + 1)
    left_mel = mel_min + np.arange(n_mels, dtype=np.float64) * delta
    center_mel = mel_min + (np.arange(n_mels, dtype=np.float64) + 1.0) * delta
    right_mel = mel_min + (np.arange(n_mels, dtype=np.float64) + 2.0) * delta

    # FFT bin frequencies (Hz)  (n_fft // 2 + 1 bins)
    n_freqs = n_fft // 2 + 1
    fft_bin_width = sr / n_fft  # Hz per bin  (31.25 for 16-kHz / 512)
    fft_freqs = np.arange(n_freqs, dtype=np.float64) * fft_bin_width

    # Convert to mel scale
    mel_freqs = _hz_to_mel(fft_freqs)  # (n_freqs,)

    # Build filterbank via triangle evaluation  (n_mels, n_freqs)
    left_mel = left_mel[:, np.newaxis]   # (n_mels, 1)
    center_mel = center_mel[:, np.newaxis]
    right_mel = right_mel[:, np.newaxis]
    mel_freqs = mel_freqs[np.newaxis, :]  # (1, n_freqs)

    up_slope = (mel_freqs - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel_freqs) / (right_mel - center_mel)

    fbank = np.maximum(0.0, np.minimum(up_slope, down_slope))

    return fbank.astype(np.float32)


def _hamming_window(n: int) -> np.ndarray:
    """Symmetric Hamming window matching torchaudio with ``periodic=False``."""
    return np.array([0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n - 1)) for i in range(n)], dtype=np.float32)


def compute_fbank(waveform: np.ndarray) -> np.ndarray:
    """Compute 80-dim log-Mel FBANK features, shape (T, 80).

    Matches torchaudio's ``fbank`` with ``dither=0.0, preemph_coef=0.97,
    window_type="hamming", remove_dc_offset=True``.

    Args:
        waveform: 1-D float32 array, 16 kHz.

    Returns:
        FBANK features shaped (T, 80), where T = number of frames.

    Raises:
        ValueError: If the waveform is too short to produce at least one
            full frame.
    """
    n_samples = len(waveform)
    if n_samples < WINDOW_LENGTH:
        raise ValueError("audio too short")

    n_frames = (n_samples - WINDOW_LENGTH) // HOP_LENGTH + 1

    # Pre-allocate framed array  (m, window_size)
    frames = np.zeros((n_frames, WINDOW_LENGTH), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frames[i] = waveform[start:start + WINDOW_LENGTH]

    # Per-frame DC offset removal  (torchaudio: before pre-emphasis)
    row_means = frames.mean(axis=1, keepdims=True)
    frames = frames - row_means

    # Per-frame pre-emphasis with left-edge replicate
    # y[j] = x[j] - 0.97 * (j==0 ? x[0] : x[j-1])
    left_pad = frames[:, :1].copy()  # replicate first column
    shifted = np.concatenate([left_pad, frames[:, :-1]], axis=1)
    frames = frames - 0.97 * shifted

    # Apply Hamming window
    window = _hamming_window(WINDOW_LENGTH)
    frames = frames * window

    # Power spectrum via FFT
    mag_spec = np.abs(np.fft.rfft(frames, n=FFT_SIZE, axis=1)) ** 2

    # Mel filter bank + log
    fbank = _mel_filterbank(FFT_SIZE, SAMPLE_RATE, NUM_MEL_BINS, LOW_FREQ, HIGH_FREQ)
    features = np.dot(mag_spec, fbank.T)  # (T, 80)
    features = np.maximum(features, np.finfo(np.float32).eps)
    features = np.log(features)

    return features.astype(np.float32)


# --------------------------------------------------------------------------- #
#  OnnxEngine
# --------------------------------------------------------------------------- #


class OnnxEngine:
    """Minimal speaker recognizer using ONNX Runtime.

    No PyTorch dependencies.  Supports multi-template matching and
    AS-Norm score normalization.

    Args:
        model_path: Path to the ``.onnx`` model file.  ``None`` uses the
            bundled ``_models/vblinkf/model.onnx``.
        providers: ONNX Runtime execution providers.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if model_path is None:
            model_path = Path(__file__).parent / "_models" / "vblinkf" / "model.onnx"

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                f"Run `uv run python scripts/export_onnx_model.py` first."
            )

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 2
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), sess_options=so,
            providers=providers or ["CPUExecutionProvider"],
        )

        self.sample_rate = SAMPLE_RATE
        self.sim_threshold = 0.55
        self.asnorm_threshold = 6.0
        self.enable_asnorm = True
        self.asnorm_top_k = 300
        self.asnorm_norm_type = "asnorm"

        # Template cache
        self._template_matrix: np.ndarray | None = None
        self._template_names: list[str] = []
        self._template_norms: np.ndarray | None = None
        self._cohort_cache: CohortCache | None = None

    # ------------------------------------------------------------------ #
    #  Embedding extraction
    # ------------------------------------------------------------------ #

    def extract_embedding(self, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
        """Extract a 256-d embedding from a PCM audio array.

        Args:
            pcm: 1-D float32 array (values in [-1, 1]).
                Multi-channel arrays are averaged to mono.
            sample_rate: Sample rate of *pcm*.  Will be resampled if
                different from 16 kHz.

        Returns:
            256-d float32 embedding, or ``None`` if too short.
        """
        if pcm.ndim > 1:
            # Average channels to mono
            pcm = pcm.mean(axis=1)

        # Resample to 16 kHz if needed
        if sample_rate != SAMPLE_RATE:
            pcm = self._resample(pcm, sample_rate, SAMPLE_RATE)

        # Compute features
        try:
            feats = compute_fbank(pcm)
        except ValueError:
            return None
        if feats.shape[0] < 1:
            return None

        # ONNX inference: input [1, T, 80], output [1, 256]
        emb = self._session.run(
            output_names=["embs"],
            input_feed={"feats": feats[np.newaxis, :, :]},
        )[0]  # (1, 256)

        return emb[0].astype(np.float32)

    @staticmethod
    def _resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample using scipy.signal (avoids torchaudio dependency)."""
        from scipy.signal import resample

        if orig_sr == target_sr:
            return signal
        n_samples = int(len(signal) * target_sr / orig_sr)
        return resample(signal, n_samples).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Template loading & matching
    # ------------------------------------------------------------------ #

    def load(self, pk_path: str | Path) -> np.ndarray:
        with open(pk_path, "rb") as f:
            return np.asarray(pickle.load(f), dtype=np.float32)

    def load_templates(
        self,
        indices: list[int] | None = None,
        files: dict[str, str | Path] | None = None,
    ) -> None:
        """Load voiceprint templates into memory.

        Args:
            indices: Built-in voiceprint indices.
            files: Custom voiceprint paths, e.g. ``{"target": "voice.pkl"}``.
        """
        from ._voiceprints import get_voiceprint_path, get_voiceprint_name

        embs: list[np.ndarray] = []
        names: list[str] = []

        if indices:
            for idx in indices:
                ref_data = self.load(get_voiceprint_path(idx))
                embs.append(ref_data.astype(np.float32))
                names.append(get_voiceprint_name(idx))

        if files:
            for name, path in files.items():
                ref_data = self.load(str(path))
                embs.append(ref_data.astype(np.float32))
                names.append(name)

        if not embs:
            raise ValueError("No templates provided")

        self._template_matrix = np.stack(embs)  # (N, 256)
        self._template_names = names
        self._template_norms = np.linalg.norm(self._template_matrix, axis=1)

        logger.info("Loaded %d templates: %s", len(names), names)

        if self.enable_asnorm and self._cohort_cache is None:
            self.load_cohort()
        self._precompute_cohort_stats_if_needed()

    def clear_templates(self) -> None:
        self._template_matrix = None
        self._template_names = []
        self._template_norms = None

    def enroll(
        self,
        audio_path: str | Path,
        pk_path: str | Path = "voice.pkl",
        **kwargs: Any,
    ) -> dict:
        """No-op stub for compatibility with WespeakerDeep API.

        OnnxEngine only supports recognition, not enrollment.
        Use WespeakerDeep.enroll() for voiceprint registration.

        Args:
            audio_path: Ignored (retained for API compat).
            pk_path: Ignored (retained for API compat).

        Returns:
            Dict with ``{"ok": False, "error": "enroll not supported by OnnxEngine"}``.
        """
        logger.warning("OnnxEngine.enroll() is a no-op stub; use WespeakerDeep for enrollment.")
        return {"ok": False, "error": "enroll not supported by OnnxEngine"}

    @property
    def config(self) -> OnnxConfig:
        """Return current configuration as an OnnxConfig."""
        return OnnxConfig(
            sim_threshold=self.sim_threshold,
            enable_asnorm=self.enable_asnorm,
            asnorm_threshold=self.asnorm_threshold,
            asnorm_top_k=self.asnorm_top_k,
            asnorm_norm_type=self.asnorm_norm_type,
        )

    # ------------------------------------------------------------------ #
    #  AS-Norm cohort
    # ------------------------------------------------------------------ #

    def load_cohort(self, path: str | Path | None = None) -> None:
        if not self.enable_asnorm:
            return

        path = Path(path) if path else Path("")

        if not (path.is_absolute() and path.is_file()) and not path.is_file():
            from importlib import resources

            bundled = resources.files("wespeaker_deep_edge._cohort") / "cohort_embeddings.npy"
            if bundled.is_file():
                path = bundled

        try:
            self._cohort_cache = CohortCache.load(str(path))
            logger.info("Loaded cohort: %s (%d speakers)", path, self._cohort_cache.size)
        except FileNotFoundError:
            logger.warning("Cohort not found: %s. AS-Norm disabled.", path)
            self._cohort_cache = None

        self._precompute_cohort_stats_if_needed()

    def _precompute_cohort_stats_if_needed(self) -> None:
        if self._cohort_cache is not None and self._template_matrix is not None:
            self._cohort_cache.precompute_enroll_stats(
                self._template_matrix,
                self._template_names,
                top_k=self.asnorm_top_k,
            )

    # ------------------------------------------------------------------ #
    #  Recognition
    # ------------------------------------------------------------------ #

    def _match_templates(self, test_emb: np.ndarray) -> RecognitionResult:
        if self._template_matrix is None:
            raise RuntimeError("No templates loaded. Call load_templates() first.")

        audio_norm = np.linalg.norm(test_emb)
        scores = (self._template_matrix @ test_emb) / (self._template_norms * audio_norm)
        scores = (scores + 1.0) / 2  # [-1, 1] → [0, 1]

        best_pos = int(np.argmax(scores))
        threshold = self.sim_threshold
        score_map = dict(zip(self._template_names, [round(float(s), 4) for s in scores]))

        return RecognitionResult(
            is_recognized=bool(scores[best_pos] >= threshold),
            confidence=round(float(scores[best_pos]), 4),
            name=self._template_names[best_pos],
            all_scores=score_map,
        )

    def recognize_multi_pcm(
        self,
        pcm: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> RecognitionResult:
        """Recognize speaker from PCM audio.

        Args:
            pcm: 1-D int16 or float32 PCM array.
                Multi-channel arrays are averaged to mono.
            sample_rate: Sample rate.

        Returns:
            RecognitionResult with best match.
        """
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)

        # Normalize int16 → float32
        if pcm.dtype == np.int16:
            pcm = pcm.astype(np.float32) / 32768.0
        else:
            pcm = pcm.astype(np.float32)

        test_emb = self.extract_embedding(pcm, sample_rate)
        if test_emb is None:
            raise ValueError("No speech detected or audio too short")

        raw_result = self._match_templates(test_emb)

        # AS-Norm post-processing
        if (
            self.enable_asnorm
            and self._cohort_cache is not None
            and self._cohort_cache._enroll_mu is not None
            and raw_result.all_scores is not None
        ):
            norm_scores, _, _ = self._cohort_cache.apply(
                test_emb,
                top_k=self.asnorm_top_k,
                norm_type=self.asnorm_norm_type,
            )

            best_idx = int(np.argmax(norm_scores))
            names = self._template_names
            threshold = self.asnorm_threshold

            return RecognitionResult(
                is_recognized=bool(norm_scores[best_idx] >= threshold),
                confidence=round(float(norm_scores[best_idx]), 4),
                name=names[best_idx],
                all_scores={
                    name: round(float(norm_scores[i]), 4)
                    for i, name in enumerate(names)
                },
            )

        return raw_result
