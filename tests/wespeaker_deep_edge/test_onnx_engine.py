"""Tests for OnnxEngine — validates FBANK, embedding, and recognition pipeline."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.wespeaker_deep_edge.onnx_engine import OnnxConfig, OnnxEngine, RecognitionResult, compute_fbank
from src.wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _sine_wave(freq: float = 440, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a sine wave PCM (int16)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)


@pytest.fixture
def engine() -> OnnxEngine:
    return OnnxEngine()


@pytest.fixture
def engine_with_templates(engine: OnnxEngine) -> OnnxEngine:
    """OnnxEngine with built-in voiceprint index 0 (John) loaded."""
    engine.load_templates(indices=[0])
    return engine


# --------------------------------------------------------------------------- #
#  test_fbank_matches_torchaudio
# --------------------------------------------------------------------------- #


def test_fbank_matches_torchaudio() -> None:
    """Numpy FBANK vs torchaudio FBANK: MSE < 1e-4."""
    try:
        import torch
        import torchaudio
    except ImportError:
        pytest.skip("torch/torchaudio not available — cannot validate FBANK")

    # Generate a test signal with some harmonic content
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    # Torchaudio FBANK
    torch_sig = torch.from_numpy(signal).unsqueeze(0).float()
    torch_fbank = torchaudio.compliance.kaldi.fbank(
        torch_sig,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        sample_frequency=sr,
        window_type="hamming",
        use_energy=False,
        dither=0.0,
    ).numpy()  # (T, 80)

    # Numpy FBANK
    np_fbank = compute_fbank(signal)  # (T, 80)

    # Align time dimension (torchaudio may pad/truncate differently)
    min_frames = min(torch_fbank.shape[0], np_fbank.shape[0])
    torch_fbank = torch_fbank[:min_frames]
    np_fbank = np_fbank[:min_frames]

    mse = float(np.mean((torch_fbank - np_fbank) ** 2))
    logger.info("FBANK MSE: %.6f", mse)
    assert mse < 1e-4, f"FBANK MSE {mse} >= 1e-4"


# --------------------------------------------------------------------------- #
#  test_embedding_pipeline
# --------------------------------------------------------------------------- #


def test_embedding_pipeline(engine: OnnxEngine) -> None:
    """Load ONNX, extract embedding from test audio, verify dim (256,)."""
    pcm = _sine_wave(440, 1.0)
    emb = engine.extract_embedding(pcm.astype(np.float32) / 32768.0, sample_rate=16000)
    assert emb is not None
    assert emb.shape == (256,), f"Expected (256,), got {emb.shape}"
    assert emb.dtype == np.float32


# --------------------------------------------------------------------------- #
#  test_recognize_consistency
# --------------------------------------------------------------------------- #


def test_recognize_consistency(engine: OnnxEngine) -> None:
    """OnnxEngine vs WespeakerDeep results must have < 0.2 deviation."""
    # Load templates matching built-in index 0 (John)
    engine.load_templates(indices=[0])

    pcm = _sine_wave(440, 2.0)
    result = engine.recognize_multi_pcm(pcm, sample_rate=16000)
    assert isinstance(result, RecognitionResult)
    assert result.name == "john"
    assert 0.0 <= result.confidence <= 1.0

    # Compare with WespeakerDeep if available
    try:
        deep = WespeakerDeep()
        deep.package_pk_index = 0
        deep_result = deep.recognize(pcm)
        if deep_result.confidence > 0:
            diff = abs(result.confidence - deep_result.confidence)
            logger.info("Confidence diff vs WespeakerDeep: %.4f", diff)
            assert diff < 0.2, f"Confidence deviation {diff} >= 0.2"
    except ImportError:
        pytest.skip("WespeakerDeep not available — skipping consistency check")


# --------------------------------------------------------------------------- #
#  test_asnorm_on_off
# --------------------------------------------------------------------------- #


def test_asnorm_off(engine_with_templates: OnnxEngine) -> None:
    """AS-Norm disabled branch: enable_asnorm=False, should use raw similarity."""
    engine_with_templates.enable_asnorm = False
    pcm = _sine_wave(440, 2.0)
    result = engine_with_templates.recognize_multi_pcm(pcm, sample_rate=16000)
    assert isinstance(result, RecognitionResult)
    assert result.name == "john"


def test_asnorm_on(engine_with_templates: OnnxEngine) -> None:
    """AS-Norm enabled but no cohort available — should fall back to raw result."""
    engine_with_templates.enable_asnorm = True
    # Without cohort file, AS-Norm is skipped silently
    pcm = _sine_wave(440, 2.0)
    result = engine_with_templates.recognize_multi_pcm(pcm, sample_rate=16000)
    assert isinstance(result, RecognitionResult)
    assert result.name == "john"


# --------------------------------------------------------------------------- #
#  test_short_audio
# --------------------------------------------------------------------------- #


def test_short_audio(engine: OnnxEngine) -> None:
    """Audio < 25ms should raise ValueError."""
    short = np.zeros(100, dtype=np.int16)  # 100 samples @ 16kHz = 6.25ms
    with pytest.raises(ValueError, match="No speech detected|audio too short"):
        engine.recognize_multi_pcm(short, sample_rate=16000)


# --------------------------------------------------------------------------- #
#  test_no_templates
# --------------------------------------------------------------------------- #


def test_no_templates(engine: OnnxEngine) -> None:
    """Calling recognize_multi_pcm without templates must raise RuntimeError."""
    pcm = _sine_wave(440, 1.0)
    with pytest.raises(RuntimeError, match="No templates loaded"):
        engine.recognize_multi_pcm(pcm, sample_rate=16000)


# --------------------------------------------------------------------------- #
#  test_multiple_templates
# --------------------------------------------------------------------------- #


def test_multiple_templates(engine: OnnxEngine) -> None:
    """Multi-template matching: returns the best match among several speakers."""
    # Load a few built-in voiceprints
    engine.load_templates(indices=[0, 1])  # john, frank
    assert len(engine._template_names) == 2

    pcm = _sine_wave(440, 2.0)
    result = engine.recognize_multi_pcm(pcm, sample_rate=16000)
    assert isinstance(result, RecognitionResult)
    assert result.name in ("john", "frank")
    assert result.all_scores is not None
    assert "john" in result.all_scores
    assert "frank" in result.all_scores


# --------------------------------------------------------------------------- #
#  test_clear_templates
# --------------------------------------------------------------------------- #


def test_clear_templates(engine_with_templates: OnnxEngine) -> None:
    """After clear_templates, recognize must raise RuntimeError."""
    engine_with_templates.clear_templates()
    pcm = _sine_wave(440, 1.0)
    with pytest.raises(RuntimeError, match="No templates loaded"):
        engine_with_templates.recognize_multi_pcm(pcm, sample_rate=16000)


# --------------------------------------------------------------------------- #
#  test_int16_normalization
# --------------------------------------------------------------------------- #


def test_int16_normalization(engine_with_templates: OnnxEngine) -> None:
    """int16 PCM is normalized to float32 internally without error."""
    pcm = _sine_wave(440, 1.0)
    assert pcm.dtype == np.int16
    result = engine_with_templates.recognize_multi_pcm(pcm, sample_rate=16000)
    assert isinstance(result, RecognitionResult)
    assert result.name == "john"


# --------------------------------------------------------------------------- #
#  test_resample
# --------------------------------------------------------------------------- #


def test_resample(engine_with_templates: OnnxEngine) -> None:
    """Audio at 48 kHz is resampled to 16 kHz automatically."""
    pcm_48k = _sine_wave(440, 1.0, sr=48000)
    result = engine_with_templates.recognize_multi_pcm(pcm_48k, sample_rate=48000)
    assert isinstance(result, RecognitionResult)
    assert result.name == "john"


# --------------------------------------------------------------------------- #
#  test_enroll_stub
# --------------------------------------------------------------------------- #


def test_enroll_stub(engine: OnnxEngine) -> None:
    """enroll() returns no-op dict, does not raise."""
    result = engine.enroll("some_audio.wav", "output.pkl")
    assert result == {"ok": False, "error": "enroll not supported by OnnxEngine"}


# --------------------------------------------------------------------------- #
#  test_onnx_config_dataclass
# --------------------------------------------------------------------------- #


def test_onnx_config_dataclass() -> None:
    """OnnxConfig has correct defaults and can be used to configure engine."""
    cfg = OnnxConfig()
    assert cfg.sim_threshold == 0.55
    assert cfg.enable_asnorm is True
    assert cfg.asnorm_threshold == 6.0
    assert cfg.asnorm_top_k == 300
    assert cfg.asnorm_norm_type == "asnorm"

    # Can round-trip through engine
    engine = OnnxEngine()
    engine.sim_threshold = 0.6
    assert engine.config.sim_threshold == 0.6


# --------------------------------------------------------------------------- #
#  test_stereo_audio
# --------------------------------------------------------------------------- #


def test_stereo_audio(engine_with_templates: OnnxEngine) -> None:
    """Multi-channel (stereo) PCM is squeezed to mono."""
    stereo = np.stack([_sine_wave(440, 1.0), _sine_wave(440, 1.0)], axis=1)  # (N, 2)
    result = engine_with_templates.recognize_multi_pcm(stereo, sample_rate=16000)
    assert isinstance(result, RecognitionResult)
    assert result.name == "john"
