# OnnxEngine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight ONNX Runtime-based speaker recognition engine that eliminates PyTorch/torchaudio dependencies, reducing memory from ~300-500 MB to ~30-50 MB.

**Architecture:** Pure numpy FBANK feature extraction → ONNX Runtime model inference (ResNet34, 256-d embedding) → template cosine similarity matching → AS-Norm score normalization. Same `.onnx` model file works on both Mac CPU and Jetson Orin Nano (CUDA/TensorRT).

**Tech Stack:** numpy, scipy (FBANK+resample), onnxruntime, CohortCache (asnorm.py, unchanged)

**Spec:** `docs/superpowers/specs/2026-05-23-onnx-engine-design.md`

---

### Task 1: Export ONNX model

**Files:**
- Modify: `pyproject.toml` — temporarily add `onnx` to deps group for export
- Run: `scripts/export_onnx_model.py`
- Create: `src/wespeaker_deep_edge/_models/vblinkf/model.onnx` (git-committed artifact)

- [ ] **Step 1: Install `onnx` package and verify export script**

The export script uses `torch.onnx.export` which requires the `onnx` Python package at runtime. Add it temporarily to the `deps` optional dependency group, then run the export.

Run: `cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv add onnx --group deps`

Expected: `onnx` is installed in the virtual environment.

- [ ] **Step 2: Run the ONNX export script**

Run: `cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv run python scripts/export_onnx_model.py`

Expected output:
```
Building SimAM_ResNet34_ASP(...)...
ONNX model saved to .../model.onnx  (24.8 MB)
```

Verify the output exists:
```bash
ls -lh src/wespeaker_deep_edge/_models/vblinkf/model.onnx
```

Expected: A ~25 MB `.onnx` file at the expected path.

- [ ] **Step 3: Verify ONNX model loads correctly**

Quick sanity check that the model loads in ONNX Runtime with correct input/output:

Run: `cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv run python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('src/wespeaker_deep_edge/_models/vblinkf/model.onnx')
print('Inputs:', [(i.name, i.shape) for i in sess.get_inputs()])
print('Outputs:', [(o.name, o.shape) for o in sess.get_outputs()])
# Dry run with random input
dummy = np.random.randn(1, 200, 80).astype(np.float32)
out = sess.run(['embs'], {'feats': dummy})[0]
print('Output shape:', out.shape)  # Expected: (1, 256)
"`

Expected: Input `feats` shape `[B, T, 80]`, output `embs` shape `[B, 256]`. Dry run succeeds.

- [ ] **Step 4: Remove `onnx` from deps group (optional dep for export only)**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv remove onnx
```

The project doesn't need `onnx` at runtime — only `onnxruntime` is needed.

- [ ] **Step 5: Commit the ONNX model**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && git add src/wespeaker_deep_edge/_models/vblinkf/model.onnx pyproject.toml
git commit -m "feat: export ONNX model for vblinkf SimAM_ResNet34_ASP

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Update pyproject.toml with correct deps and package-data

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace main dependencies and add optional dep groups**

Current main dep is just `onnx` (wrong). Replace with the correct core + optional dependencies.

Edit `pyproject.toml`: replace the `dependencies` list and `[project.optional-dependencies]` section:

Old content (lines 21-63):
```toml
dependencies = [
    "onnx",
]

[project.urls]
...

[project.optional-dependencies]
deps = [
    "audiomentations>=0.43.1",
    ...
    "huggingface-hub",
]
```

New content:
```toml
dependencies = [
    "numpy==1.26.4",
    "scipy>=1.0",
    "soundfile>=0.13.1",
]

[project.urls]
Homepage = "https://github.com/yourusername/wespeaker-deep-edge"
Documentation = "https://github.com/yourusername/wespeaker-deep-edge#readme"
Repository = "https://github.com/yourusername/wespeaker-deep-edge"
"Bug Tracker" = "https://github.com/yourusername/wespeaker-deep-edge/issues"

[project.optional-dependencies]
deps = [
    "audiomentations>=0.43.1",
    "hdbscan>=0.8.40",
    "kaldiio",
    "numpy==1.26.4",
    "pyannote-audio>=3.3.2",
    "pydub>=0.25.0",
    "pyyaml==6.0.3",
    "scipy>=1.0",
    "silero-vad",
    "sounddevice==0.5.2",
    "soundfile>=0.13.1",
    "s3prl",
    "tqdm",
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "umap-learn==0.5.6",
    # wespeaker 内置依赖（vendored _wespeaker/）
    "accelerate",
    "onnxruntime>=1.16.0,<2.0",
    "openai-whisper",
    "peft",
    "scikit-learn",
    "seaborn>=0.13.2",
    "matplotlib>=3.10.9",
    "datasets",
    "huggingface-hub",
]
cpu = ["onnxruntime"]
gpu = ["onnxruntime-gpu"]
```

- [ ] **Step 2: Add `*.onnx` to package-data for `_models`**

The current `_models` glob `"**/*"` doesn't match `*.onnx` in some setuptools configs. Add `*.onnx` explicitly.

Edit `pyproject.toml` `[tool.setuptools.package-data]` section:

Old:
```toml
"wespeaker_deep_edge._models" = ["**/*"]
```

New:
```toml
"wespeaker_deep_edge._models" = ["**/*", "*.onnx"]
```

- [ ] **Step 3: Run `uv lock` to update lockfile**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv lock
```

Expected: Lockfile regenerated successfully. No conflicts (numpy==1.26.4, scipy, soundfile are all compatible).

- [ ] **Step 4: Verify the dependency tree is correct**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv pip show numpy scipy soundfile
```

Expected: All three core deps are installed. `onnxruntime` NOT in the base install (only in `[cpu]` / `[gpu]` extras).

- [ ] **Step 5: Commit**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && git add pyproject.toml && git commit -m "feat: update deps for OnnxEngine — core deps + optional onnxruntime extras

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Update onnx_engine.py — add OnnxConfig and enroll() stub

**Files:**
- Modify: `src/wespeaker_deep_edge/onnx_engine.py`

- [ ] **Step 1: Add OnnxConfig dataclass before RecognitionResult**

Add after the constants block (~line 43), before RecognitionResult:

```python
from dataclasses import dataclass, field


@dataclass
class OnnxConfig:
    """Configuration for OnnxEngine recognition behavior."""
    sim_threshold: float = 0.55
    enable_asnorm: bool = True
    asnorm_threshold: float = 6.0
    asnorm_top_k: int = 300
    asnorm_norm_type: str = "asnorm"
```

- [ ] **Step 2: Add `enroll()` stub method to OnnxEngine class**

Add after `clear_templates()` (~line 307):

```python
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
```

- [ ] **Step 3: Add `config` property to OnnxEngine**

Add after `__init__` or after `clear_templates`:

```python
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
```

- [ ] **Step 4: Verify module imports cleanly**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv run python -c "from wespeaker_deep_edge.onnx_engine import OnnxEngine, OnnxConfig, RecognitionResult; print('OK')"
```

Expected: `OK` printed, no import errors.

- [ ] **Step 5: Quick smoke test of OnnxEngine API**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv run python -c "
from wespeaker_deep_edge.onnx_engine import OnnxEngine, OnnxConfig
import numpy as np

engine = OnnxEngine()
print('Config:', engine.config)

# Test enroll() stub
result = engine.enroll('dummy.wav')
print('Enroll result:', result)

# Test no-templates error
try:
    engine.recognize_multi_pcm(np.zeros(16000, dtype=np.int16))
except RuntimeError as e:
    print('Expected error:', e)
"
```

Expected: Config prints, enroll returns `{"ok": False}`, RuntimeError raised for missing templates.

- [ ] **Step 6: Commit**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && git add src/wespeaker_deep_edge/onnx_engine.py && git commit -m "feat: add OnnxConfig dataclass, enroll() stub, config property to OnnxEngine

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Update __init__.py exports

**Files:**
- Modify: `src/wespeaker_deep_edge/__init__.py`

- [ ] **Step 1: Add OnnxEngine and OnnxConfig to imports and `__all__`**

Edit `src/wespeaker_deep_edge/__init__.py`:

Current import block (~line 22):
```python
from . import diagnostics, realtime_monitor, reporters
from .wespeaker_deep_dege import DeepConfig, WespeakerDeep
```

Append ONNX imports:
```python
from . import diagnostics, realtime_monitor, reporters
from .onnx_engine import OnnxConfig, OnnxEngine
from .wespeaker_deep_dege import DeepConfig, WespeakerDeep
```

Current `__all__` (~line 25):
```python
__all__ = [
    "DeepConfig",
    "WespeakerDeep",
    "realtime_monitor",
    "diagnostics",
    "reporters",
]
```

Append:
```python
__all__ = [
    "DeepConfig",
    "OnnxConfig",
    "OnnxEngine",
    "WespeakerDeep",
    "realtime_monitor",
    "diagnostics",
    "reporters",
]
```

- [ ] **Step 2: Verify imports work**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv run python -c "from wespeaker_deep_edge import OnnxEngine, OnnxConfig; print('OnnxEngine imported from package OK')"
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && git add src/wespeaker_deep_edge/__init__.py && git commit -m "feat: export OnnxEngine and OnnxConfig from package

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Write comprehensive test suite

**Files:**
- Create: `tests/wespeaker_deep_edge/test_onnx_engine.py`

- [ ] **Step 1: Write the test file with all test cases**

Create `tests/wespeaker_deep_edge/test_onnx_engine.py`:

```python
"""Tests for OnnxEngine — validates FBANK, embedding, and recognition pipeline."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wespeaker_deep_edge import OnnxConfig, OnnxEngine, WespeakerDeep
from wespeaker_deep_edge.onnx_engine import RecognitionResult, compute_fbank

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
    """OnnxEngine vs WespeakerDeep results must have < 0.01 deviation."""
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
            assert diff < 0.01, f"Confidence deviation {diff} >= 0.01"
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
```

- [ ] **Step 2: Run tests and verify all pass**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && uv run pytest tests/wespeaker_deep_edge/test_onnx_engine.py -v
```

Expected: All test functions pass. Some tests may skip (e.g., `test_fbank_matches_torchaudio` and `test_recognize_consistency` if WespeakerDeep isn't available) but the core OnnxEngine tests must pass.

- [ ] **Step 3: Commit**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID && git add tests/wespeaker_deep_edge/test_onnx_engine.py && git commit -m "feat: add comprehensive OnnxEngine test suite

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Self-Review Checklist

**1. Spec coverage:**
- ✅ `compute_fbank()` — pure numpy, matches Kaldi defaults (Task 3 was actually done before this plan)
- ✅ `OnnxEngine.__init__()` with model_path, providers (pre-existing)
- ✅ `load_templates()` with indices and files (pre-existing)
- ✅ `clear_templates()` (pre-existing)
- ✅ `recognize_multi_pcm()` with int16→float32 normalization (pre-existing)
- ✅ `enroll()` no-op stub (Task 3)
- ✅ `OnnxConfig` dataclass (Task 3)
- ✅ `config` property (Task 3)
- ✅ `extract_embedding()` with resampling (pre-existing)
- ✅ Package-data `*.onnx` (Task 2)
- ✅ pyproject.toml deps — core + optional cpu/gpu (Task 2)
- ✅ __init__.py exports (Task 4)
- ✅ Tests: FBANK MSE < 1e-4, embedding dim, recognize consistency, AS-Norm on/off, short audio, no templates, multi templates, enroll stub (Task 5)
- ✅ ONNX model export (Task 1)
- ✅ Error handling: audio too short → ValueError, no templates → RuntimeError, model not found → FileNotFoundError, enroll → no-op stub

**2. Placeholder scan:** No TBDs, TODOs, or implementation-gaps. Every step has complete code.

**3. Type consistency:** OnnxConfig fields match between dataclass definition, engine `__init__` defaults, and test assertions. `enroll()` signature matches spec. All method names consistent.
