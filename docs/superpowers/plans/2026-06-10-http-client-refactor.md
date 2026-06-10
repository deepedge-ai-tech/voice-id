# WeSpeaker HTTP Client Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor WespeakerDeep from a local ONNX inference engine into a lightweight HTTP client for the voice-id REST API, deleting all ML dependencies and model files (~230MB → ~10KB).

**Architecture:** Single HTTP client class wrapping voice-id's REST API (`POST /voiceprint/register`, `POST /voiceprint/identify`), with backward-compatible method signatures. Dependencies reduced to just `requests` and `soundfile`.

**Tech Stack:** Python 3.10+, requests, soundfile, no PyTorch/ONNX Runtime

---

### Task 1: Create branch and delete unnecessary files

**Files:**
- Delete: `src/wespeaker_deep_edge/wespeaker_deep_dege.py`
- Delete: `src/wespeaker_deep_edge/onnx_engine.py`
- Delete: `src/wespeaker_deep_edge/asnorm.py`
- Delete: `src/wespeaker_deep_edge/_utils.py`
- Delete: `src/wespeaker_deep_edge/diagnostics.py`
- Delete: `src/wespeaker_deep_edge/reporters.py`
- Delete: `src/wespeaker_deep_edge/realtime_monitor.py`
- Delete: `src/wespeaker_deep_edge/client/` (entire directory)
- Delete: `src/wespeaker_deep_edge/server/` (entire directory)
- Delete: `src/wespeaker_deep_edge/_wespeaker/` (entire directory)
- Delete: `src/wespeaker_deep_edge/_models/` (entire directory)
- Delete: `src/wespeaker_deep_edge/_cohort/` (entire directory)
- Delete: `src/wespeaker_deep_edge/_voiceprints/*.pkl`
- Delete: `src/wespeaker_deep_edge/__pycache__/` (if tracked)
- Delete: `tests/wespeaker_deep_edge/test_wespeaker_deep_dege.py`
- Delete: `tests/wespeaker_deep_edge/test_speaker_client.py`
- Delete: `tests/wespeaker_deep_edge/test_template_manager.py`
- Delete: `tests/wespeaker_deep_edge/test_ws_server.py`
- Delete: `tests/wespeaker_deep_edge/test_asnorm.py`
- Delete: `tests/wespeaker_deep_edge/test_onnx_engine.py`
- Modify: `src/wespeaker_deep_edge/_voiceprints/__init__.py`

- [ ] **Step 1: Create the branch and commit current state**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID
git checkout -b refactor/onnx-http-client
```

- [ ] **Step 2: Delete all Python module files that are being removed**

```bash
git rm src/wespeaker_deep_edge/wespeaker_deep_dege.py
git rm src/wespeaker_deep_edge/onnx_engine.py
git rm src/wespeaker_deep_edge/asnorm.py
git rm src/wespeaker_deep_edge/_utils.py
git rm src/wespeaker_deep_edge/diagnostics.py
git rm src/wespeaker_deep_edge/reporters.py
git rm src/wespeaker_deep_edge/realtime_monitor.py
```

- [ ] **Step 3: Delete entire directories that are being removed**

```bash
git rm -r src/wespeaker_deep_edge/client/
git rm -r src/wespeaker_deep_edge/server/
git rm -r src/wespeaker_deep_edge/_wespeaker/
git rm -r src/wespeaker_deep_edge/_models/
git rm -r src/wespeaker_deep_edge/_cohort/
```

- [ ] **Step 4: Delete old voiceprint .pkl files and old test files**

```bash
# Remove all .pkl voiceprints (keep __init__.py)
cd src/wespeaker_deep_edge/_voiceprints
git rm *.pkl 2>/dev/null || true
cd ../../..

# Remove old test files
git rm tests/wespeaker_deep_edge/test_wespeaker_deep_dege.py
git rm tests/wespeaker_deep_edge/test_speaker_client.py
git rm tests/wespeaker_deep_edge/test_template_manager.py
git rm tests/wespeaker_deep_edge/test_ws_server.py
git rm tests/wespeaker_deep_edge/test_asnorm.py
git rm tests/wespeaker_deep_edge/test_onnx_engine.py
```

- [ ] **Step 5: Update `_voiceprints/__init__.py` — remove john_double_mic from mapping**

Edit `src/wespeaker_deep_edge/_voiceprints/__init__.py`:

```python
"""内置声纹包 — 索引↔名称映射。

映射：
    =====  ============
    Index  Name
    =====  ============
    0      John
    1      Frank
    2      Michael
    3      Qingqing
    4      Xixi
    5      Zhong
    6      Angle
    7      Albert
    =====  ============
"""

from importlib import resources

_PEOPLE: list[str] = [
    "john",
    "frank",
    "michael",
    "qingqing",
    "xixi",
    "zhong",
    "angle",
    "albert",
]


def get_voiceprint_path(index: int) -> str:
    if index < 0 or index >= len(_PEOPLE):
        raise IndexError(
            f"package_pk_index {index} out of range (0-{len(_PEOPLE) - 1})"
        )
    return ""


def get_voiceprint_name(index: int) -> str:
    if index < 0 or index >= len(_PEOPLE):
        raise IndexError(
            f"package_pk_index {index} out of range (0-{len(_PEOPLE) - 1})"
        )
    return _PEOPLE[index]
```

Note: `get_voiceprint_path()` returns empty string since voiceprints are no longer stored locally. The function is kept for backward compatibility.

- [ ] **Step 6: Commit**

```bash
git add src/wespeaker_deep_edge/_voiceprints/__init__.py
git commit -m "refactor: delete all ML engine code, vendored libs, old voiceprints, and tests

Remove PyTorch engine, ONNX engine, AS-Norm, vendored wespeaker,
client/, server/, models, cohort and all old test files.
Keep only _voiceprints/__init__.py for index-to-name mapping.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Create HTTP client

**Files:**
- Create: `src/wespeaker_deep_edge/client.py`

- [ ] **Step 1: Write the test first**

Create `tests/wespeaker_deep_edge/test_client.py`:

```python
"""Tests for WespeakerDeep HTTP Client — uses mocked HTTP responses."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

from src.wespeaker_deep_edge.client import WespeakerDeep, RecognitionResult

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def client() -> WespeakerDeep:
    return WespeakerDeep(
        base_url="http://test:8005",
        api_key="test-key",
    )


@pytest.fixture
def mock_response():
    """Helper to create a mock requests.Response."""
    def _make(status_code=200, json_data=None):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.raise_for_status.return_value = None
        return resp
    return _make


# --------------------------------------------------------------------------- #
#  Constructor tests
# --------------------------------------------------------------------------- #


def test_default_constructor():
    """Default constructor uses env var fallback values."""
    client = WespeakerDeep()
    assert client.base_url == "http://127.0.0.1:8005"
    assert client.api_key == ""


def test_custom_constructor():
    """Custom base_url and api_key are set correctly."""
    client = WespeakerDeep(base_url="http://custom:9999", api_key="abc123")
    assert client.base_url == "http://custom:9999"
    assert client.api_key == "abc123"


# --------------------------------------------------------------------------- #
#  enroll() tests
# --------------------------------------------------------------------------- #


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_enroll_success(mock_post, client, mock_response, tmp_path):
    """Successful enrollment returns ok=True with message."""
    mock_post.return_value = mock_response(
        200, {"success": True, "msg": "已登记: john"}
    )

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-wav-content")

    result = client.enroll(str(audio_file), "voice_john.pkl")

    assert result["ok"] is True
    assert "john" in result["msg"]
    mock_post.assert_called_once()


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_enroll_extracts_speaker_id_from_pkpath(mock_post, client, mock_response, tmp_path):
    """Enroll extracts speaker_id from pk_path filename (voice_X.pkl → X)."""
    mock_post.return_value = mock_response(
        200, {"success": True, "msg": "已登记: frank"}
    )

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-wav-content")

    client.enroll(str(audio_file), "voice_frank.pkl")

    # Check that the correct speaker_id was sent
    call_kwargs = mock_post.call_args.kwargs
    data = call_kwargs.get("data", {})
    # requests.post(data=...) can be dict
    assert data["speaker_id"] == "frank" or any(
        "frank" in str(v) for v in data.values()
    )


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_enroll_failure(mock_post, client, mock_response, tmp_path):
    """API failure returns ok=False with error message."""
    mock_post.return_value = mock_response(
        500, {"detail": "声纹注册失败"}
    )
    mock_post.return_value.ok = False
    mock_post.return_value.raise_for_status.side_effect = requests.HTTPError()

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-wav-content")

    with pytest.raises(RuntimeError, match="register failed"):
        client.enroll(str(audio_file), "voice_john.pkl")


# --------------------------------------------------------------------------- #
#  recognize() tests
# --------------------------------------------------------------------------- #


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_recognize_success(mock_post, client, mock_response, tmp_path):
    """Recognize returns identified speaker."""
    mock_post.return_value = mock_response(
        200, {"speaker_id": "john", "score": 0.85}
    )

    client.load_templates(indices=[0])
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-wav-content")

    result = client.recognize(str(audio_file))

    assert result["is_recognized"] is True
    assert result["name"] == "john"
    assert result["confidence"] == 0.85


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_recognize_no_match(mock_post, client, mock_response, tmp_path):
    """Recognize returns is_recognized=False when no match."""
    mock_post.return_value = mock_response(
        200, {"speaker_id": "", "score": 0.0}
    )

    client.load_templates(indices=[0])
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-wav-content")

    result = client.recognize(str(audio_file))

    assert result["is_recognized"] is False
    assert result["name"] == ""


# --------------------------------------------------------------------------- #
#  recognize_multi_pcm() tests
# --------------------------------------------------------------------------- #


@patch("src.wespeaker_deep_edge.client.requests.post")
@patch("src.wespeaker_deep_edge.client.soundfile.write")
def test_recognize_multi_pcm(mock_sf_write, mock_post, client, mock_response, tmp_path):
    """recognize_multi_pcm creates temp WAV and calls identify."""
    mock_post.return_value = mock_response(
        200, {"speaker_id": "john", "score": 0.85}
    )

    import numpy as np
    pcm = np.zeros(16000, dtype=np.int16)
    client.load_templates(indices=[0])

    result = client.recognize_multi_pcm(pcm, sample_rate=16000)

    assert isinstance(result, RecognitionResult)
    assert result.is_recognized is True
    assert result.name == "john"
    assert result.confidence == 0.85
    mock_sf_write.assert_called_once()


@patch("src.wespeaker_deep_edge.client.requests.post")
@patch("src.wespeaker_deep_edge.client.soundfile.write")
def test_recognize_multi_pcm_no_match(mock_sf_write, mock_post, client, mock_response, tmp_path):
    """recognize_multi_pcm returns not recognized when no match."""
    mock_post.return_value = mock_response(
        200, {"speaker_id": "", "score": 0.0}
    )

    import numpy as np
    pcm = np.zeros(16000, dtype=np.int16)
    client.load_templates(indices=[0])

    result = client.recognize_multi_pcm(pcm, sample_rate=16000)

    assert isinstance(result, RecognitionResult)
    assert result.is_recognized is False
    assert result.name == ""
    assert result.confidence == 0.0


# --------------------------------------------------------------------------- #
#  load_templates() tests
# --------------------------------------------------------------------------- #


def test_load_templates_indices(client):
    """load_templates with indices caches the speaker names."""
    client.load_templates(indices=[0, 1])
    assert "john" in client._speaker_ids
    assert "frank" in client._speaker_ids
    assert len(client._speaker_ids) == 2


def test_load_templates_files(client):
    """load_templates with files dict caches file keys as speaker_ids."""
    client.load_templates(files={"target1": "path1.pkl", "target2": "path2.pkl"})
    assert "target1" in client._speaker_ids
    assert "target2" in client._speaker_ids


def test_load_templates_both(client):
    """load_templates combines indices and files."""
    client.load_templates(indices=[0], files={"extra": "extra.pkl"})
    assert "john" in client._speaker_ids
    assert "extra" in client._speaker_ids


def test_load_templates_empty_raises(client):
    """load_templates with no arguments raises ValueError."""
    with pytest.raises(ValueError, match="No templates"):
        client.load_templates()


# --------------------------------------------------------------------------- #
#  RecognitionResult tests
# --------------------------------------------------------------------------- #


def test_recognition_result():
    """RecognitionResult NamedTuple works correctly."""
    result = RecognitionResult(
        is_recognized=True,
        confidence=0.85,
        name="john",
    )
    assert result.is_recognized is True
    assert result.confidence == 0.85
    assert result.name == "john"

    result_tuple = (True, 0.85, "john")
    assert result == RecognitionResult(*result_tuple)


# --------------------------------------------------------------------------- #
#  HTTP error handling tests
# --------------------------------------------------------------------------- #


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_recognize_http_error(mock_post, client, mock_response, tmp_path):
    """HTTP errors in recognize are wrapped in RuntimeError."""
    mock_post.return_value = mock_response(401, {"detail": "无效的接口令牌"})
    mock_post.return_value.ok = False
    mock_post.return_value.raise_for_status.side_effect = requests.HTTPError()

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-wav-content")
    client.load_templates(indices=[0])

    with pytest.raises(RuntimeError, match="identify failed"):
        client.recognize(str(audio_file))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID
uv run pytest tests/wespeaker_deep_edge/test_client.py -v
```
Expected: FAIL with "No module named 'src.wespeaker_deep_edge.client'"

- [ ] **Step 3: Write the HTTP client implementation**

Create `src/wespeaker_deep_edge/client.py`:

```python
"""Minimal speaker recognition — HTTP client for voice-id REST API.

No PyTorch or ONNX Runtime dependencies. All recognition is delegated to
the voice-id service (voiceprint-api) via HTTP.

Usage::

    from wespeaker_deep_edge import WespeakerDeep

    client = WespeakerDeep(
        base_url="http://127.0.0.1:8005",
        api_key="your-api-key",
    )
    client.enroll("audio.wav", "voice_john.pkl")
    result = client.recognize("test.wav")
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import requests
import soundfile as sf

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  RecognitionResult
# --------------------------------------------------------------------------- #


class RecognitionResult(NamedTuple):
    """Speaker recognition result.

    Attributes:
        is_recognized: Whether the confidence meets the server's threshold.
        confidence: Cosine similarity score [0, 1].
        name: Matched speaker name (empty string if no match).
    """
    is_recognized: bool
    confidence: float
    name: str


# --------------------------------------------------------------------------- #
#  WespeakerDeep — HTTP Client
# --------------------------------------------------------------------------- #


class WespeakerDeep:
    """Voiceprint recognition client that delegates to a REST API.

    Wraps the voice-id HTTP API (``/voiceprint/register``,
    ``/voiceprint/identify``) behind the same method signatures
    as the original local engine for backward compatibility.

    Args:
        base_url: The voice-id service URL. Falls back to ``VOICE_ID_URL``
            env var, then ``http://127.0.0.1:8005``.
        api_key: API authentication token. Falls back to ``VOICE_ID_KEY``
            env var, then empty string.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("VOICE_ID_URL", "http://127.0.0.1:8005")).rstrip("/")
        self.api_key = api_key or os.getenv("VOICE_ID_KEY", "")
        self._speaker_ids: list[str] = []
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    #  Registration
    # ------------------------------------------------------------------ #

    def enroll(
        self,
        audio_path: str | Path,
        pk_path: str | Path = "voice.pkl",
    ) -> dict:
        """Register a speaker voiceprint via the REST API.

        The ``pk_path`` argument is retained for API compatibility but is
        **not** used to write a local file. The ``speaker_id`` is inferred
        from the ``pk_path`` filename: ``voice_john.pkl`` → ``"john"``.
        If the filename doesn't match ``voice_*.pkl``, the ``speaker_id``
        is derived from the audio filename instead.

        Args:
            audio_path: Path to the WAV audio file for enrollment.
            pk_path: Retained for backward compatibility. Speaker ID is
                inferred from the filename (``voice_X.pkl`` → ``"X"``).

        Returns:
            ``{"ok": True, "msg": "已登记: <speaker_id>"}`` on success.

        Raises:
            RuntimeError: If the API request fails.
        """
        speaker_id = self._resolve_speaker_id(audio_path, pk_path)

        with open(audio_path, "rb") as f:
            resp = self._session.post(
                f"{self.base_url}/voiceprint/register",
                headers=self._auth_headers,
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"speaker_id": speaker_id},
            )

        try:
            resp.raise_for_status()
            result = resp.json()
            return {
                "ok": result.get("success", False),
                "msg": result.get("msg", ""),
            }
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"register failed for {speaker_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    #  Recognition
    # ------------------------------------------------------------------ #

    def recognize(
        self,
        audio_path: str | Path | object,
        voiceprint: object | None = None,
    ) -> dict:
        """Identify a speaker from an audio file.

        Args:
            audio_path: Path to a WAV file.
            voiceprint: Ignored for HTTP client (kept for API compat).
                The speaker ID list from ``load_templates()`` is used
                as candidates.

        Returns:
            ``{"is_recognized": bool, "confidence": float, "name": str}``

        Raises:
            RuntimeError: If the API request fails.
        """
        speaker_ids = self._speaker_ids or ["john"]

        with open(audio_path, "rb") as f:
            resp = self._session.post(
                f"{self.base_url}/voiceprint/identify",
                headers=self._auth_headers,
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"speaker_ids": ",".join(speaker_ids)},
            )

        try:
            resp.raise_for_status()
            result = resp.json()
            speaker_id = result.get("speaker_id", "")
            score = result.get("score", 0.0)
            return {
                "is_recognized": bool(speaker_id),
                "confidence": float(score),
                "name": speaker_id,
            }
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"identify failed: {exc}"
            ) from exc

    def recognize_multi_pcm(
        self,
        pcm: object,
        sample_rate: int = 16000,
    ) -> RecognitionResult:
        """Identify a speaker from a raw PCM array.

        Writes the PCM to a temporary WAV file, calls the identify API,
        and cleans up.

        Args:
            pcm: 1-D int16 or float32 PCM array.
            sample_rate: Sample rate of the PCM data.

        Returns:
            RecognitionResult with the best match.

        Raises:
            RuntimeError: If the API request fails.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            try:
                sf.write(tmp_path, pcm, sample_rate)
                result = self.recognize(tmp_path)
            finally:
                os.unlink(tmp_path)

        return RecognitionResult(
            is_recognized=result["is_recognized"],
            confidence=result["confidence"],
            name=result["name"],
        )

    # ------------------------------------------------------------------ #
    #  Template management
    # ------------------------------------------------------------------ #

    def load_templates(
        self,
        indices: list[int] | None = None,
        files: dict[str, str | Path] | None = None,
    ) -> None:
        """Cache candidate speaker IDs for subsequent recognize calls.

        Args:
            indices: Built-in voiceprint indices (0-7).
            files: Custom voiceprint mappings, e.g. ``{"target": "path.pkl"}``.

        Raises:
            ValueError: If no templates are provided.
        """
        from ._voiceprints import get_voiceprint_name

        ids: list[str] = []

        if indices:
            for idx in indices:
                ids.append(get_voiceprint_name(idx))

        if files:
            ids.extend(files.keys())

        if not ids:
            raise ValueError("No templates provided")

        self._speaker_ids = ids
        logger.info("Loaded %d speaker IDs: %s", len(ids), ids)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _resolve_speaker_id(audio_path: str | Path, pk_path: str | Path) -> str:
        """Extract the speaker ID from the pk_path or audio filename."""
        pk_name = Path(pk_path).stem  # e.g. "voice_john" or "voice"
        if pk_name.startswith("voice_"):
            return pk_name[len("voice_"):]
        # Fallback: use audio filename stem
        return Path(audio_path).stem
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID
uv run pytest tests/wespeaker_deep_edge/test_client.py -v
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/wespeaker_deep_edge/client.py tests/wespeaker_deep_edge/test_client.py
git commit -m "feat: add WespeakerDeep HTTP client with full test suite

New client.py wraps voice-id REST API behind backward-compatible
method signatures (enroll, recognize, recognize_multi_pcm, load_templates).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Update `__init__.py` and `__main__.py`

**Files:**
- Modify: `src/wespeaker_deep_edge/__init__.py`
- Modify: `src/wespeaker_deep_edge/__main__.py`

- [ ] **Step 1: Write the test first**

Add to `tests/wespeaker_deep_edge/test_client.py`:

```python
# --------------------------------------------------------------------------- #
#  Package-level imports
# --------------------------------------------------------------------------- #


def test_wespeaker_deep_import():
    """WespeakerDeep is importable from the package root."""
    from wespeaker_deep_edge import WespeakerDeep
    assert WespeakerDeep is not None


def test_recognition_result_import():
    """RecognitionResult is importable from the package root."""
    from wespeaker_deep_edge import RecognitionResult
    assert RecognitionResult is not None
```

- [ ] **Step 2: Rewrite `__init__.py`**

Edit `src/wespeaker_deep_edge/__init__.py`:

```python
"""WeSpeaker 声纹识别工具 — HTTP Client for voice-id REST API.

Usage::

    from wespeaker_deep_edge import WespeakerDeep

    client = WespeakerDeep(
        base_url="http://127.0.0.1:8005",
        api_key="your-token",
    )
    client.enroll("audio.wav", "voice_john.pkl")
    result = client.recognize("test.wav")
"""

from .client import WespeakerDeep, RecognitionResult

__all__ = [
    "WespeakerDeep",
    "RecognitionResult",
]
```

- [ ] **Step 3: Rewrite `__main__.py`**

Edit `src/wespeaker_deep_edge/__main__.py`:

```python
"""CLI entry point — uses WespeakerDeep HTTP client.

Supports ``python -m wespeaker_deep_edge``.
"""

import argparse
import logging
import sys
from importlib.metadata import version as _pkg_version

from .client import WespeakerDeep

logger = logging.getLogger(__name__)

_PKG_VERSION = _pkg_version("wespeaker-deep-edge")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WeSpeaker 声纹注册与识别（voice-id HTTP API）"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_PKG_VERSION}",
    )
    parser.add_argument(
        "--url", default=None,
        help="voice-id API base URL (default: $VOICE_ID_URL or http://127.0.0.1:8005)",
    )
    parser.add_argument(
        "--key", default=None,
        help="API key (default: $VOICE_ID_KEY)",
    )

    sub = parser.add_subparsers(dest="cmd")

    # ---- enroll ----
    p_reg = sub.add_parser("enroll", help="注册声纹")
    p_reg.add_argument("speaker_id", help="说话人 ID")
    p_reg.add_argument("audio", help="音频文件路径")

    # ---- recognize ----
    p_rec = sub.add_parser("recognize", help="识别声纹")
    p_rec.add_argument("audio", help="音频文件路径")
    p_rec.add_argument(
        "speaker_ids", nargs="?", default=None,
        help="候选说话人 ID，逗号分隔（默认使用内置声纹）",
    )

    # ---- list-voiceprints ----
    sub.add_parser("list-voiceprints", help="列出所有内置声纹")

    args = parser.parse_args()

    client = WespeakerDeep(base_url=args.url, api_key=args.key)

    if args.cmd == "enroll":
        pk_path = f"voice_{args.speaker_id}.pkl"
        r = client.enroll(args.audio, pk_path)
        print(f"{'✅' if r['ok'] else '❌'} {r['msg']}")

    elif args.cmd == "recognize":
        if args.speaker_ids:
            client.load_templates(files={s: "" for s in args.speaker_ids.split(",")})
        else:
            client.load_templates(indices=[0])
        r = client.recognize(args.audio)
        status = "✅ 识别成功" if r["is_recognized"] else "❌ 未识别"
        print(f"{status}  name={r['name']}  confidence={r['confidence']:.4f}")

    elif args.cmd == "list-voiceprints":
        from ._voiceprints import _PEOPLE
        print("内置声纹列表:")
        for i, name in enumerate(_PEOPLE):
            print(f"  {i}: {name}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID
uv run pytest tests/wespeaker_deep_edge/test_client.py -v
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/wespeaker_deep_edge/__init__.py src/wespeaker_deep_edge/__main__.py
git commit -m "refactor: update __init__.py and __main__.py for HTTP client

Remove old PyTorch engine imports.
CLI now uses WespeakerDeep HTTP client with --url and --key options.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update dependencies and package config**

Edit `pyproject.toml` — replace the entire file:

```toml
[project]
name = "wespeaker-deep-edge"
version = "0.2.0"
description = "WeSpeaker 声纹识别工具 — voice-id REST API HTTP Client"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "WeSpeaker Contributors" }
]
keywords = ["speaker", "verification", "voice", "recognition", "speaker-identification"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Operating System :: OS Independent",
]
dependencies = [
    "requests>=2.28",
    "soundfile>=0.13",
]

[project.urls]
Homepage = "https://github.com/yourusername/wespeaker-deep-edge"
Documentation = "https://github.com/yourusername/wespeaker-deep-edge#readme"
Repository = "https://github.com/yourusername/wespeaker-deep-edge"
"Bug Tracker" = "https://github.com/yourusername/wespeaker-deep-edge/issues"

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"wespeaker_deep_edge" = ["*.yaml", "*.yml", "*.json", "*.txt"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
addopts = "-v --cov=src/wespeaker_deep_edge --cov-report=term-missing --cov-fail-under=25"

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.isort]
profile = "black"
```

Key changes:
- Version bumped to 0.2.0
- Dependencies: only `requests` + `soundfile` (removed numpy, scipy, all ML deps)
- Removed `[project.optional-dependencies]` (deps, cpu, gpu no longer needed)
- Removed `[project.scripts]` section
- Removed `package-data` for `_models`, `_voiceprints`, `_wespeaker`, `_cohort` (no longer bundled)
- Removed black/isort tool config (optional, can keep)

- [ ] **Step 2: Run tests to verify install**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID
uv sync
uv run pytest tests/wespeaker_deep_edge/test_client.py -v
```
Expected: uv sync succeeds (fast, no heavy downloads), tests PASS

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: update pyproject.toml for v0.2.0 HTTP client

- Dependencies reduced to requests + soundfile
- Removed all ML optional dependencies
- Removed bundled package-data entries
- Version bumped to 0.2.0

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Register all 8 voiceprints via API

**Files:**
- No code changes — script to call the live API

- [ ] **Step 1: Delete existing voiceprints from the server**

```bash
# First get the current list
curl -s "http://127.0.0.1:8005/voiceprint/health?key=5640546d-531a-4e3e-8b4c-6cddf2125686"
```

If there are existing voiceprints, delete each:
```bash
for sid in john frank michael qingqing xixi zhong angle albert; do
  echo "Deleting $sid..."
  curl -s -X DELETE "http://127.0.0.1:8005/voiceprint/$sid" \
    -H "Authorization: Bearer 5640546d-531a-4e3e-8b4c-6cddf2125686"
done
```

- [ ] **Step 2: Register all 8 speakers**

```bash
ASSET_DIR="/Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID/asset_combine"
API="http://127.0.0.1:8005"
KEY="5640546d-531a-4e3e-8b4c-6cddf2125686"

declare -A FILES
FILES["john"]="$ASSET_DIR/John.wav"
FILES["frank"]="$ASSET_DIR/Frank.wav"
FILES["michael"]="$ASSET_DIR/Michael.wav"
FILES["qingqing"]="$ASSET_DIR/Qingqing.wav"
FILES["xixi"]="$ASSET_DIR/Xixi.wav"
FILES["zhong"]="$ASSET_DIR/Zhong.wav"
FILES["angle"]="$ASSET_DIR/angle.wav"
FILES["albert"]="$ASSET_DIR/Albert.wav"

for sid in "${!FILES[@]}"; do
  echo ""
  echo "=== Registering: $sid ==="
  curl -s -X POST "$API/voiceprint/register" \
    -H "Authorization: Bearer $KEY" \
    -F "speaker_id=$sid" \
    -F "file=@${FILES[$sid]}" | python3 -m json.tool
done
```

- [ ] **Step 3: Verify all 8 voiceprints are registered**

```bash
curl -s "http://127.0.0.1:8005/voiceprint/health?key=5640546d-531a-4e3e-8b4c-6cddf2125686" | python3 -m json.tool
```
Expected: `{"total_voiceprints": 8, "status": "healthy"}`

- [ ] **Step 4: Quick recognition test for each speaker**

```bash
API="http://127.0.0.1:8005"
KEY="5640546d-531a-4e3e-8b4c-6cddf2125686"

# Test one speaker to verify
curl -s -X POST "$API/voiceprint/identify" \
  -H "Authorization: Bearer $KEY" \
  -F "speaker_ids=john,frank,michael,qingqing,xixi,zhong,angle,albert" \
  -F "file=@/Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID/asset/zhong/test_segments/$(ls /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID/asset/zhong/test_segments/ | head -1)" | python3 -m json.tool
```

- [ ] **Step 5: Commit the registration result as a record**

```bash
# Create a record of the registration
cat > asset_combine/REGISTERED.md << 'EOF'
# Registered Voiceprints

All 8 voiceprints registered on 2026-06-10.

| Index | speaker_id | Source file             |
|-------|-----------|-------------------------|
| 0     | john      | asset_combine/John.wav |
| 1     | frank     | asset_combine/Frank.wav |
| 2     | michael   | asset_combine/Michael.wav |
| 3     | qingqing  | asset_combine/Qingqing.wav |
| 4     | xixi      | asset_combine/Xixi.wav |
| 5     | zhong     | asset_combine/Zhong.wav |
| 6     | angle     | asset_combine/angle.wav |
| 7     | albert    | asset_combine/Albert.wav |

API: http://127.0.0.1:8005
EOF

git add asset_combine/REGISTERED.md
git commit -m "feat: register 8 voiceprints via API

All 8 built-in speakers registered to voice-id service at
http://127.0.0.1:8005 with key from data/.voiceprint.yaml.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Package as tar

- [ ] **Step 1: Build the tar**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID

# Ensure we're on the right branch
git checkout refactor/onnx-http-client

# Build the tar
tar -czf wespeaker-deep-edge-docker-v0.2.0.tar.gz \
    --transform 's|^|wespeaker-deep-edge-docker-v0.2.0/|' \
    src/wespeaker_deep_edge/ \
    pyproject.toml \
    README.md
```

The `--transform` adds a root directory so it extracts cleanly.

- [ ] **Step 2: Verify the tar content**

```bash
tar -tzf wespeaker-deep-edge-docker-v0.2.0.tar.gz
```
Expected output shows files under `wespeaker-deep-edge-docker-v0.2.0/`:
```
wespeaker-deep-edge-docker-v0.2.0/src/wespeaker_deep_edge/__init__.py
wespeaker-deep-edge-docker-v0.2.0/src/wespeaker_deep_edge/__main__.py
wespeaker-deep-edge-docker-v0.2.0/src/wespeaker_deep_edge/client.py
wespeaker-deep-edge-docker-v0.2.0/src/wespeaker_deep_edge/_voiceprints/__init__.py
wespeaker-deep-edge-docker-v0.2.0/pyproject.toml
wespeaker-deep-edge-docker-v0.2.0/README.md
```

- [ ] **Step 3: Verify pip install works from tar**

```bash
# Install from tar
uv pip install wespeaker-deep-edge-docker-v0.2.0.tar.gz

# Verify the import works
uv run python -c "
from wespeaker_deep_edge import WespeakerDeep, RecognitionResult
client = WespeakerDeep()
print('WespeakerDeep imported OK')
print('base_url:', client.base_url)
"
```
Expected: No import errors, prints the base_url.

- [ ] **Step 4: Commit**

```bash
# Add tar to .gitignore is not needed (tar is outside src, not tracked)
# Just note the tar in the commit
git add wespeaker-deep-edge-docker-v0.2.0.tar.gz
git commit -m "chore: package v0.2.0 as tar for docker deployment

tar: wespeaker-deep-edge-docker-v0.2.0.tar.gz
Content: src/wespeaker_deep_edge/ + pyproject.toml + README.md
Install: pip install wespeaker-deep-edge-docker-v0.2.0.tar.gz

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Update README with new usage

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README.md with new HTTP client usage**

Edit `README.md` to reflect the new architecture. Key sections to update:
- Installation: now requires only `requests` + `soundfile`
- Usage: no more model paths or `.pkl` management
- Environment variables: `VOICE_ID_URL` and `VOICE_ID_KEY`

Key usage example:

```markdown
## Usage

```python
from wespeaker_deep_edge import WespeakerDeep

# Connect to voice-id service
client = WespeakerDeep(
    base_url="http://127.0.0.1:8005",
    api_key="your-api-key",
)

# Register
client.enroll("speaker.wav", "voice_john.pkl")

# Recognize
result = client.recognize("test.wav")
print(result["name"], result["confidence"])
```

CLI:
```bash
# Register
uv run python -m wespeaker_deep_edge enroll john audio.wav

# Recognize
uv run python -m wespeaker_deep_edge recognize test.wav

# Recognize with specific candidates
uv run python -m wespeaker_deep_edge recognize test.wav "john,frank"
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for HTTP client usage

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID
uv sync
uv run pytest tests/ -v --cov --cov-fail-under=25
```
Expected: All tests PASS, coverage >= 25%

- [ ] **Step 2: Check git log**

```bash
git log --oneline refactor/onnx-http-client
```
Expected: Clear commit history showing the refactoring steps.

- [ ] **Step 3: Check tar is distributable**

```bash
# Create a temporary venv and install from tar
cd /tmp
python3 -m venv test-install
source test-install/bin/activate
pip install /Users/john/Documents/project/python/wespeaker-auto-research/Voice-ID/wespeaker-deep-edge-docker-v0.2.0.tar.gz
python -c "
from wespeaker_deep_edge import WespeakerDeep, RecognitionResult
print('Import OK')
client = WespeakerDeep()
assert client.base_url == 'http://127.0.0.1:8005'
print('Default URL OK')
print('All checks passed')
"
deactivate
rm -rf /tmp/test-install
```
Expected: Clean install, no missing dependencies, all checks pass.
