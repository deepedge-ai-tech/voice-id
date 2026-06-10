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


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_enroll_extracts_speaker_id_from_pkpath(mock_post, client, mock_response, tmp_path):
    """Enroll extracts speaker_id from pk_path filename (voice_X.pkl → X)."""
    mock_post.return_value = mock_response(
        200, {"success": True, "msg": "已登记: frank"}
    )

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-wav-content")

    client.enroll(str(audio_file), "voice_frank.pkl")

    call_kwargs = mock_post.call_args.kwargs
    data = call_kwargs.get("data", {})
    assert "frank" in str(data)


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_enroll_failure_http_error(mock_post, client, mock_response, tmp_path):
    """HTTP error during enrollment raises RuntimeError."""
    resp = mock_response(400, {})
    resp.raise_for_status.side_effect = requests.HTTPError(
        "400 Client Error: Bad Request"
    )
    mock_post.return_value = resp

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-wav-content")

    with pytest.raises(RuntimeError, match="register failed"):
        client.enroll(str(audio_file), "voice_john.pkl")


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_enroll_failure_network_error(mock_post, client, tmp_path):
    """Network error during enrollment propagates."""
    mock_post.side_effect = RuntimeError("register failed for john")

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


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_recognize_http_error(mock_post, client, mock_response, tmp_path):
    """HTTP error during recognition raises RuntimeError."""
    resp = mock_response(500, {})
    resp.raise_for_status.side_effect = requests.HTTPError(
        "500 Server Error: Internal Server Error"
    )
    mock_post.return_value = resp

    client.load_templates(indices=[0])
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-wav-content")

    with pytest.raises(RuntimeError, match="identify failed"):
        client.recognize(str(audio_file))


# --------------------------------------------------------------------------- #
#  recognize_multi_pcm() tests
# --------------------------------------------------------------------------- #


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_recognize_multi_pcm(mock_post, client, mock_response):
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


@patch("src.wespeaker_deep_edge.client.requests.post")
def test_recognize_multi_pcm_no_match(mock_post, client, mock_response):
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
#  _resolve_speaker_id tests
# --------------------------------------------------------------------------- #


def test_resolve_speaker_id_from_pk_path():
    """_resolve_speaker_id extracts name from voice_X.pkl."""
    result = WespeakerDeep._resolve_speaker_id("/audio/test.wav", "voice_john.pkl")
    assert result == "john"


def test_resolve_speaker_id_fallback():
    """_resolve_speaker_id falls back to audio filename stem."""
    result = WespeakerDeep._resolve_speaker_id("/audio/john_test.wav", "voiceprint.pkl")
    assert result == "john_test"


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
