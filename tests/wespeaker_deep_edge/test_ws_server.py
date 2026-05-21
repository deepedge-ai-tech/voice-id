"""WSServer 单元测试。

使用 mock WebSocket 连接测试协议处理逻辑。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_ws() -> MagicMock:
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    return ws


@pytest.fixture
def mock_wespeaker() -> MagicMock:
    mock = MagicMock()
    mock.enroll.return_value = {"ok": True, "embedding_dim": 256, "pk_path": "/tmp/test.pkl"}
    mock.load.return_value = np.random.randn(256).astype(np.float32)
    mock._model = MagicMock()
    # Real model returns torch.Tensor; mock with tensor to avoid .cpu() error
    import torch
    mock._model.extract_embedding.return_value = torch.from_numpy(np.random.randn(256).astype(np.float32))
    return mock


@pytest.fixture
def server_with_mocks(mock_wespeaker):
    with patch(
        "src.wespeaker_deep_edge.server.template_manager.Path.is_file",
        return_value=True,
    ):
        from src.wespeaker_deep_edge.server.template_manager import TemplateManager
        from src.wespeaker_deep_edge.server.ws_server import WSServer

        server = WSServer(host="127.0.0.1", port=0, storage_dir="/tmp/ws_test")
        server._wespeaker = mock_wespeaker
        # Build a fresh TemplateManager that bypasses __init__ path checks
        tm = TemplateManager.__new__(TemplateManager)
        tm._wespeaker = mock_wespeaker
        tm._voiceprints_dir = MagicMock()
        tm._storage_dir = MagicMock()
        tm._templates = {
            "preset_john": np.array([1.0] * 256, dtype=np.float32),
        }
        yield server, tm


# ============================================================================ #
#  enroll
# ============================================================================ #


class TestWSServerEnroll:
    async def test_enroll_success(self, server_with_mocks, mock_ws) -> None:
        server, tm = server_with_mocks
        header = {"type": "enroll", "id": "user_001"}
        audio = b"fake_wav_data"

        with patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("pathlib.Path.unlink"):
            mock_tmp.return_value.__enter__.return_value.name = "/tmp/test.wav"
            mock_tmp.return_value.__enter__.return_value.write = MagicMock()
            mock_tmp.return_value.__enter__.return_value.close = MagicMock()
            await server._handle_enroll(mock_ws, tm, header, audio)

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["status"] == "ok"
        assert sent["data"]["id"] == "user_001"

    async def test_enroll_missing_id(self, server_with_mocks, mock_ws) -> None:
        server, tm = server_with_mocks
        await server._handle_enroll(mock_ws, tm, {"type": "enroll"}, b"")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["status"] == "error"
        assert sent["code"] == "INVALID_PARAMS"

    async def test_enroll_failure(self, server_with_mocks, mock_ws) -> None:
        server, tm = server_with_mocks
        server._wespeaker.enroll.return_value = {"ok": False, "error": "注册失败"}

        with patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("pathlib.Path.unlink"):
            mock_tmp.return_value.__enter__.return_value.name = "/tmp/test.wav"
            mock_tmp.return_value.__enter__.return_value.write = MagicMock()
            mock_tmp.return_value.__enter__.return_value.close = MagicMock()
            await server._handle_enroll(mock_ws, tm, {"type": "enroll", "id": "u1"}, b"data")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["status"] == "error"


# ============================================================================ #
#  load
# ============================================================================ #


class TestWSServerLoad:
    async def test_load_success(self, server_with_mocks, mock_ws) -> None:
        server, tm = server_with_mocks
        tm.load = MagicMock(return_value=["preset_john"])
        await server._handle_load(mock_ws, tm, {"type": "load", "ids": ["preset_john"]})

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["status"] == "ok"
        assert sent["data"]["loaded"] == 1
        assert sent["data"]["template_ids"] == ["preset_john"]

    async def test_load_empty_ids(self, server_with_mocks, mock_ws) -> None:
        server, tm = server_with_mocks
        await server._handle_load(mock_ws, tm, {"type": "load", "ids": []})

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["status"] == "error"
        assert sent["code"] == "INVALID_PARAMS"


# ============================================================================ #
#  recognize
# ============================================================================ #


class TestWSServerRecognize:
    async def test_recognize_success(self, server_with_mocks, mock_ws) -> None:
        server, tm = server_with_mocks
        tm.recognize = MagicMock(return_value=("preset_john", 0.85))

        with patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("pathlib.Path.unlink"):
            mock_tmp.return_value.__enter__.return_value.name = "/tmp/test.wav"
            mock_tmp.return_value.__enter__.return_value.write = MagicMock()
            mock_tmp.return_value.__enter__.return_value.close = MagicMock()
            await server._handle_recognize(mock_ws, tm, b"audio_data")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["status"] == "ok"
        assert sent["data"]["id"] == "preset_john"
        assert sent["data"]["score"] == 0.85

    async def test_recognize_empty_templates(self, server_with_mocks, mock_ws) -> None:
        server, tm = server_with_mocks
        tm._templates = {}
        await server._handle_recognize(mock_ws, tm, b"audio_data")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["status"] == "error"
        assert sent["code"] == "TEMPLATE_NOT_FOUND"

    async def test_recognize_no_speech(self, server_with_mocks, mock_ws) -> None:
        server, tm = server_with_mocks
        server._wespeaker._model.extract_embedding.return_value = None

        with patch("tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("pathlib.Path.unlink"):
            mock_tmp.return_value.__enter__.return_value.name = "/tmp/test.wav"
            mock_tmp.return_value.__enter__.return_value.write = MagicMock()
            mock_tmp.return_value.__enter__.return_value.close = MagicMock()
            await server._handle_recognize(mock_ws, tm, b"audio_data")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["status"] == "error"
        assert sent["code"] == "NO_SPEECH"
