"""SpeakerClient 单元测试。

mock websockets.connect 来验证协议编码和解码逻辑。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.wespeaker_deep_edge.client.speaker_client import SpeakerClient, SpeakerClientError


@pytest.fixture
def mock_ws() -> AsyncMock:
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def client(mock_ws) -> SpeakerClient:
    with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
        c = SpeakerClient("ws://localhost:10000")
        c._ws = mock_ws
        return c


# ============================================================================ #
#  connect
# ============================================================================ #


class TestSpeakerClientConnect:
    async def test_connect_success(self, mock_ws) -> None:
        with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
            client = SpeakerClient("ws://localhost:10000")
            await client.connect()
            assert client._ws is mock_ws

    async def test_connect_failure_raises(self) -> None:
        with patch("websockets.connect", side_effect=ConnectionError("refused")):
            client = SpeakerClient("ws://localhost:10000")
            with pytest.raises(SpeakerClientError, match="连接失败"):
                await client.connect()


# ============================================================================ #
#  enroll
# ============================================================================ #


class TestSpeakerClientEnroll:
    async def test_enroll_success(self, client, mock_ws) -> None:
        mock_ws.recv.return_value = json.dumps({"status": "ok", "data": {"id": "user_001"}})

        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.read_bytes", return_value=b"audio"):
            result = await client.enroll("/tmp/test.wav", "user_001")

        assert result == "user_001"
        # Verify protocol: JSON header + audio data
        header_sent = json.loads(mock_ws.send.call_args_list[0][0][0])
        assert header_sent["type"] == "enroll"
        assert header_sent["id"] == "user_001"
        assert mock_ws.send.call_args_list[1][0][0] == b"audio"

    async def test_enroll_missing_file(self, client) -> None:
        with patch("pathlib.Path.is_file", return_value=False):
            with pytest.raises(SpeakerClientError, match="文件不存在"):
                await client.enroll("/tmp/nonexistent.wav", "u1")


# ============================================================================ #
#  load
# ============================================================================ #


class TestSpeakerClientLoad:
    async def test_load_success(self, client, mock_ws) -> None:
        mock_ws.recv.return_value = json.dumps({
            "status": "ok",
            "data": {"loaded": 2, "template_ids": ["preset_john", "user_001"]},
        })

        result = await client.load(["preset_john", "user_001"])
        assert result == ["preset_john", "user_001"]

        header_sent = json.loads(mock_ws.send.call_args[0][0])
        assert header_sent["type"] == "load"
        assert header_sent["ids"] == ["preset_john", "user_001"]


# ============================================================================ #
#  recognize
# ============================================================================ #


class TestSpeakerClientRecognize:
    async def test_recognize_success(self, client, mock_ws) -> None:
        mock_ws.recv.return_value = json.dumps({
            "status": "ok",
            "data": {"id": "preset_john", "score": 0.8523},
        })

        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.read_bytes", return_value=b"audio"):
            result = await client.recognize("/tmp/test.wav")

        assert result["id"] == "preset_john"
        assert result["score"] == 0.8523

    async def test_recognize_missing_file(self, client) -> None:
        with patch("pathlib.Path.is_file", return_value=False):
            with pytest.raises(SpeakerClientError, match="文件不存在"):
                await client.recognize("/tmp/nonexistent.wav")


# ============================================================================ #
#  error handling
# ============================================================================ #


class TestSpeakerClientErrors:
    async def test_server_error_raises(self, client, mock_ws) -> None:
        mock_ws.recv.return_value = json.dumps({
            "status": "error",
            "error": "模板库为空",
            "code": "TEMPLATE_NOT_FOUND",
        })

        with pytest.raises(SpeakerClientError, match="模板库为空"):
            await client.load(["nonexistent"])

    async def test_not_connected_raises(self) -> None:
        client = SpeakerClient("ws://localhost:10000")
        client._ws = None
        with pytest.raises(SpeakerClientError, match="未连接"):
            await client.enroll("/tmp/test.wav", "u1")


# ============================================================================ #
#  close
# ============================================================================ #


class TestSpeakerClientClose:
    async def test_close_calls_ws_close(self, client, mock_ws) -> None:
        await client.close()
        mock_ws.close.assert_awaited_once()
        assert client._ws is None

    async def test_close_idempotent(self, client) -> None:
        client._ws = None
        await client.close()  # should not raise
