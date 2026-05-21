"""WebSocket 声纹客户端 SDK。

封装 WebSocket 连接细节，调用方无需关心协议和连接管理。
一次 connect()，可多次调用 enroll/load/recognize。

用法:
    client = SpeakerClient("ws://localhost:8765")
    await client.connect()
    await client.enroll("audio.wav", "user_001")
    await client.load(["user_001", "preset_john"])
    result = await client.recognize("test.wav")
    # → {"id": "preset_john", "score": 0.8523}
    await client.close()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import websockets

logger = logging.getLogger(__name__)


class SpeakerClientError(Exception):
    """客户端异常基类。

    所有与 WebSocket 通信相关的异常都转换为此类，
    调用方无需捕获 WebSocket 底层异常。
    """


class SpeakerClient:
    """WebSocket 声纹客户端 SDK。"""

    def __init__(self, url: str = "ws://localhost:10000", auto_reconnect: bool = True) -> None:
        self._url = url
        self._auto_reconnect = auto_reconnect
        self._ws: WebSocketClientProtocol | None = None

    async def connect(self) -> None:
        """建立 WebSocket 连接。

        Raises:
            SpeakerClientError: 连接失败。
        """
        try:
            self._ws = await websockets.connect(self._url)
        except Exception as e:
            raise SpeakerClientError(f"连接失败: {e}") from e

    async def enroll(self, audio_path: str, id: str) -> str:
        """注册声纹。

        Args:
            audio_path: 本地音频文件路径。
            id: 声纹标识符。

        Returns:
            注册成功的 id。

        Raises:
            SpeakerClientError: 注册失败。
        """
        data = await self._send_request("enroll", audio_path=audio_path, id=id)
        return data["id"]

    async def load(self, ids: list[str]) -> list[str]:
        """加载声纹模板到服务端内存。

        Args:
            ids: 声纹 ID 列表（``preset_`` 前缀为内置声纹）。

        Returns:
            实际加载的模板 ID 列表。

        Raises:
            SpeakerClientError: 加载失败。
        """
        data = await self._send_request("load", ids=ids)
        return data["template_ids"]

    async def recognize(self, audio_path: str) -> dict:
        """识别音频。

        Args:
            audio_path: 本地音频文件路径。

        Returns:
            {"id": str, "score": float} 最高分模板 ID 和分数。

        Raises:
            SpeakerClientError: 识别失败。
        """
        return await self._send_request("recognize", audio_path=audio_path)

    async def close(self) -> None:
        """关闭连接。"""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send_request(
        self,
        msg_type: str,
        audio_path: str | None = None,
        **kwargs,
    ) -> dict:
        """发送请求并接收响应。

        Args:
            msg_type: 消息类型（enroll/load/recognize）。
            audio_path: 可选的音频文件路径。
            **kwargs: 附加到请求头部的参数。

        Returns:
            响应中的 data 字段。

        Raises:
            SpeakerClientError: 通信错误或服务端返回错误。
        """
        if self._ws is None:
            raise SpeakerClientError("未连接，请先调用 connect()")

        header = {"type": msg_type, **kwargs}
        try:
            if audio_path:
                path = Path(audio_path)
                if not path.is_file():
                    raise SpeakerClientError(f"文件不存在: {audio_path}")
                audio_data = path.read_bytes()
                await self._ws.send(json.dumps(header).encode())
                await self._ws.send(audio_data)
            else:
                await self._ws.send(json.dumps(header).encode())

            resp = json.loads(await self._ws.recv())
        except SpeakerClientError:
            raise
        except Exception as e:
            raise SpeakerClientError(str(e)) from e

        if resp.get("status") == "error":
            raise SpeakerClientError(resp.get("error", "未知错误"))
        return resp.get("data", {})
