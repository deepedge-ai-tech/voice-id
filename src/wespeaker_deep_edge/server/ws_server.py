"""WebSocket 声纹识别服务端。"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import websockets
from websockets.asyncio.server import ServerConnection

from ..wespeaker_deep_dege import WespeakerDeep

from .template_manager import TemplateManager

logger = logging.getLogger(__name__)


class WSServer:
    """WebSocket 声纹识别服务。

    用法:
        server = WSServer(host="0.0.0.0", port=8765)
        await server.start()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 10000,
        storage_dir: str = "./voiceprints",
        model_path: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._storage_dir = Path(storage_dir)
        self._model_path = model_path
        self._wespeaker: WespeakerDeep | None = None
        self._server = None

    async def start(self) -> None:
        """启动 WebSocket 服务。

        加载模型后开始监听连接。此方法会阻塞直到服务停止。
        """
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._wespeaker = WespeakerDeep(model_path=self._model_path)

        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
        )
        logger.info("WSServer started on ws://%s:%d", self._host, self._port)
        await self._server.wait_closed()

    async def stop(self) -> None:
        """优雅停止服务。"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # ------------------------------------------------------------------ #
    #  连接处理
    # ------------------------------------------------------------------ #

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """处理单个 WebSocket 连接。

        每个连接创建独立的 TemplateManager，接收请求并分发处理。
        """
        voiceprints_dir = Path(__file__).parent.parent / "_voiceprints"
        mgr = TemplateManager(
            self._wespeaker,
            str(voiceprints_dir),
            str(self._storage_dir),
        )

        while True:
            try:
                message = await ws.recv()
            except websockets.ConnectionClosed:
                break

            if isinstance(message, str):
                await self._send_error(ws, "格式错误", "INVALID_PARAMS")
                break

            try:
                header = json.loads(message)
            except json.JSONDecodeError:
                await self._send_error(ws, "JSON 解析失败", "INVALID_PARAMS")
                continue

            msg_type = header.get("type")
            logger.debug("收到请求: type=%s", msg_type)

            try:
                if msg_type == "enroll":
                    audio_data = await ws.recv()
                    await self._handle_enroll(ws, mgr, header, audio_data)
                elif msg_type == "load":
                    await self._handle_load(ws, mgr, header)
                elif msg_type == "recognize":
                    audio_data = await ws.recv()
                    await self._handle_recognize(ws, mgr, audio_data)
                else:
                    await self._send_error(ws, f"未知类型: {msg_type}", "INVALID_PARAMS")
            except Exception as e:
                logger.exception("处理请求失败")
                await self._send_error(ws, str(e), "INTERNAL_ERROR")

    # ------------------------------------------------------------------ #
    #  请求处理
    # ------------------------------------------------------------------ #

    async def _handle_enroll(
        self,
        ws: ServerConnection,
        mgr: TemplateManager,
        header: dict,
        audio_data: bytes,
    ) -> None:
        """处理 enroll 请求：注册声纹。"""
        enroll_id = header.get("id")
        if not enroll_id:
            await self._send_error(ws, "缺少 id", "INVALID_PARAMS")
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.write(audio_data)
            tmp.close()

            result = self._wespeaker.enroll(
                str(tmp_path),
                str(self._storage_dir / f"{enroll_id}.pkl"),
            )
            if not result.get("ok"):
                await self._send_error(ws, result.get("error", "注册失败"), "NO_SPEECH")
                return
        finally:
            tmp_path.unlink(missing_ok=True)

        await ws.send(json.dumps({"status": "ok", "data": {"id": enroll_id}}))

    async def _handle_load(
        self,
        ws: ServerConnection,
        mgr: TemplateManager,
        header: dict,
    ) -> None:
        """处理 load 请求：加载声纹模板到内存。"""
        ids = header.get("ids", [])
        if not ids:
            await self._send_error(ws, "ids 为空", "INVALID_PARAMS")
            return

        try:
            loaded = mgr.load(ids)
        except FileNotFoundError as e:
            await self._send_error(ws, str(e), "TEMPLATE_NOT_FOUND")
            return

        await ws.send(
            json.dumps({
                "status": "ok",
                "data": {"loaded": len(loaded), "template_ids": loaded},
            })
        )

    async def _handle_recognize(
        self,
        ws: ServerConnection,
        mgr: TemplateManager,
        audio_data: bytes,
    ) -> None:
        """处理 recognize 请求：识别音频。"""
        if mgr.template_count == 0:
            await self._send_error(ws, "模板库为空", "TEMPLATE_NOT_FOUND")
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.write(audio_data)
            tmp.close()

            test_emb = self._wespeaker._model.extract_embedding(str(tmp_path))
            if test_emb is None:
                await self._send_error(ws, "音频中未检测到有效语音", "NO_SPEECH")
                return
        finally:
            tmp_path.unlink(missing_ok=True)

        best_id, score = mgr.recognize(np.asarray(test_emb.cpu().numpy(), dtype=np.float32))
        await ws.send(
            json.dumps({
                "status": "ok",
                "data": {"id": best_id, "score": round(score, 4)},
            })
        )

    # ------------------------------------------------------------------ #
    #  工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _send_error(ws: ServerConnection, msg: str, code: str = "INTERNAL_ERROR") -> None:
        """发送错误响应。"""
        await ws.send(json.dumps({"status": "error", "error": msg, "code": code}))
