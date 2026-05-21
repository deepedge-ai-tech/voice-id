"""服务端 CLI 入口。

用法:
    python -m wespeaker_deep_edge.server --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from .ws_server import WSServer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WeSpeaker WebSocket 声纹识别服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=10000, help="监听端口")
    parser.add_argument(
        "--storage-dir",
        default="./voiceprints",
        help="声纹文件存储目录",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="模型路径（None 使用内置模型）",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="推理设备 (cpu/cuda)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = WSServer(
        host=args.host,
        port=args.port,
        storage_dir=args.storage_dir,
        model_path=args.model_path,
        device=args.device,
    )
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
