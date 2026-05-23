"""WeSpeaker 声纹识别工具 — 使用官方 wespeaker SimAM_ResNet34_ASP。

日志配置（导入后使用方自行控制）::

    import logging
    logging.basicConfig(level=logging.INFO)  # 开启所有日志
    logging.getLogger("wespeaker_deep_edge").setLevel(logging.DEBUG)
"""

import sys
from pathlib import Path

# vendored wespeaker 包（位于 _wespeaker/）
_vendored = str(Path(__file__).parent / "_wespeaker")
if _vendored not in sys.path:
    sys.path.insert(0, _vendored)

import logging

logging.getLogger("wespeaker_deep_edge").addHandler(logging.NullHandler())

from .onnx_engine import OnnxConfig, OnnxEngine

# WespeakerDeep 默认使用 ONNX Runtime 轻量版本。
# diagnostics / realtime_monitor / reporters / DeepConfig 需要 PyTorch，
# 改为懒导入，不触发 import torch。
WespeakerDeep = OnnxEngine


def __getattr__(name: str):
    import importlib

    lazy: dict[str, str] = {
        "DeepConfig": "wespeaker_deep_dege",
        "diagnostics": "diagnostics",
        "realtime_monitor": "realtime_monitor",
        "reporters": "reporters",
    }
    if name in lazy:
        mod = importlib.import_module(f".{lazy[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DeepConfig",
    "OnnxConfig",
    "OnnxEngine",
    "WespeakerDeep",
    "realtime_monitor",
    "diagnostics",
    "reporters",
]
