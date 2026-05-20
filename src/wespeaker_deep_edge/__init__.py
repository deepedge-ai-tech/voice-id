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

from . import diagnostics, realtime_monitor, reporters
from .wespeaker_deep_dege import DeepConfig, WespeakerDeep

__all__ = [
    "DeepConfig",
    "WespeakerDeep",
    "realtime_monitor",
    "diagnostics",
    "reporters",
]
