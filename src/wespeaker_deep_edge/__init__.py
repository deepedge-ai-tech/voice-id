"""WeSpeaker 声纹识别工具。

日志配置（导入后使用方自行控制）::

    import logging
    logging.basicConfig(level=logging.INFO)  # 开启所有日志
    # 或只针对本包
    logging.getLogger("wespeaker_deep_edge").setLevel(logging.DEBUG)
    logging.getLogger("wespeaker_deep_edge").addHandler(logging.StreamHandler())
"""

import logging

# 库日志最佳实践：不加 handler，由使用方配置
logging.getLogger("wespeaker_deep_edge").addHandler(logging.NullHandler())

from . import (
    diagnostics,
    realtime_monitor,
    reporters,
)
from .best import (
    BestConfig,
    WespeakerBest,
)
from .wespeaker import (
    WespeakerClient,
    _crop_verify,
    _estimate_snr,
    _extract_embedding,
    _load_audio,
    _load_model,
    _vad_segments,
)
from .wespeaker_deep_dege import (
    DeepConfig,
    WespeakerDeep,
)

__all__ = [
    "WespeakerClient",
    "WespeakerBest",
    "BestConfig",
    "WespeakerDeep",
    "DeepConfig",
    "_crop_verify",
    "_estimate_snr",
    "_extract_embedding",
    "_load_audio",
    "_load_model",
    "_vad_segments",
    "realtime_monitor",
    # 诊断模块
    "diagnostics",
    "reporters",
]
