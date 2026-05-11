"""WeSpeaker 声纹识别工具。"""

from . import realtime_monitor
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

__all__ = [
    "WespeakerClient",
    "WespeakerBest",
    "BestConfig",
    "_crop_verify",
    "_estimate_snr",
    "_extract_embedding",
    "_load_audio",
    "_load_model",
    "_vad_segments",
    "realtime_monitor",
]
