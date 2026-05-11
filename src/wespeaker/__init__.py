"""WeSpeaker 声纹识别工具。"""

from .wespeaker import (
    WespeakerClient,
    _crop_verify,
    _extract_embedding,
    _load_audio,
    _load_model,
)

__all__ = [
    "WespeakerClient",
    "_crop_verify",
    "_extract_embedding",
    "_load_audio",
    "_load_model",
]
