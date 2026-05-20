"""共享工具函数 — 音频加载、VAD、裁剪等。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  音频加载
# --------------------------------------------------------------------------- #


def _load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """读取音频文件 → 单声道 → 目标采样率 → float32 [-1,1] waveform."""
    path = str(Path(path).expanduser())
    logger.debug("Loading audio: %s (target_sr=%d)", path, target_sr)

    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform[0]
    except Exception:
        logger.warning("torchaudio failed for %s, trying librosa", path)

    try:
        import librosa

        arr, _ = librosa.load(path, sr=target_sr, mono=True)
        return torch.from_numpy(arr)
    except Exception as exc:
        raise RuntimeError(f"无法加载音频 {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
#  裁剪
# --------------------------------------------------------------------------- #


def _crop_to_duration(
    waveform: torch.Tensor, max_duration_secs: float, sample_rate: int
) -> torch.Tensor:
    """裁剪音频到指定时长（保留头部）。"""
    max_samples = int(max_duration_secs * sample_rate)
    if waveform.numel() <= max_samples:
        return waveform
    return waveform[:max_samples]


# --------------------------------------------------------------------------- #
#  Silero VAD
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _load_silero_vad():
    """加载 Silero VAD 模型，缓存单例."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    return {
        "model": load_silero_vad(),
        "get_speech_timestamps": get_speech_timestamps,
    }


def _apply_silero_vad(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> torch.Tensor:
    """使用 Silero VAD 去除非人声片段。"""
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad_utils = _load_silero_vad()
    model = vad_utils["model"]

    if waveform.is_floating_point():
        audio = waveform.clamp(-1, 1)
    else:
        audio = waveform.float() / 32768.0
        audio = audio.clamp(-1, 1)

    speech_timestamps = vad_utils["get_speech_timestamps"](
        audio,
        model,
        threshold=0.5,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    if speech_timestamps:
        speech_chunks = []
        for segment in speech_timestamps:
            start = segment["start"]
            end = segment["end"]
            speech_chunks.append(audio[start:end])
        if speech_chunks:
            return torch.cat(speech_chunks)

    return waveform
