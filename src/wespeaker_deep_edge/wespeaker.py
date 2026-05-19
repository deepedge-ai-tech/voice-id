"""
WespeakerClient — 独立的 WeSpeaker 声纹注册与识别工具。

零项目依赖，仅需安装: torch, torchaudio, numpy, audiomentations(可选)

用法:
    client = WespeakerClient()
    client.mp3_to_pk("voice.mp3", "voice.pkl")
    result = client.recognize("voice2.mp3", "voice.pkl")
"""

from __future__ import annotations

import logging
import math
import os
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_default_model_path() -> str:
    """返回内置模型路径（优先），fallback 到 ./models/wespeaker。"""
    try:
        return str(
            resources.files("wespeaker_deep_edge._models") / "wespeaker"
        )
    except (ModuleNotFoundError, TypeError):
        return "./models/wespeaker"


# --------------------------------------------------------------------------- #
#  音频加载
# --------------------------------------------------------------------------- #


def _load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """读取音频文件 → 单声道 → 目标采样率 → float32 [-1,1] waveform."""
    path = str(Path(path).expanduser())
    logger.debug("Loading audio: %s (target_sr=%d)", path, target_sr)

    # 优先 torchaudio（支持 mp3/wav/ogg，需 ffmpeg）
    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)
        logger.debug("Loaded with torchaudio: sr=%d, shape=%s", sr, waveform.shape)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform = waveform[0]  # 使用左声道，避免相位抵消和 AGC 差异
        return waveform
    except Exception:
        logger.warning("torchaudio failed for %s, trying librosa", path)
        pass
    # fallback: librosa
    try:
        import librosa

        arr, _ = librosa.load(path, sr=target_sr, mono=True)
        return torch.from_numpy(arr)
    except Exception as exc:
        raise RuntimeError(f"无法加载音频 {path}: {exc}") from exc


def _crop_to_duration(
    waveform: torch.Tensor, max_duration_secs: float, sample_rate: int
) -> torch.Tensor:
    """裁剪音频到指定时长（保留头部）。

    Args:
        waveform: 音频波形 tensor
        max_duration_secs: 最大时长（秒）
        sample_rate: 采样率

    Returns:
        裁剪后的音频，如果原音频短于 max_duration_secs 则返回原音频
    """
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
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        return {
            "model": load_silero_vad(),
            "get_speech_timestamps": get_speech_timestamps,
        }
    except ImportError as exc:
        raise ImportError("未安装 silero-vad: uv sync") from exc


def _apply_silero_vad(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> torch.Tensor:
    """使用 Silero VAD 去除非人声片段。

    Args:
        waveform: 音频波形 tensor (1D)
        sample_rate: 采样率
        min_speech_duration_ms: 最小语音持续时长（毫秒）
        min_silence_duration_ms: 最小静音持续时长（毫秒）
        speech_pad_ms: 语音边界填充（毫秒）

    Returns:
        去除静音后的音频波形 tensor (1D)
    """
    vad_utils = _load_silero_vad()
    model = vad_utils["model"]

    # 转换为正确的格式 (float32, [-1, 1])
    if waveform.is_floating_point():
        audio = waveform.clamp(-1, 1)
    else:
        audio = waveform.float() / 32768.0
        audio = audio.clamp(-1, 1)

    # 获取语音时间戳
    speech_timestamps = vad_utils["get_speech_timestamps"](
        audio,
        model,
        threshold=0.5,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    # 提取语音片段
    if speech_timestamps:
        speech_chunks = []
        for segment in speech_timestamps:
            start = segment["start"]
            end = segment["end"]
            speech_chunks.append(audio[start:end])
        if speech_chunks:
            return torch.cat(speech_chunks)

    return waveform  # 如果没有检测到语音，返回原音频


# --------------------------------------------------------------------------- #
#  模型加载 (pyannote.audio 后端)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _load_model(model_path: str, device: str) -> torch.nn.Module:
    """加载 WeSpeaker ResNet34 模型，缓存单例.

    model_path 为空字符串时，自动使用默认模型路径。
    """
    if not model_path:
        model_path = get_default_model_path()

    try:
        from pyannote.audio import Model
    except ImportError as exc:
        raise ImportError("未安装 pyannote.audio: pip install pyannote.audio") from exc

    # 兼容旧版 torch.load(weights_only=False)
    _real = torch.load

    def _load(*a, **kw):
        kw["weights_only"] = False
        return _real(*a, **kw)

    torch.load = _load  # type: ignore[assignment]

    dev = torch.device(device)

    # 如果 model_path 是本地目录，指向具体的 .bin 文件
    # （新版 huggingface_hub 不接受目录路径作为 repo_id）
    if os.path.isdir(model_path):
        bin_files = [f for f in os.listdir(model_path) if f.endswith(".bin")]
        if bin_files:
            model_path = os.path.join(model_path, bin_files[0])

    model = Model.from_pretrained(model_path)
    model.eval()
    model.to(dev)
    return model


def _extract_embedding(model: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """从 waveform 提取 256 维 embedding."""
    device = next(model.parameters()).device
    logger.debug("Extracting embedding: waveform shape=%s", waveform.shape)
    with torch.no_grad():
        x = waveform.to(device).unsqueeze(0).unsqueeze(0)  # (1,1,T)
        emb = model(x)  # (1, 256)
        logger.debug("Extracted embedding shape: %s", emb.shape)
        return emb.squeeze(0).cpu()


# --------------------------------------------------------------------------- #
#  噪声增强 (可选)
# --------------------------------------------------------------------------- #


class _NoiseAugmentor:
    def __init__(
        self,
        sample_rate: int = 16000,
        augment_ratio: float = 0.6,
        noise_dir: str = "",
        seed: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.augment_ratio = augment_ratio
        import random

        self._rand = random.Random(seed)

        try:
            from audiomentations import (
                AddBackgroundNoise,
                AddGaussianSNR,
                AddShortNoises,
            )
        except ImportError:
            self._available = False
            return
        self._available = True
        noise_path = noise_dir if Path(noise_dir).is_dir() else None
        has_noise = bool(noise_path and any(Path(noise_path).rglob("*.wav")))
        self._gaussian = AddGaussianSNR(min_snr_db=5, max_snr_db=20, p=1.0)
        self._background = (
            AddBackgroundNoise(sounds_path=noise_path, min_snr_db=5, max_snr_db=20, p=1.0)
            if has_noise
            else None
        )
        self._short = (
            AddShortNoises(sounds_path=noise_path, min_snr_db=3, max_snr_db=18, p=1.0)
            if has_noise
            else None
        )

    def augment(self, segments: list[np.ndarray]) -> list[np.ndarray]:
        if not self._available or self.augment_ratio <= 0:
            return [np.asarray(s, dtype=np.float32) for s in segments]
        n = len(segments)
        k = max(0, min(int(round(n * self.augment_ratio)), n))
        aug_idx = set(self._rand.sample(range(n), k=k)) if k > 0 else set()
        out = []
        for i, seg in enumerate(segments):
            arr = np.asarray(seg, dtype=np.float32)
            if i in aug_idx:
                choice = np.random.choice(["background", "short", "gaussian"], p=[0.5, 0.2, 0.3])
                if choice == "background" and self._background:
                    arr = self._background(samples=arr, sample_rate=self.sample_rate)
                elif choice == "short" and self._short:
                    arr = self._short(samples=arr, sample_rate=self.sample_rate)
                else:
                    arr = self._gaussian(samples=arr, sample_rate=self.sample_rate)
            out.append(np.asarray(arr, dtype=np.float32))
        return out


# --------------------------------------------------------------------------- #
#  SNR 估计 & VAD
# --------------------------------------------------------------------------- #


def _estimate_snr(
    waveform: torch.Tensor, sample_rate: int = 16000, frame_ms: int = 25, hop_ms: int = 10
) -> float:
    """估计音频片段的信噪比（dB）。

    将信号分帧，以能量最低的 10% 帧作为噪声估计，
    其余帧的平均能量与噪声能量之比即为 SNR。
    """
    frame_len = int(frame_ms / 1000 * sample_rate)
    hop_len = int(hop_ms / 1000 * sample_rate)
    if frame_len < 1:
        frame_len = 160
        hop_len = 80

    frames = []
    for start in range(0, len(waveform) - frame_len + 1, hop_len):
        seg = waveform[start : start + frame_len]
        energy = (seg**2).mean().item()
        if energy > 1e-10:
            frames.append(energy)

    if not frames:
        return 0.0

    frames.sort()
    n_noise = max(1, len(frames) // 10)
    noise_energy = np.mean(frames[:n_noise])
    signal_energy = np.mean(frames[n_noise:])

    if noise_energy < 1e-10:
        return 30.0  # cap
    return round(10 * np.log10(signal_energy / noise_energy), 2)


def _vad_segments(
    waveform: torch.Tensor,
    rms_threshold: float = 0.005,
    min_duration_ms: int = 100,
    sample_rate: int = 16000,
) -> list[torch.Tensor]:
    """基于 RMS 能量的语音活动检测，返回有语音的片段列表。

    使用自适应阈值：以 RMS 的 25 分位数作为噪声底噪估计，
    阈值 = noise_floor * adaptive_factor（默认 2.0 倍）。
    如果计算出的阈值低于固定下限，则使用固定下限。
    """
    frame_len = int(0.02 * sample_rate)  # 20ms
    hop_len = int(0.01 * sample_rate)  # 10ms
    min_samples = int(min_duration_ms / 1000 * sample_rate)

    rms_values = []
    starts = []
    for start in range(0, len(waveform) - frame_len + 1, hop_len):
        seg = waveform[start : start + frame_len]
        rms = float(torch.sqrt((seg**2).mean()))
        rms_values.append(rms)
        starts.append(start)

    if not rms_values:
        return [waveform]

    # 自适应阈值：25 分位数 * 2.0，不低于固定下限
    sorted_rms = sorted(rms_values)
    noise_floor = sorted_rms[max(0, len(sorted_rms) // 4)]
    adaptive_threshold = max(noise_floor * 2.0, rms_threshold)

    # 合并连续的高能量帧
    segments: list[torch.Tensor] = []
    in_speech = False
    seg_start = 0
    for i, rms in enumerate(rms_values):
        if rms >= adaptive_threshold and not in_speech:
            in_speech = True
            seg_start = starts[i]
        elif (rms < adaptive_threshold or i == len(rms_values) - 1) and in_speech:
            in_speech = False
            seg_end = starts[i] + frame_len
            if seg_end - seg_start >= min_samples:
                segments.append(waveform[seg_start:seg_end])

    return segments if segments else [waveform]


# --------------------------------------------------------------------------- #
#  动态阈值
# --------------------------------------------------------------------------- #


def get_dynamic_threshold(vad_duration: float) -> float:
    """根据 VAD 后音频时长返回推荐阈值.

    基于 5x5 交叉测试实验数据（80 个测试用例）优化：
    - 测试说话人: John, Xixi, Frank, Qingqing, Zhong
    - 测试时间: 2026-05-13
    - 数据来源: experiment_log/cross_test_data_20260513_181002.json

    Args:
        vad_duration: VAD 处理后的音频时长（秒）

    Returns:
        推荐的相似度阈值

    时长分组与阈值:
        < 0.5s:  0.22  (极短音频，大幅降低阈值避免误拒)
        0.5-1.0s: 0.35  (短音频，适度降低阈值)
        1.0-1.5s: 0.50  (中等长度，标准阈值)
        >= 1.5s: 0.55  (长音频，保持高安全性)
    """
    if vad_duration < 0.5:
        return 0.22
    elif vad_duration < 1.0:
        return 0.35
    elif vad_duration < 1.5:
        return 0.50
    else:
        return 0.55


def get_dynamic_threshold_smooth(vad_duration: float) -> float:
    """平滑动态阈值公式（基于对数）.

    基准阈值 0.36，根据时长对数调整，范围限制在 [0.22, 0.55]。

    Args:
        vad_duration: VAD 处理后的音频时长（秒）

    Returns:
        推荐的相似度阈值
    """
    base = 0.36
    duration_factor = 0.12 * math.log(max(vad_duration, 0.3))
    return max(0.22, min(0.55, base + duration_factor))


def get_score_compensation_factor(vad_duration: float, target_duration: float = 2.0) -> float:
    """根据 VAD 后音频时长计算分数补偿系数.

    短音频的相似度分数会被补偿（乘以大于1的系数），
    使得不同时长的音频可以在同一阈值下公平比较。

    Args:
        vad_duration: VAD 处理后的音频时长（秒）
        target_duration: 目标标准时长（秒），默认 2.0 秒

    Returns:
        补偿系数，范围 [1.0, 1.5]

    示例:
        2.0s → 1.0  (标准时长，不补偿)
        1.5s → 1.08
        1.0s → 1.17
        0.5s → 1.33
        0.3s → 1.5  (极短音频，最大补偿)
    """
    if vad_duration >= target_duration:
        return 1.0

    # 线性补偿：系数 = 1 + (target - actual) / target * max_bonus
    # 限制最大补偿为 1.5
    max_bonus = 0.5
    factor = 1.0 + (target_duration - vad_duration) / target_duration * max_bonus
    return min(1.5, max(1.0, factor))


# --------------------------------------------------------------------------- #
#  主类
# --------------------------------------------------------------------------- #


@dataclass
class WespeakerClient:
    """WeSpeaker 声纹注册与识别。

    所有参数可在构造时或运行时通过属性调整。
    """

    # ---- 模型 ----
    model_path: str = field(default_factory=get_default_model_path)
    device: str = "cpu"

    # ---- 注册 ----
    enrollment_segment_secs: float = 1.0
    enable_augmentation: bool = True
    augment_ratio: float = 0.6
    noise_dir: str = ""
    sample_rate: int = 16000
    enable_snr_weighting: bool = True  # 注册时按 SNR 加权
    enable_multi_scale_enrollment: bool = (
        True  # 注册时加入多尺度短语音 embedding（0.3s, 0.5s, 0.8s）
    )
    multi_scale_durations: list[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.8]
    )  # 短语音时长（秒）
    multi_scale_crops_per_duration: int = 3  # 每个时长随机 crop 的数量

    # ---- 识别 ----
    sim_threshold: float = 0.55
    enable_dynamic_threshold: bool = False  # 启用基于 VAD 时长的动态阈值
    dynamic_threshold_mode: str = "segmented"  # segmented / smooth
    verify_crop_mode: str = "full_utterance"  # full_utterance / tail_window / head_window
    verify_window_secs: float = 1.0
    verify_buffer_keep_secs: float = 8.0  # 验证 buffer 最大保留时长
    enable_vad: bool = True  # 识别时启用 VAD 去静音
    vad_rms_threshold: float = 0.005  # VAD 能量阈值
    # ---- 滑动窗口识别 ----
    enable_sliding_window: bool = True  # 启用滑动窗口多窗口识别
    sliding_window_secs: float = 0.6  # 滑动窗口长度（秒）
    sliding_hop_secs: float = 0.2  # 滑动窗口步长（秒）
    sliding_score_mode: str = "max"  # 分数聚合模式: max 或 top_k_mean
    # ---- 内置声纹 ----
    package_pk_index: int | None = None  # 内置声纹索引: 0=John 1=Frank 2=Michael 3=Qingqing 4=Xixi 5=Zhong

    # ---- 内部 ----
    _model: Optional[torch.nn.Module] = field(init=False, default=None, repr=False)
    _aug: Optional[_NoiseAugmentor] = field(init=False, default=None, repr=False)

    # ------------------------------------------------------------------ #
    #  公开 API
    # ------------------------------------------------------------------ #

    def mp3_to_pk(self, mp3_path: str, pk_path: str) -> dict:
        """从音频文件注册声纹 → 保存 .pkl 文件."""
        logger.info("Enrolling voiceprint from %s", mp3_path)
        if not Path(mp3_path).is_file():
            return {"ok": False, "error": f"文件不存在: {mp3_path}"}

        self._ensure_model()
        waveform = _load_audio(mp3_path, self.sample_rate)

        # 切段
        seg_len = int(self.enrollment_segment_secs * self.sample_rate)
        if waveform.numel() < seg_len:
            return {"ok": False, "error": "音频太短，无法切分出有效片段"}
        segments = [
            waveform[i * seg_len : (i + 1) * seg_len].cpu().numpy()
            for i in range(len(waveform) // seg_len)
        ]

        # 多尺度短语音 embedding（解决 domain mismatch）
        if self.enable_multi_scale_enrollment:
            import random

            random.seed(42)  # 固定种子保证可重复性
            for duration in self.multi_scale_durations:
                crop_samples = int(duration * self.sample_rate)
                if waveform.numel() < crop_samples:
                    continue
                for _ in range(self.multi_scale_crops_per_duration):
                    start = random.randint(0, waveform.numel() - crop_samples)
                    end = start + crop_samples
                    segment = waveform[start:end].cpu().numpy()
                    segments.append(segment)
            logger.debug(
                "Multi-scale enrollment: added %d short segments, total %d",
                len(self.multi_scale_durations) * self.multi_scale_crops_per_duration,
                len(segments),
            )

        # 增强
        if self.enable_augmentation:
            segments = self._augmentor().augment(segments)

        # 提取 embedding
        embeddings = []
        for seg in segments:
            t = torch.from_numpy(seg)
            emb = _extract_embedding(self._model, t)
            embeddings.append(emb)

        # 均值 + 归一化（可选 SNR 加权）
        if self.enable_snr_weighting:
            snr_values = [
                _estimate_snr(torch.from_numpy(seg), sample_rate=self.sample_rate)
                for seg in segments
            ]
            weights = torch.tensor([max(s, 0.0) for s in snr_values], dtype=torch.float32)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                stacked = torch.stack(embeddings, dim=0)
                mean_emb = F.normalize((stacked * weights.unsqueeze(1)).sum(dim=0), dim=0)
            else:
                mean_emb = F.normalize(torch.stack(embeddings, dim=0).mean(dim=0), dim=0)
        else:
            mean_emb = F.normalize(torch.stack(embeddings, dim=0).mean(dim=0), dim=0)

        # 保存
        out = Path(pk_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(mean_emb.cpu().numpy(), f)

        logger.info("Voiceprint enrolled: %d segments, dim=%d", len(segments), mean_emb.numel())
        return {
            "ok": True,
            "num_segments": len(segments),
            "embedding_dim": mean_emb.numel(),
            "pk_path": str(out.resolve()),
        }

    def recognize(self, audio_path: str, pk_path: str | None = None) -> dict:
        """将音频与已保存的声纹比对，返回识别结果.

        pk_path 为 None 时默认使用内置 John 声纹 (index 0)。
        若 package_pk_index 已设置则优先使用。
        """
        # 解析声纹路径: package_pk_index > pk_path > 默认 John
        if self.package_pk_index is not None:
            from ._voiceprints import get_voiceprint_path

            pk_path = get_voiceprint_path(self.package_pk_index)
        elif pk_path is None:
            from ._voiceprints import get_voiceprint_path

            pk_path = get_voiceprint_path(0)

        logger.info("Recognizing %s against %s", audio_path, pk_path)
        if not Path(audio_path).is_file():
            return {"is_recognized": False, "confidence": 0.0, "error": f"文件不存在: {audio_path}"}
        if not Path(pk_path).is_file():
            return {
                "is_recognized": False,
                "confidence": 0.0,
                "error": f"声纹文件不存在: {pk_path}",
            }

        self._ensure_model()

        # 加载参考声纹
        with open(pk_path, "rb") as f:
            ref = F.normalize(torch.from_numpy(np.asarray(pickle.load(f), dtype=np.float32)), dim=0)

        # 加载音频 + 限制长度 + 裁剪
        waveform = _load_audio(audio_path, self.sample_rate)
        original_duration = waveform.numel() / self.sample_rate
        max_samples = int(self.verify_buffer_keep_secs * self.sample_rate)
        if waveform.numel() > max_samples:
            if self.verify_crop_mode == "head_window":
                waveform = waveform[:max_samples]
            else:
                waveform = waveform[-max_samples:]

        # VAD 去静音/去纯噪声
        vad_duration = original_duration
        if self.enable_vad and self.verify_crop_mode == "full_utterance":
            speech_segs = _vad_segments(
                waveform, rms_threshold=self.vad_rms_threshold, sample_rate=self.sample_rate
            )
            if speech_segs and len(speech_segs) > 0:
                pcm = torch.cat(speech_segs)
                vad_duration = pcm.numel() / self.sample_rate
            else:
                pcm = waveform
        else:
            pcm = _crop_verify(
                waveform, self.verify_crop_mode, self.verify_window_secs, self.sample_rate
            )
        if pcm.numel() == 0:
            return {"is_recognized": False, "confidence": 0.0, "error": "音频太短"}

        # 计算动态阈值
        threshold = self.sim_threshold
        if self.enable_dynamic_threshold:
            if self.dynamic_threshold_mode == "smooth":
                threshold = get_dynamic_threshold_smooth(vad_duration)
            else:
                threshold = get_dynamic_threshold(vad_duration)
            logger.debug(
                "Using dynamic threshold: %.4f (VAD duration: %.2fs)", threshold, vad_duration
            )

        # 提取 embedding + 比对
        if self.enable_sliding_window and pcm.numel() >= int(
            self.sliding_window_secs * self.sample_rate
        ):
            score, all_scores = _sliding_window_scores(
                pcm,
                ref,
                self._model,
                self.sliding_window_secs,
                self.sliding_hop_secs,
                self.sample_rate,
                self.sliding_score_mode,
            )
            # 对于滑动窗口，使用窗口时长而不是 VAD 时长来计算阈值
            effective_duration = self.sliding_window_secs
        else:
            emb = _extract_embedding(self._model, pcm)
            emb = F.normalize(emb, dim=0)
            score = float(torch.dot(emb, ref).clamp(-1.0, 1.0).item())
            all_scores = [score]
            effective_duration = vad_duration

        # 如果启用了滑动窗口，重新计算基于窗口时长的动态阈值
        if self.enable_sliding_window and self.enable_dynamic_threshold:
            if self.dynamic_threshold_mode == "smooth":
                threshold = get_dynamic_threshold_smooth(effective_duration)
            else:
                threshold = get_dynamic_threshold(effective_duration)
            logger.debug(
                "Using sliding-window-aware threshold: %.4f (duration: %.2fs)",
                threshold,
                effective_duration,
            )

        logger.debug("Recognition score: %.4f (threshold=%.2f)", score, threshold)

        saved_path = _debug_save_test_audio(waveform, self.sample_rate, score)

        return {
            "is_recognized": score >= threshold,
            "confidence": round(score, 4),
            "threshold": threshold,
            "vad_duration": round(vad_duration, 2),
            "all_scores": [round(s, 4) for s in all_scores],
            "num_windows": len(all_scores),
            "debug_audio": str(saved_path) if saved_path else None,
        }

    # ------------------------------------------------------------------ #
    #  内部
    # ------------------------------------------------------------------ #

    def enroll_mixed(self, clean_paths: list[str], noise_paths: list[str], pk_path: str) -> dict:
        """混合注册：结合 clean 和 noisy 音频生成声纹。"""
        self._ensure_model()
        all_segments: list[torch.Tensor] = []
        seg_len = int(self.enrollment_segment_secs * self.sample_rate)

        for path in clean_paths + noise_paths:
            if not Path(path).is_file():
                continue
            waveform = _load_audio(path, self.sample_rate)
            if waveform.numel() < seg_len:
                continue
            # 原始长片段
            for i in range(len(waveform) // seg_len):
                all_segments.append(waveform[i * seg_len : (i + 1) * seg_len])

            # 多尺度短语音
            if self.enable_multi_scale_enrollment:
                import random

                random.seed(42)
                for duration in self.multi_scale_durations:
                    crop_samples = int(duration * self.sample_rate)
                    if waveform.numel() < crop_samples:
                        continue
                    for _ in range(self.multi_scale_crops_per_duration):
                        start = random.randint(0, waveform.numel() - crop_samples)
                        end = start + crop_samples
                        all_segments.append(waveform[start:end])

        if not all_segments:
            return {"ok": False, "error": "无有效音频片段"}

        embeddings = [_extract_embedding(self._model, seg) for seg in all_segments]

        if self.enable_snr_weighting:
            snr_values = [_estimate_snr(seg, sample_rate=self.sample_rate) for seg in all_segments]
            weights = torch.tensor([max(s, 0.0) for s in snr_values], dtype=torch.float32)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                mean_emb = F.normalize(
                    (torch.stack(embeddings, dim=0) * weights.unsqueeze(1)).sum(dim=0), dim=0
                )
            else:
                mean_emb = F.normalize(torch.stack(embeddings, dim=0).mean(dim=0), dim=0)
        else:
            mean_emb = F.normalize(torch.stack(embeddings, dim=0).mean(dim=0), dim=0)

        out = Path(pk_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(mean_emb.cpu().numpy(), f)

        return {"ok": True, "num_segments": len(all_segments), "pk_path": str(out.resolve())}

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = _load_model(self.model_path, self.device)

    def _augmentor(self) -> _NoiseAugmentor:
        if self._aug is None:
            self._aug = _NoiseAugmentor(
                sample_rate=self.sample_rate,
                augment_ratio=self.augment_ratio,
                noise_dir=self.noise_dir,
            )
        return self._aug


def _crop_verify(
    waveform: torch.Tensor, mode: str, window_secs: float, sample_rate: int
) -> torch.Tensor:
    """裁剪音频用于验证。"""
    mode = mode.lower()
    window = int(window_secs * sample_rate)
    if mode == "full_utterance":
        return waveform
    if mode == "head_window":
        return waveform[:window]
    return waveform[-window:] if waveform.numel() > window else waveform


def _sliding_window_scores(
    waveform: torch.Tensor,
    reference: torch.Tensor,
    model: torch.nn.Module,
    window_secs: float,
    hop_secs: float,
    sample_rate: int,
    score_mode: str = "max",
) -> tuple[float, list[float]]:
    """使用滑动窗口提取多个 embedding 并计算分数。

    Args:
        waveform: 音频波形
        reference: 参考声纹 embedding（已归一化）
        model: 声纹模型
        window_secs: 窗口长度（秒）
        hop_secs: 步长（秒）
        sample_rate: 采样率
        score_mode: 分数聚合模式，"max" 或 "top_k_mean"

    Returns:
        (最终分数, 所有窗口分数列表)
    """
    window_samples = int(window_secs * sample_rate)
    hop_samples = int(hop_secs * sample_rate)
    total_samples = waveform.numel()

    scores = []
    positions = []

    for start in range(0, total_samples - window_samples + 1, hop_samples):
        end = start + window_samples
        segment = waveform[start:end]
        emb = _extract_embedding(model, segment)
        emb = F.normalize(emb, dim=0)
        score = float(torch.dot(emb, reference).clamp(-1.0, 1.0).item())
        scores.append(score)
        positions.append(start / sample_rate)

    if not scores:
        # 音频太短，使用完整音频
        emb = _extract_embedding(model, waveform)
        emb = F.normalize(emb, dim=0)
        score = float(torch.dot(emb, reference).clamp(-1.0, 1.0).item())
        return score, [score]

    if score_mode == "max":
        final_score = max(scores)
    elif score_mode == "top_k_mean":
        k = max(1, min(3, len(scores) // 2))  # top 3 或一半
        top_scores = sorted(scores, reverse=True)[:k]
        final_score = sum(top_scores) / len(top_scores)
    else:
        final_score = max(scores)

    logger.debug(
        "Sliding window: %d windows, scores=[%.3f, %.3f, ..., %.3f], final=%.3f (mode=%s)",
        len(scores),
        scores[0] if scores else 0,
        scores[len(scores) // 2] if len(scores) > 1 else 0,
        scores[-1] if scores else 0,
        final_score,
        score_mode,
    )

    return final_score, scores


# --------------------------------------------------------------------------- #
#  Debug 辅助 — 环境变量 ENV_NAME == DEBUG 时保存测试音频到临时文件夹
# --------------------------------------------------------------------------- #


def _debug_save_test_audio(
    waveform: torch.Tensor, sample_rate: int, score: float
) -> Path | None:
    """将每次识别的音频保存到临时文件夹 wespeaker_debug/。

    文件名: {当前日期时间}-{置信度}.wav

    Returns:
        保存路径，失败返回 None。
    """
    try:
        import tempfile
        from datetime import datetime

        import torchaudio

        dst_dir = Path(tempfile.gettempdir()) / "wespeaker_debug"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}-{score:.4f}.wav"
        save_wav = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform
        torchaudio.save(str(dst), save_wav.cpu(), sample_rate)
        logger.info("[DEBUG] 测试音频已保存 → %s", dst)
        return dst
    except Exception as e:
        logger.error("[DEBUG] 测试音频保存失败: %s", e, exc_info=True)
        return None


# --------------------------------------------------------------------------- #
#  CLI 入口
# --------------------------------------------------------------------------- #


def main() -> None:
    """CLI 入口点 - 用于 pip 安装后的 console script."""
    import argparse

    parser = argparse.ArgumentParser(description="WeSpeaker 声纹注册与识别")
    sub = parser.add_subparsers(dest="cmd")

    p_reg = sub.add_parser("enroll", help="注册声纹")
    p_reg.add_argument("audio")
    p_reg.add_argument("output", nargs="?", default="voice.pkl")
    p_reg.add_argument("--model-path", default="./models/wespeaker")
    p_reg.add_argument("--device", default="cpu")
    p_reg.add_argument("--no-aug", action="store_true")
    p_reg.add_argument(
        "--no-multi-scale", action="store_true", help="禁用多尺度短语音注册（0.3s, 0.5s, 0.8s）"
    )

    p_rec = sub.add_parser("recognize", help="识别声纹")
    p_rec.add_argument("audio")
    p_rec.add_argument("voiceprint", nargs="?", default=None, help="声纹 .pkl 路径（省略则使用内置 John 声纹）")
    p_rec.add_argument("--model-path", default="./models/wespeaker")
    p_rec.add_argument("--device", default="cpu")
    p_rec.add_argument(
        "--package-pk-index",
        type=int,
        default=None,
        help="内置声纹索引: 0=John 1=Frank 2=Michael 3=Qingqing 4=Xixi 5=Zhong（优先级高于 voiceprint）",
    )
    p_rec.add_argument("--threshold", type=float, default=0.75)
    p_rec.add_argument(
        "--dynamic-threshold", action="store_true", help="启用基于 VAD 时长的动态阈值"
    )
    p_rec.add_argument(
        "--threshold-mode",
        choices=["segmented", "smooth"],
        default="segmented",
        help="动态阈值模式: segmented (分段) 或 smooth (平滑公式)",
    )
    p_rec.add_argument("--no-sliding-window", action="store_true", help="禁用滑动窗口多窗口识别")
    p_rec.add_argument(
        "--sliding-window-secs", type=float, default=0.6, help="滑动窗口长度（秒），默认 0.6"
    )
    p_rec.add_argument(
        "--sliding-hop-secs", type=float, default=0.2, help="滑动窗口步长（秒），默认 0.2"
    )
    p_rec.add_argument(
        "--sliding-score-mode",
        choices=["max", "top_k_mean"],
        default="max",
        help="滑动窗口分数聚合模式: max（最高分）或 top_k_mean（top k 均值）",
    )

    args = parser.parse_args()

    if args.cmd == "enroll":
        client = WespeakerClient(
            model_path=args.model_path,
            device=args.device,
            enable_augmentation=not args.no_aug,
            enable_multi_scale_enrollment=not args.no_multi_scale,
        )
        r = client.mp3_to_pk(args.audio, args.output)
        logger.info("注册结果: %s", r)
    elif args.cmd == "recognize":
        client = WespeakerClient(
            model_path=args.model_path,
            device=args.device,
            sim_threshold=args.threshold,
            enable_dynamic_threshold=args.dynamic_threshold,
            dynamic_threshold_mode=args.threshold_mode,
            enable_sliding_window=not args.no_sliding_window,
            sliding_window_secs=args.sliding_window_secs,
            sliding_hop_secs=args.sliding_hop_secs,
            sliding_score_mode=args.sliding_score_mode,
            package_pk_index=args.package_pk_index,
        )
        r = client.recognize(args.audio, args.voiceprint)
        logger.info("识别结果: %s", {k: v for k, v in r.items() if k != "debug_audio"})
        if r.get("debug_audio"):
            logger.info("[DEBUG] 音频已保存到: %s", r["debug_audio"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
