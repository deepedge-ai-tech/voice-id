"""WeSpeaker 最佳声纹识别配置 — 基于实验验证的最优参数。

核心差异 vs WespeakerClient.mp3_to_pk:
  - 注册时使用 multi-SNR 真实噪声注入（而非 audiomentations 随机增强）
  - 验证时禁用 VAD（实验表明完整音频得分更高）
  - verify_buffer_keep_secs = 60.0（不截断，使用完整音频）

用法:
    from wespeaker.best import WespeakerBest, BestConfig

    # 方式 1: 使用默认最佳配置
    recognizer = WespeakerBest(model_path="./models/wespeaker")

    # 方式 2: 自定义部分参数
    config = BestConfig(sim_threshold=0.50, verify_buffer_keep_secs=30.0)
    recognizer = WespeakerBest(model_path="./models/wespeaker", config=config)

    # 注册（需要噪声音频）
    noise_profile = WespeakerBest.extract_noise_profile("noise.wav")
    recognizer.enroll("clean_segments_dir", noise_profile, "voice.pkl")

    # 识别
    result = recognizer.recognize("test_audio.wav", "voice.pkl")
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from .wespeaker import (
    WespeakerClient,
    _apply_silero_vad,
    _extract_embedding,
    _load_audio,
    _vad_segments,  # 用于噪声提取，不用于注册/识别预处理
)

# --------------------------------------------------------------------------- #
#  最佳配置
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BestConfig:
    """实验验证的最优声纹识别参数。

    所有值来自系统测试。修改前请了解其对识别率的影响。

    Attributes:
        sim_threshold: 余弦相似度阈值。0.55 时 clean 通过率 98.3%。
            降低会增加误接受，提高会增加误拒绝。
        verify_crop_mode: 验证音频裁剪策略。"full_utterance" 使用完整音频，
            实验表明得分始终高于滑动窗口方式。
        verify_buffer_keep_secs: 验证时最大保留音频长度（秒）。60.0 意味着
            不截断，与 WespeakerClient 默认的 8.0 不同。
        verify_window_secs: 滑动窗口测试时的窗口长度（秒），仅用于诊断。
        enrollment_segment_secs: 注册时切分段长度（秒）。
        enable_vad: 验证时是否启用 VAD。False — 实验表明对完整音频不做 VAD
            去静音得分更高，VAD 可能切掉低能量但有判别性的语音段。
        vad_rms_threshold: VAD RMS 能量阈值，仅在 enable_vad=True 时生效。
        noise_injection_snrs: 注册时噪声注入的 SNR 级别列表（dB）。
            [20, 15, 10, 5, 0] 覆盖从轻微到严重噪声环境，
            这是本方案与 WespeakerClient.mp3_to_pk 的核心区别。
    """

    sim_threshold: float = 0.55
    verify_crop_mode: str = "full_utterance"
    verify_buffer_keep_secs: float = 60.0
    verify_window_secs: float = 1.0
    enrollment_segment_secs: float = 1.0
    enable_vad: bool = False
    vad_rms_threshold: float = 0.002  # VAD 能量阈值（已降低以减少误剪）
    noise_injection_snrs: tuple[float, ...] = (20, 15, 10, 5, 0)


# --------------------------------------------------------------------------- #
#  噪声提取与注入（模块级函数，可独立测试）
# --------------------------------------------------------------------------- #


def _extract_noise_profile(
    noisy_audio_path: str,
    sample_rate: int = 16000,
    rms_threshold: float = 0.005,
) -> np.ndarray:
    """从噪声音频中提取环境噪声 profile（非语音段）。"""
    waveform = _load_audio(noisy_audio_path, sample_rate)
    speech_segs = _vad_segments(waveform, rms_threshold=rms_threshold, sample_rate=sample_rate)
    if not speech_segs:
        return waveform.cpu().numpy()

    frame_len = int(0.02 * sample_rate)
    hop_len = int(0.01 * sample_rate)
    rms_values = []
    starts = []
    for start in range(0, len(waveform) - frame_len + 1, hop_len):
        seg = waveform[start : start + frame_len]
        rms = float(torch.sqrt((seg**2).mean()))
        rms_values.append(rms)
        starts.append(start)

    if not rms_values:
        return waveform.cpu().numpy()

    sorted_rms = sorted(rms_values)
    noise_floor = sorted_rms[max(0, len(sorted_rms) // 4)]
    adaptive_threshold = max(noise_floor * 2.0, rms_threshold)

    noise_regions = []
    in_noise = False
    noise_start = 0
    for i, rms in enumerate(rms_values):
        if rms < adaptive_threshold and not in_noise:
            in_noise = True
            noise_start = starts[i]
        elif (rms >= adaptive_threshold or i == len(rms_values) - 1) and in_noise:
            in_noise = False
            noise_end = starts[i] if i < len(starts) else starts[-1] + frame_len
            if noise_end - noise_start > sample_rate // 10:
                noise_regions.append((noise_start, noise_end))

    if noise_regions:
        parts = [waveform[s:e] for s, e in noise_regions]
        result = torch.cat(parts)
        return result.cpu().numpy()

    return waveform.cpu().numpy()


def _mix_noise_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """将噪声混合到 clean 音频，达到目标 SNR (dB)。"""
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-12:
        return clean.copy()

    target_noise_power = clean_power / (10 ** (target_snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    if len(noise) < len(clean):
        noise = np.tile(noise, (len(clean) // len(noise)) + 2)
    noise = noise[: len(clean)]
    return (clean + noise_scale * noise).astype(np.float32)


# --------------------------------------------------------------------------- #
#  主类
# --------------------------------------------------------------------------- #


class WespeakerBest:
    """基于最优实验配置的声纹注册与识别。

    内部组合 WespeakerClient 用于模型加载和底层音频处理，
    但使用独立的注册流程（multi-SNR 真实噪声注入）。

    三个核心公开方法:
      - enroll():     注册声纹并保存 .pkl
      - load():       从 .pkl 加载声纹
      - recognize():  识别/验证音频

    两个噪声工具方法（注册前调用）:
      - extract_noise_profile():  从噪声音频提取噪声 profile
      - mix_noise_at_snr():       在指定 SNR 下混合噪声
    """

    def __init__(
        self,
        model_path: str = "./models/wespeaker",
        device: str = "cpu",
        sample_rate: int = 16000,
        config: BestConfig | None = None,
    ) -> None:
        """初始化。

        Args:
            model_path: pyannote.audio 模型路径。
            device: 推理设备 ("cpu" / "cuda")。
            sample_rate: 音频采样率。
            config: 最佳配置对象。None 时使用默认 BestConfig()。
        """
        self._client = WespeakerClient(
            model_path=model_path,
            device=device,
            sample_rate=sample_rate,
            enable_augmentation=False,
        )
        self.config = config if config is not None else BestConfig()

    # ------------------------------------------------------------------ #
    #  核心方法 1: 注册
    # ------------------------------------------------------------------ #

    def enroll(
        self,
        clean_dir: str | Path,
        noise_profile: np.ndarray,
        pk_path: str | Path,
        snr_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        """用 multi-SNR 真实噪声注入注册声纹，保存为 .pkl 文件。

        每个 clean 注册片段在多个 SNR 级别下与噪声混合，
        所有混合片段的 embedding 取均值并归一化，作为参考声纹。

        Args:
            clean_dir: 包含 clean 注册片段的目录（支持 .wav 及任意格式）。
            noise_profile: 通过 extract_noise_profile() 提取的噪声数组。
            pk_path: 输出 .pkl 文件路径。
            snr_levels: SNR 级别列表（dB）。None 时使用 config.noise_injection_snrs。

        Returns:
            包含 ok, num_segments, num_snr_levels, total_enrollments,
            embedding_dim, pk_path, embedding 的字典。

        Raises:
            FileNotFoundError: clean_dir 不存在或无有效音频文件。
        """
        self._client._ensure_model()

        clean_dir = str(Path(clean_dir))
        clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        if not clean_paths:
            clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*")))
        if not clean_paths:
            raise FileNotFoundError(f"注册目录无有效音频文件: {clean_dir}")

        snrs = snr_levels if snr_levels is not None else list(self.config.noise_injection_snrs)
        logger.info("Enrolling %s with %d clean segments", clean_dir, len(clean_paths))
        logger.debug("SNR levels: %s", snrs)
        all_embeddings: list[torch.Tensor] = []

        for path in clean_paths:
            seg = _load_audio(path, self._client.sample_rate)
            # 应用 Silero VAD 去除静音
            seg_vad = _apply_silero_vad(seg, self._client.sample_rate)
            seg = seg_vad.cpu().numpy()
            for snr in snrs:
                mixed = _mix_noise_at_snr(seg, noise_profile, snr)
                emb = _extract_embedding(self._client._model, torch.from_numpy(mixed))
                all_embeddings.append(F.normalize(emb, dim=0))

        ref = F.normalize(torch.stack(all_embeddings).mean(dim=0), dim=0)

        pk_path = Path(pk_path)
        pk_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pk_path, "wb") as f:
            pickle.dump(ref.cpu().numpy(), f)

        logger.info(
            "Enrolled %s: %d embeddings, dim=%d",
            clean_dir,
            len(all_embeddings),
            ref.numel(),
        )

        return {
            "ok": True,
            "num_segments": len(clean_paths),
            "num_snr_levels": len(snrs),
            "total_enrollments": len(all_embeddings),
            "embedding_dim": ref.numel(),
            "pk_path": str(pk_path.resolve()),
            "embedding": ref,
            "fragment_embeddings": all_embeddings,  # 添加片段 embeddings 用于诊断
        }

    # ------------------------------------------------------------------ #
    #  核心方法 2: 加载
    # ------------------------------------------------------------------ #

    def load(self, pk_path: str | Path) -> np.ndarray:
        """从 .pkl 文件加载已保存的声纹 embedding。

        Args:
            pk_path: .pkl 文件路径。

        Returns:
            归一化的 256 维 numpy 数组。

        Raises:
            FileNotFoundError: pk_path 不存在。
            ValueError: .pkl 内容不是有效的 256 维向量。
        """
        pk_path = Path(pk_path)
        if not pk_path.is_file():
            raise FileNotFoundError(f"声纹文件不存在: {pk_path}")

        with open(pk_path, "rb") as f:
            data = pickle.load(f)

        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != 256:
            raise ValueError(f"无效的声纹维度: {arr.shape}，期望 (256,)")

        return F.normalize(torch.from_numpy(arr), dim=0).cpu().numpy()

    # ------------------------------------------------------------------ #
    #  核心方法 3: 识别
    # ------------------------------------------------------------------ #

    def recognize(
        self,
        audio_path: str | Path,
        voiceprint: np.ndarray | str | Path,
    ) -> dict[str, Any]:
        """将音频与声纹比对，返回识别结果。

        使用 BestConfig 的参数：
          - verify_buffer_keep_secs: 限制最大音频长度
          - enable_vad: 决定是否做 VAD 去静音
          - sim_threshold: 判定阈值

        Args:
            audio_path: 待测试的音频文件路径。
            voiceprint: 声纹，可以是:
                - numpy 数组（直接传入）
                - str / Path（从 .pkl 文件加载，内部调用 self.load()）

        Returns:
            包含 is_recognized, confidence, threshold 的字典。
            如果出错，额外包含 error 字段。
        """
        audio_path = str(Path(audio_path))
        logger.debug("Recognizing %s", audio_path)
        if not Path(audio_path).is_file():
            return {
                "is_recognized": False,
                "confidence": 0.0,
                "error": f"文件不存在: {audio_path}",
            }

        # 加载声纹
        if isinstance(voiceprint, np.ndarray):
            ref = F.normalize(torch.from_numpy(voiceprint.astype(np.float32)), dim=0)
        else:
            ref_data = self.load(voiceprint)
            ref = F.normalize(torch.from_numpy(ref_data.astype(np.float32)), dim=0)

        # 加载音频
        self._client._ensure_model()
        waveform = _load_audio(audio_path, self._client.sample_rate)

        # 限制最大长度
        max_samples = int(self.config.verify_buffer_keep_secs * self._client.sample_rate)
        if waveform.numel() > max_samples:
            if self.config.verify_crop_mode == "head_window":
                waveform = waveform[:max_samples]
            else:
                waveform = waveform[-max_samples:]

        # VAD 去静音
        if self.config.enable_vad:
            speech_segs = _vad_segments(
                waveform,
                rms_threshold=self.config.vad_rms_threshold,
                sample_rate=self._client.sample_rate,
            )
            if speech_segs:
                pcm = torch.cat(speech_segs)
            else:
                pcm = waveform
        else:
            pcm = waveform

        if pcm.numel() == 0:
            return {"is_recognized": False, "confidence": 0.0, "error": "音频太短"}

        emb = F.normalize(_extract_embedding(self._client._model, pcm), dim=0)
        score = float(torch.dot(emb, ref).clamp(-1.0, 1.0).item())

        logger.debug("Recognition score: %.4f", score)

        return {
            "is_recognized": score >= self.config.sim_threshold,
            "confidence": round(score, 4),
            "threshold": self.config.sim_threshold,
        }

    # ------------------------------------------------------------------ #
    #  噪声工具方法（静态，供外部调用）
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_noise_profile(
        noisy_audio_path: str | Path,
        sample_rate: int = 16000,
        rms_threshold: float = 0.005,
    ) -> np.ndarray:
        """从噪声音频中提取环境噪声 profile（非语音段）。

        使用 VAD 找出静音/噪声区域，拼接成噪声 profile。
        该 profile 用于 enroll() 中的噪声注入。

        Args:
            noisy_audio_path: 包含目标环境噪声的音频文件。
            sample_rate: 采样率。
            rms_threshold: VAD 能量阈值。

        Returns:
            噪声波形 numpy 数组。
        """
        return _extract_noise_profile(str(noisy_audio_path), sample_rate, rms_threshold)

    @staticmethod
    def mix_noise_at_snr(
        clean: np.ndarray,
        noise: np.ndarray,
        target_snr_db: float,
    ) -> np.ndarray:
        """将噪声混合到 clean 音频，达到目标 SNR。

        这是纯数学函数，不依赖模型。公开暴露供高级用法
        （如自定义 SNR 组合、批量预处理等）。

        Args:
            clean: 干净语音波形。
            noise: 噪声波形。
            target_snr_db: 目标信噪比（dB）。

        Returns:
            混合后的波形。
        """
        return _mix_noise_at_snr(clean, noise, target_snr_db)
