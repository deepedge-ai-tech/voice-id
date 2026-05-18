"""WeSpeaker Deep — 基于实验验证的最优策略声纹识别。

继承 WespeakerBest，覆写 enroll/recognize，加入:
  - 纯净注册（clean-only, skip-VAD, 多模板存储）
  - 多模板匹配（测试时对所有 template 取 max）
  - 短音频滑动窗口（0.4s/0.15s，仅对 < 1.5s 音频）
  - sqrt 分数补偿（短音频自动提分）

用法:
    from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep, DeepConfig

    config = DeepConfig(sim_threshold=0.50)
    recognizer = WespeakerDeep(model_path="./models/wespeaker", config=config)

    # 注册（纯干净注册，不需要噪声音频）
    recognizer.enroll("clean_segments_dir", pk_path="voice.pkl")

    # 识别
    result = recognizer.recognize("test_audio.wav", "voice.pkl")
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from .best import BestConfig, WespeakerBest
from .wespeaker import (
    _apply_silero_vad,
    _extract_embedding,
    _load_audio,
    get_score_compensation_factor,
)

# --------------------------------------------------------------------------- #
#  Debug 辅助 — 环境变量 ENV_NAME == DEBUG 时保存测试音频到临时文件夹
# --------------------------------------------------------------------------- #


def _debug_save_test_audio(waveform: torch.Tensor, sample_rate: int, score: float) -> None:
    """当 ENV_NAME == DEBUG 时，将测试音频保存到临时文件夹。

    文件名: {当前日期时间}-{置信度}.wav

    Args:
        waveform: 音频波形（1D 或 2D tensor，值域 [-1, 1]）。
        sample_rate: 采样率。
        score: 识别置信度（用于文件名）。
    """
    if os.environ.get("ENV_NAME") != "DEBUG":
        return
    try:
        import tempfile
        from datetime import datetime

        import torchaudio

        dst_dir = Path(tempfile.gettempdir()) / "wespeaker_debug"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}-{score:.4f}.wav"
        # torchaudio 需要 (channels, samples) 格式
        save_wav = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform
        torchaudio.save(str(dst), save_wav.cpu(), sample_rate)
        logger.debug("测试音频已保存: %s", dst)
    except Exception:
        logger.warning("保存调试音频失败", exc_info=True)


# --------------------------------------------------------------------------- #
#  DeepConfig — winning 策略参数
# --------------------------------------------------------------------------- #


@dataclass
class DeepConfig:
    """WespeakerDeep 最优参数配置。

    所有默认值来自 18 轮自动实验验证的最优策略。
    与 BestConfig 不同，此 dataclass 是可变（mutable）的。
    """

    # 识别参数
    sim_threshold: float = 0.50
    verify_crop_mode: str = "head_window"
    verify_buffer_keep_secs: float = 60.0
    verify_window_secs: float = 0.4

    # 注册参数
    enrollment_segment_secs: float = 0.6

    # VAD 参数
    enable_vad: bool = False
    vad_rms_threshold: float = 0.002

    # 分数补偿
    enable_score_compensation: bool = True
    score_compensation_target_duration: float = 2.0
    score_compensation_mode: str = "sqrt"  # "sqrt" / "linear" / "none"

    # 噪声注入（默认空 = clean-only）
    noise_injection_snrs: tuple[float, ...] = ()

    # 滑动窗口
    sliding_window_secs: float = 0.4
    sliding_hop_secs: float = 0.15

    # ---- 新增字段（BestConfig 没有的）----

    # 注册侧
    enroll_skip_vad: bool = True  # 注册时跳过 VAD
    enroll_clean_only: bool = True  # 纯干净注册，不注入噪声

    # 识别侧
    enable_multi_template: bool = True  # 多模板匹配（对每文件 embedding 取 max）
    enable_sliding_window_test: bool = False  # 短音频滑动窗口（实验表明会推高 FAR，默认关闭）
    short_audio_max_duration: float = 1.5  # 短音频判定阈值（秒）

    def to_dict(self) -> dict:
        return {
            "sim_threshold": self.sim_threshold,
            "verify_crop_mode": self.verify_crop_mode,
            "verify_buffer_keep_secs": self.verify_buffer_keep_secs,
            "verify_window_secs": self.verify_window_secs,
            "enrollment_segment_secs": self.enrollment_segment_secs,
            "enable_vad": self.enable_vad,
            "vad_rms_threshold": self.vad_rms_threshold,
            "enable_score_compensation": self.enable_score_compensation,
            "score_compensation_target_duration": self.score_compensation_target_duration,
            "score_compensation_mode": self.score_compensation_mode,
            "noise_injection_snrs": list(self.noise_injection_snrs),
            "sliding_window_secs": self.sliding_window_secs,
            "sliding_hop_secs": self.sliding_hop_secs,
            "enroll_skip_vad": self.enroll_skip_vad,
            "enroll_clean_only": self.enroll_clean_only,
            "enable_multi_template": self.enable_multi_template,
            "enable_sliding_window_test": self.enable_sliding_window_test,
            "short_audio_max_duration": self.short_audio_max_duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeepConfig":
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in field_names}
        if "noise_injection_snrs" in kwargs and isinstance(kwargs["noise_injection_snrs"], list):
            kwargs["noise_injection_snrs"] = tuple(kwargs["noise_injection_snrs"])
        return cls(**kwargs)

    @classmethod
    def from_best_config(cls, bc: BestConfig) -> "DeepConfig":
        """从 BestConfig 转换为 DeepConfig，补齐新增字段默认值。"""
        return cls(
            sim_threshold=bc.sim_threshold,
            verify_crop_mode=bc.verify_crop_mode,
            verify_buffer_keep_secs=bc.verify_buffer_keep_secs,
            verify_window_secs=bc.verify_window_secs,
            enrollment_segment_secs=bc.enrollment_segment_secs,
            enable_vad=bc.enable_vad,
            vad_rms_threshold=bc.vad_rms_threshold,
            noise_injection_snrs=bc.noise_injection_snrs,
            sliding_window_secs=bc.sliding_window_secs,
            sliding_hop_secs=bc.sliding_hop_secs,
        )


# --------------------------------------------------------------------------- #
#  WespeakerDeep — 继承 WespeakerBest，覆写核心方法
# --------------------------------------------------------------------------- #


class WespeakerDeep(WespeakerBest):
    """基于实验最优策略的声纹注册与识别。

    继承 WespeakerBest，覆写 enroll/recognize/load，
    加入多模板匹配、滑动窗口、sqrt 分数补偿等 winning 策略。

    用法:
        recognizer = WespeakerDeep(model_path="./models/wespeaker")
        recognizer.enroll("segments/", pk_path="voice.pkl")
        result = recognizer.recognize("test.wav", "voice.pkl")
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        sample_rate: int = 16000,
        config: DeepConfig | None = None,
    ) -> None:
        self._deep_config = config if config is not None else DeepConfig()

        if model_path is None:
            from .wespeaker import _get_default_model_path

            model_path = _get_default_model_path()

        # 构造兼容的 BestConfig 传给父类
        best_config = BestConfig(
            sim_threshold=self._deep_config.sim_threshold,
            enable_score_compensation=self._deep_config.enable_score_compensation,
            score_compensation_target_duration=self._deep_config.score_compensation_target_duration,
            verify_crop_mode=self._deep_config.verify_crop_mode,
            verify_buffer_keep_secs=self._deep_config.verify_buffer_keep_secs,
            verify_window_secs=self._deep_config.verify_window_secs,
            enrollment_segment_secs=self._deep_config.enrollment_segment_secs,
            enable_vad=self._deep_config.enable_vad,
            vad_rms_threshold=self._deep_config.vad_rms_threshold,
            noise_injection_snrs=self._deep_config.noise_injection_snrs,
            sliding_window_secs=self._deep_config.sliding_window_secs,
            sliding_hop_secs=self._deep_config.sliding_hop_secs,
        )

        super().__init__(
            model_path=model_path,
            device=device,
            sample_rate=sample_rate,
            config=best_config,
        )

    @property
    def deep_config(self) -> DeepConfig:
        """返回 DeepConfig（可变，可在运行时调整）。"""
        return self._deep_config

    # ------------------------------------------------------------------ #
    #  .pkl 加载
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_raw(pk_path: str | Path) -> Any:
        """加载 .pkl 文件的原始内容。"""
        pk_path = Path(pk_path)
        if not pk_path.is_file():
            raise FileNotFoundError(f"声纹文件不存在: {pk_path}")
        with open(pk_path, "rb") as f:
            return pickle.load(f)

    def load(self, pk_path: str | Path) -> np.ndarray:
        """从 .pkl 加载声纹 reference embedding。

        兼容新旧格式：
          - 旧格式（单数组）→ 直接返回
          - 新格式（dict）→ 返回 data["reference"]

        Returns:
            归一化的 256 维 numpy 数组。
        """
        data = self._load_raw(pk_path)
        if isinstance(data, dict):
            arr = np.asarray(data["reference"], dtype=np.float32)
        else:
            arr = np.asarray(data, dtype=np.float32)
        return F.normalize(torch.from_numpy(arr), dim=0).cpu().numpy()

    def load_full(self, pk_path: str | Path) -> dict:
        """加载 .pkl 完整内容，兼容新旧格式。

        旧格式（单数组）自动包装为:
            {"version": 0, "templates": [arr], "reference": arr}
        新格式直接返回 dict。

        Returns:
            {"version": int, "templates": list[np.ndarray], "reference": np.ndarray}
        """
        data = self._load_raw(pk_path)
        if isinstance(data, dict):
            return data
        # 旧格式：单数组
        arr = np.asarray(data, dtype=np.float32)
        return {"version": 0, "templates": [arr], "reference": arr}

    # ------------------------------------------------------------------ #
    #  分数补偿
    # ------------------------------------------------------------------ #

    def _apply_score_compensation(self, raw_score: float, duration: float) -> tuple[float, float]:
        """对短音频应用分数补偿。

        Args:
            raw_score: 原始相似度分数。
            duration: 音频时长（秒）。

        Returns:
            (compensated_score, factor)
        """
        if not self._deep_config.enable_score_compensation:
            return raw_score, 1.0

        mode = self._deep_config.score_compensation_mode
        target = self._deep_config.score_compensation_target_duration

        if mode == "sqrt":
            effective_dur = max(duration, 0.3)
            factor = min((target / effective_dur) ** 0.5, 2.0)
        elif mode == "linear":
            factor = get_score_compensation_factor(duration, target)
        else:
            factor = 1.0

        compensated = min(raw_score * factor, 1.0)
        logger.debug(
            "Score compensation: %.4f → %.4f (mode=%s, factor=%.3f, dur=%.2fs)",
            raw_score,
            compensated,
            mode,
            factor,
            duration,
        )
        return compensated, factor

    # ------------------------------------------------------------------ #
    #  覆写: 注册
    # ------------------------------------------------------------------ #

    def enroll(
        self,
        clean_dir: str | Path,
        noise_profile: np.ndarray | None = None,
        pk_path: str | Path = "voice.pkl",
        snr_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        """纯净注册声纹 — 不注入噪声，保留每文件独立 embedding。

        与父类 enroll() 的核心区别:
          - noise_profile 默认 None（不需要噪声音频）
          - 跳过 VAD（enroll_skip_vad=True）
          - 不注入噪声（enroll_clean_only=True）
          - .pkl 保存为 dict 格式，含 templates 列表供多模板匹配

        Args:
            clean_dir: 包含 clean 注册片段的目录。
            noise_profile: 噪声数组。clean_only 模式下传入也忽略，打 warning。
            pk_path: 输出 .pkl 文件路径。
            snr_levels: SNR 级别列表。clean_only 模式下忽略。

        Returns:
            {"ok": True, "num_segments": int, "num_templates": int,
             "embedding_dim": int, "pk_path": str}
        """
        self._client._ensure_model()

        clean_dir = str(Path(clean_dir))
        clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        if not clean_paths:
            clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*")))
        if not clean_paths:
            raise FileNotFoundError(f"注册目录无有效音频文件: {clean_dir}")

        if noise_profile is not None and self._deep_config.enroll_clean_only:
            logger.warning("enroll_clean_only=True，noise_profile 参数被忽略")

        logger.info("Deep enroll: %d clean segments (clean-only=%s, skip-vad=%s)",
                     len(clean_paths), self._deep_config.enroll_clean_only,
                     self._deep_config.enroll_skip_vad)

        all_embeddings: list[np.ndarray] = []

        for path in clean_paths:
            seg = _load_audio(path, self._client.sample_rate)

            if not self._deep_config.enroll_skip_vad:
                seg = _apply_silero_vad(seg, self._client.sample_rate)

            emb = _extract_embedding(self._client._model, seg)
            emb = F.normalize(emb, dim=0)
            all_embeddings.append(emb.cpu().numpy())

        if not all_embeddings:
            return {"ok": False, "error": "无有效注册片段"}

        # reference = 所有 template 的均值（归一化）
        stacked = torch.from_numpy(np.stack(all_embeddings))
        reference = F.normalize(stacked.mean(dim=0), dim=0).cpu().numpy()

        # 保存为新格式 dict
        pkl_data = {
            "version": 1,
            "templates": all_embeddings,
            "reference": reference,
        }

        out = Path(pk_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(pkl_data, f)

        logger.info(
            "Deep enrolled: %d segments → %d templates, dim=%d, saved to %s",
            len(clean_paths),
            len(all_embeddings),
            reference.shape[0],
            out,
        )

        return {
            "ok": True,
            "num_segments": len(clean_paths),
            "num_templates": len(all_embeddings),
            "embedding_dim": reference.shape[0],
            "pk_path": str(out.resolve()),
            "reference": reference,
            "embedding": reference,
        }

    # ------------------------------------------------------------------ #
    #  覆写: 识别
    # ------------------------------------------------------------------ #

    def recognize(
        self,
        audio_path: str | Path,
        voiceprint: np.ndarray | str | Path,
    ) -> dict[str, Any]:
        """将音频与声纹比对 — 多模板 + 滑动窗口 + sqrt 分数补偿。

        核心流程:
          1. load_full() 获取所有 template embeddings
          2. 短音频（< short_audio_max_duration）：滑动窗口提取 N 个 window embedding
          3. 多模板匹配：对所有 (window × template) 组合取 max 分数
          4. sqrt 分数补偿提升短音频分数
          5. 长音频只做多模板 max，不滑动窗口

        Args:
            audio_path: 待测试音频文件路径。
            voiceprint: 声纹（numpy 数组 / .pkl 路径）。

        Returns:
            {"is_recognized": bool, "confidence": float, "threshold": float,
             "num_templates_used": int, "sliding_windows_used": int,
             "score_compensation_factor": float}
        """
        audio_path = str(Path(audio_path))
        if not Path(audio_path).is_file():
            return {
                "is_recognized": False,
                "confidence": 0.0,
                "error": f"文件不存在: {audio_path}",
            }

        self._client._ensure_model()
        cfg = self._deep_config

        # ---- 加载声纹 ----
        if isinstance(voiceprint, np.ndarray):
            # 直接传入数组 → 单模板模式
            templates = [F.normalize(torch.from_numpy(voiceprint.astype(np.float32)), dim=0)]
            num_templates = 1
        else:
            full = self.load_full(voiceprint)
            templates = [
                F.normalize(torch.from_numpy(np.asarray(t, dtype=np.float32)), dim=0)
                for t in full["templates"]
            ]
            num_templates = len(templates)

        if not cfg.enable_multi_template and num_templates > 1:
            # 关闭多模板：只用 reference
            ref_arr = self.load(voiceprint) if not isinstance(voiceprint, np.ndarray) else voiceprint
            templates = [F.normalize(torch.from_numpy(ref_arr.astype(np.float32)), dim=0)]
            num_templates = 1

        # ---- 加载测试音频 ----
        waveform = _load_audio(audio_path, self._client.sample_rate)
        original_duration = waveform.numel() / self._client.sample_rate

        # 裁剪
        max_samples = int(cfg.verify_buffer_keep_secs * self._client.sample_rate)
        if waveform.numel() > max_samples:
            if cfg.verify_crop_mode == "head_window":
                waveform = waveform[:max_samples]
            else:
                waveform = waveform[-max_samples:]

        # VAD
        vad_duration = original_duration
        if cfg.enable_vad:
            from .wespeaker import _vad_segments

            speech_segs = _vad_segments(
                waveform,
                rms_threshold=cfg.vad_rms_threshold,
                sample_rate=self._client.sample_rate,
            )
            if speech_segs:
                pcm = torch.cat(speech_segs)
                vad_duration = pcm.numel() / self._client.sample_rate
            else:
                pcm = waveform
        else:
            pcm = waveform

        if pcm.numel() == 0:
            return {"is_recognized": False, "confidence": 0.0, "error": "音频太短"}

        # ---- 判断是否短音频 → 滑动窗口 ----
        is_short = vad_duration < cfg.short_audio_max_duration
        use_sliding = (
            cfg.enable_sliding_window_test
            and is_short
            and pcm.numel() >= int(cfg.sliding_window_secs * self._client.sample_rate)
        )

        sliding_windows_used = 0

        if use_sliding:
            # 滑动窗口：提取多个 window embedding
            window_samples = int(cfg.sliding_window_secs * self._client.sample_rate)
            hop_samples = int(cfg.sliding_hop_secs * self._client.sample_rate)

            best_score = -1.0
            for start in range(0, pcm.numel() - window_samples + 1, hop_samples):
                window = pcm[start : start + window_samples]
                w_emb = F.normalize(
                    _extract_embedding(self._client._model, window), dim=0
                )
                sliding_windows_used += 1
                # 多模板 max
                for t in templates:
                    s = float(torch.dot(w_emb, t).clamp(-1.0, 1.0))
                    if s > best_score:
                        best_score = s
            raw_score = best_score
        else:
            # 不滑动：单次 embedding 提取
            test_emb = F.normalize(
                _extract_embedding(self._client._model, pcm), dim=0
            )
            if cfg.enable_multi_template and num_templates > 1:
                template_scores = [
                    float(torch.dot(test_emb, t).clamp(-1.0, 1.0))
                    for t in templates
                ]
                raw_score = max(template_scores)
            else:
                raw_score = float(
                    torch.dot(test_emb, templates[0]).clamp(-1.0, 1.0)
                )

        # ---- 分数补偿 ----
        effective_duration = (
            cfg.sliding_window_secs if use_sliding else vad_duration
        )
        score, factor = self._apply_score_compensation(raw_score, effective_duration)

        # ---- 判定 ----
        threshold = cfg.sim_threshold
        logger.debug(
            "Deep recognize: score=%.4f (raw=%.4f), thresh=%.2f, "
            "templates=%d, sliding_windows=%d, factor=%.3f, dur=%.2fs",
            score, raw_score, threshold, num_templates,
            sliding_windows_used, factor, vad_duration,
        )

        _debug_save_test_audio(waveform, self._client.sample_rate, score)

        return {
            "is_recognized": score >= threshold,
            "confidence": round(score, 4),
            "raw_confidence": round(raw_score, 4),
            "threshold": threshold,
            "vad_duration": round(vad_duration, 2),
            "num_templates_used": num_templates,
            "sliding_windows_used": sliding_windows_used,
            "score_compensation_factor": round(factor, 4),
        }
