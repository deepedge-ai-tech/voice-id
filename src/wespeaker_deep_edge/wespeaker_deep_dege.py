"""WeSpeaker Deep — 基于官方 WeSpeaker 的声纹注册与识别。

使用 wespeaker.load_model() 加载官方预训练模型，
使用纯 embedding 提取 + cosine_similarity 做比对。

用法:
    from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep, DeepConfig

    config = DeepConfig(sim_threshold=0.70)
    recognizer = WespeakerDeep(config=config)

    # 注册
    recognizer.enroll("enroll_audio.wav", pk_path="voice.pkl")

    # 识别
    result = recognizer.recognize("test_audio.wav", "voice.pkl")
"""

from __future__ import annotations

# 使用 vendored wespeaker（位于 _wespeaker/），确保可导入
import sys
from pathlib import Path

_vendored = str(Path(__file__).parent / "_wespeaker")
if _vendored not in sys.path:
    sys.path.insert(0, _vendored)

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  DeepConfig — 精简参数配置
# --------------------------------------------------------------------------- #


@dataclass
class DeepConfig:
    """WespeakerDeep 参数配置。

    与官方 WeSpeaker demo 保持一致:
      - sim_threshold: 0.70（demo 使用的阈值）
      - package_pk_index: 内置声纹索引
    """

    sim_threshold: float = 0.70
    package_pk_index: int | None = None


# --------------------------------------------------------------------------- #
#  WespeakerDeep — 基于官方 WeSpeaker 的声纹注册与识别
# --------------------------------------------------------------------------- #


class WespeakerDeep:
    """基于官方 WeSpeaker 的声纹注册与识别。

    内部使用 wespeaker.load_model() 加载预训练模型，
    extract_embedding() 提取声纹 embedding，
    cosine_similarity() 计算归一化 [0, 1] 相似度分数。

    用法:
        recognizer = WespeakerDeep()
        recognizer.enroll("audio.wav", pk_path="voice.pkl")
        result = recognizer.recognize("test.wav", "voice.pkl")
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        sample_rate: int = 16000,
        config: DeepConfig | None = None,
        package_pk_index: int | None = None,
    ) -> None:
        """初始化。

        Args:
            model_path: 模型路径。None 使用内置 _models/vblinkf 模型，
                本地路径则从本地加载。
            device: 推理设备（官方模型内部处理，此参数保留以保持接口一致）。
            sample_rate: 采样率（官方模型固定 16000，此参数保留以保持接口一致）。
            config: DeepConfig 配置对象。None 时使用默认 DeepConfig()。
            package_pk_index: 内置声纹索引，优先级高于 config 中的设置。
        """
        import wespeaker


        model_dir = Path(__file__).parent / "_models" / "vblinkf"
        self._model = wespeaker.load_model(str(model_dir))
        self._model_dir = str(model_dir.resolve())
        self.sample_rate = sample_rate
        self._deep_config = config if config is not None else DeepConfig()
        if package_pk_index is not None:
            self._deep_config.package_pk_index = package_pk_index

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
        """从 .pkl 加载声纹 embedding。

        Args:
            pk_path: .pkl 文件路径。

        Returns:
            256 维 numpy 数组。
        """
        data = self._load_raw(pk_path)
        return np.asarray(data, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  注册
    # ------------------------------------------------------------------ #

    def enroll(
        self,
        audio_path: str | Path,
        pk_path: str | Path = "voice.pkl",
    ) -> dict[str, Any]:
        """注册声纹 — 提取 embedding 并保存到 .pkl。

        使用官方 WeSpeaker 模型提取音频的声纹 embedding，
        保存为标准 numpy 数组格式的 .pkl 文件。

        Args:
            audio_path: 注册音频文件路径。
            pk_path: 输出 .pkl 文件路径，默认 "voice.pkl"。

        Returns:
            {"ok": bool, "embedding_dim": int, "pk_path": str}
            如果失败，额外包含 "error" 字段。
        """
        audio_path = str(Path(audio_path))
        if not Path(audio_path).is_file():
            return {"ok": False, "error": f"文件不存在: {audio_path}"}

        embedding = self._model.extract_embedding(audio_path)
        if embedding is None:
            return {"ok": False, "error": "音频中未检测到有效语音"}

        out = Path(pk_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(embedding.cpu().numpy(), f)

        logger.info(
            "Enrolled: %s → %s (dim=%d)", audio_path, out, embedding.shape[0]
        )

        return {
            "ok": True,
            "embedding_dim": embedding.shape[0],
            "pk_path": str(out.resolve()),
        }

    # ------------------------------------------------------------------ #
    #  识别
    # ------------------------------------------------------------------ #

    def recognize(
        self,
        audio_path: str | Path,
        voiceprint: np.ndarray | str | Path | None = None,
    ) -> dict[str, Any]:
        """将音频与声纹比对，返回识别结果。

        使用官方 WeSpeaker 模型的 cosine_similarity() 计算相似度，
        分数范围归一化为 [0, 1]（和 official-demo.py 一致）。

        Args:
            audio_path: 待测试音频文件路径。
            voiceprint: 声纹，可以是 numpy 数组 / .pkl 路径。
                None 时使用内置声纹（由 package_pk_index 或默认 John 决定）。

        Returns:
            {"is_recognized": bool, "confidence": float, "threshold": float}
            如果出错，额外包含 "error" 字段。
        """
        audio_path = str(Path(audio_path))
        if not Path(audio_path).is_file():
            return {
                "is_recognized": False,
                "confidence": 0.0,
                "error": f"文件不存在: {audio_path}",
            }

        # ---- 解析声纹路径 ----
        if voiceprint is None:
            from ._voiceprints import get_voiceprint_path

            index = (
                self._deep_config.package_pk_index
                if self._deep_config.package_pk_index is not None
                else 0
            )
            voiceprint = get_voiceprint_path(index)

        # ---- 加载声纹 ----
        if isinstance(voiceprint, np.ndarray):
            import torch

            ref_emb = torch.from_numpy(voiceprint.astype(np.float32))
        else:
            ref_data = self.load(voiceprint)
            import torch

            ref_emb = torch.from_numpy(ref_data.astype(np.float32))

        # ---- 提取测试音频 embedding ----
        test_emb = self._model.extract_embedding(audio_path)
        if test_emb is None:
            return {
                "is_recognized": False,
                "confidence": 0.0,
                "error": "音频中未检测到有效语音",
            }

        # ---- 计算相似度 ----
        score = self._model.cosine_similarity(test_emb, ref_emb)

        threshold = self._deep_config.sim_threshold
        logger.debug(
            "Recognize: score=%.4f, threshold=%.2f, result=%s",
            score,
            threshold,
            score >= threshold,
        )

        return {
            "is_recognized": score >= threshold,
            "confidence": round(score, 4),
            "threshold": threshold,
        }
