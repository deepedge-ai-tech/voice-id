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

import importlib.util
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  RecognitionResult — 识别结果
# --------------------------------------------------------------------------- #


class RecognitionResult(NamedTuple):
    """声纹识别结果。

    Attributes:
        is_recognized: 是否匹配（置信度 ≥ 阈值）。
        confidence: 余弦相似度 [0, 1]。
        name: 匹配的说话人名称。
        all_scores: 所有模板的相似度字典 {name: score}，None 表示未启用。
    """

    is_recognized: bool
    confidence: float
    name: str
    all_scores: dict[str, float] | None = None


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
            device: 推理设备，"cpu" 或 "cuda"。设为 "cuda" 时自动使用 GPU。
            sample_rate: 采样率（官方模型固定 16000，此参数保留以保持接口一致）。
            config: DeepConfig 配置对象。None 时使用默认 DeepConfig()。
            package_pk_index: 内置声纹索引，优先级高于 config 中的设置。
        """
        # 首次使用时加载 vendored wespeaker（位于 _wespeaker/），
        # 通过 importlib 注册确保子包绝对导入正确解析
        if 'wespeaker' not in sys.modules:
            _wespeaker_dir = Path(__file__).parent / "_wespeaker" / "wespeaker"
            _init_file = _wespeaker_dir / "__init__.py"
            _spec = importlib.util.spec_from_file_location(
                "wespeaker",
                str(_init_file),
                submodule_search_locations=[str(_wespeaker_dir)],
            )
            if _spec is None or _spec.loader is None:
                raise ImportError(f"cannot load vendored wespeaker: {_wespeaker_dir}")
            _module = importlib.util.module_from_spec(_spec)
            sys.modules['wespeaker'] = _module
            _spec.loader.exec_module(_module)
        import wespeaker


        model_dir = Path(__file__).parent / "_models" / "vblinkf"
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self._model = wespeaker.load_model(
            str(model_dir),
            dtype="float16" if use_cuda else "float32",
        )
        if use_cuda:
            self._model.set_device("cuda")
            logger.info("Model loaded on GPU (float16)")
        else:
            reason = "device not set to cuda" if device != "cuda" else "CUDA not available"
            logger.info("Model loaded on CPU (float32, %s)", reason)
        self._model_dir = str(model_dir.resolve())
        self.sample_rate = sample_rate
        self._deep_config = config if config is not None else DeepConfig()
        # 预计算模板矩阵: [N, 256] + 名称列表 + 行 norms
        self._template_matrix: np.ndarray | None = None
        self._template_names: list[str] = []
        self._template_norms: np.ndarray | None = None
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

    def load_templates(
        self,
        indices: list[int] | None = None,
        files: dict[str, str | Path] | None = None,
    ) -> None:
        """加载声纹模板到内存，预计算匹配矩阵。

        支持内置声纹（通过 index）和自定义 .pkl 文件。
        模板被堆叠为 [N, 256] 矩阵并预计算 L2 norms，
        后续 ``recognize_multi()`` / ``recognize_multi_pcm()`` 直接做矩阵乘法。

        Args:
            indices: 内置声纹索引列表，如 [0, 1, 2]。
            files: 自定义声纹文件，如 {"target": "path/to/voice.pkl"}。
        """
        from ._voiceprints import get_voiceprint_path, get_voiceprint_name

        embs: list[np.ndarray] = []
        names: list[str] = []

        if indices:
            for idx in indices:
                ref_data = self.load(get_voiceprint_path(idx))
                embs.append(ref_data.astype(np.float32))
                names.append(get_voiceprint_name(idx))

        if files:
            for name, path in files.items():
                ref_data = self.load(str(path))
                embs.append(ref_data.astype(np.float32))
                names.append(name)

        if not embs:
            raise ValueError("未提供任何声纹模板")

        self._template_matrix = np.stack(embs)  # [N, 256]
        self._template_names = names
        self._template_norms = np.linalg.norm(self._template_matrix, axis=1)

        logger.info("Loaded %d templates: %s", len(names), names)

    def clear_templates(self) -> None:
        """清空声纹模板缓存。"""
        self._template_matrix = None
        self._template_names = []
        self._template_norms = None

    # ------------------------------------------------------------------ #
    #  批量识别（内部 + 公开）
    # ------------------------------------------------------------------ #

    def _match_templates(self, test_emb: torch.Tensor) -> RecognitionResult:
        """内部：与已缓存的模板做批量余弦相似度，返回最佳匹配。

        Args:
            test_emb: 测试音频的 embedding tensor。

        Returns:
            RecognitionResult。

        Raises:
            RuntimeError: 模板缓存为空。
        """
        if self._template_matrix is None:
            raise RuntimeError("模板为空，请先调用 load_templates()")

        audio_vec = np.asarray(test_emb.cpu().numpy(), dtype=np.float32).reshape(1, -1)
        audio_norm = np.linalg.norm(audio_vec)

        scores = (self._template_matrix @ audio_vec.T).flatten() / (self._template_norms * audio_norm)
        scores = (scores + 1.0) / 2  # [-1, 1] => [0, 1]

        best_pos = int(np.argmax(scores))
        threshold = self._deep_config.sim_threshold

        logger.debug(
            "Match templates: best=%s (score=%.4f, threshold=%.2f)",
            self._template_names[best_pos], scores[best_pos], threshold,
        )

        return RecognitionResult(
            is_recognized=bool(scores[best_pos] >= threshold),
            confidence=round(float(scores[best_pos]), 4),
            name=self._template_names[best_pos],
            all_scores={
                name: round(float(score), 4)
                for name, score in zip(self._template_names, scores.tolist())
            },
        )

    def recognize_multi(self, audio_path: str | Path) -> RecognitionResult:
        """比对音频与已缓存的模板，返回最佳匹配。

        Args:
            audio_path: WAV 文件路径。

        Returns:
            RecognitionResult。

        Raises:
            FileNotFoundError: 音频文件不存在。
            ValueError: 音频中未检测到有效语音。
            RuntimeError: 模板缓存为空。
        """
        audio_path = str(Path(audio_path))
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"文件不存在: {audio_path}")

        test_emb = self._model.extract_embedding(audio_path)
        if test_emb is None:
            raise ValueError("音频中未检测到有效语音")

        return self._match_templates(test_emb)

    def recognize_multi_pcm(
        self,
        pcm: np.ndarray,
        sample_rate: int = 16000,
    ) -> RecognitionResult:
        """比对 16-bit 16kHz PCM 数据与已缓存的模板，返回最佳匹配。

        适用于流式/麦克风场景，pcm 为 int16 数组（16kHz 单声道）。
        内部自动将 int16 归一化为 float32。

        Args:
            pcm: 16-bit PCM 音频数据，形状 (samples,) 或 (1, samples)。
            sample_rate: 采样率，默认 16000。

        Returns:
            RecognitionResult。

        Raises:
            ValueError: 音频数据无效或未检测到有效语音。
            RuntimeError: 模板缓存为空。
        """
        if pcm.ndim == 1:
            pcm = pcm.reshape(1, -1)

        if pcm.dtype == np.int16:
            pcm_tensor = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
        else:
            pcm_tensor = torch.from_numpy(pcm.astype(np.float32))

        test_emb = self._model.extract_embedding_from_pcm(pcm_tensor, sample_rate)
        if test_emb is None:
            raise ValueError("音频中未检测到有效语音")

        return self._match_templates(test_emb)
