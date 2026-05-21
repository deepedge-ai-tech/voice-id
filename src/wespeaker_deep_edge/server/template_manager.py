"""多模板加载与矩阵批量比对。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..wespeaker_deep_dege import WespeakerDeep

logger = logging.getLogger(__name__)

# 预设声纹 ID → .pkl 文件名映射
_PRESET_MAP: dict[str, str] = {
    "preset_john": "voice_john.pkl",
    "preset_frank": "voice_frank.pkl",
    "preset_michael": "voice_michael.pkl",
    "preset_qingqing": "voice_qingqing.pkl",
    "preset_xixi": "voice_xixi.pkl",
    "preset_zhong": "voice_zhong.pkl",
    "preset_angle": "voice_angle.pkl",
    "preset_john_usb_yun": "voice_john_usb_yun.pkl",
}


class TemplateManager:
    """管理声纹模板，支持多模板矩阵批量比对。

    每个模板是一个 256 维 embedding。load() 时加载到内存字典，
    recognize() 时将所有模板堆叠为 [N, 256] 矩阵，做批量 cosine similarity。
    """

    def __init__(
        self,
        wespeaker: WespeakerDeep,
        voiceprints_dir: str,
        storage_dir: str,
    ) -> None:
        self._wespeaker = wespeaker
        self._voiceprints_dir = Path(voiceprints_dir)
        self._storage_dir = Path(storage_dir)
        self._templates: dict[str, np.ndarray] = {}

    @property
    def template_count(self) -> int:
        """返回已加载的模板数量。"""
        return len(self._templates)

    @property
    def template_ids(self) -> list[str]:
        """返回所有已加载的模板 ID 列表。"""
        return list(self._templates.keys())

    def load(self, ids: list[str]) -> list[str]:
        """加载多个 .pkl 到内存模板库。

        内置声纹用 ``preset_`` 前缀标识（如 preset_john），
        从 ``_voiceprints/`` 目录加载。
        用户注册声纹从 ``storage_dir`` 加载（文件名为 ``{id}.pkl``）。

        Args:
            ids: 声纹 ID 列表。

        Returns:
            实际加载成功的 ID 列表。

        Raises:
            FileNotFoundError: 任一 .pkl 文件不存在。
        """
        loaded: list[str] = []
        for tid in ids:
            if tid.startswith("preset_"):
                filename = _PRESET_MAP.get(tid)
                if filename is None:
                    logger.warning("未知预设声纹: %s", tid)
                    continue
                pk_path = self._voiceprints_dir / filename
            else:
                pk_path = self._storage_dir / f"{tid}.pkl"

            if not pk_path.is_file():
                raise FileNotFoundError(f"声纹文件不存在: {pk_path}")

            data = self._wespeaker.load(str(pk_path))
            self._templates[tid] = np.asarray(data, dtype=np.float32)
            loaded.append(tid)
            logger.info("已加载声纹模板: %s (%s)", tid, pk_path)

        return loaded

    def recognize(self, embedding: np.ndarray) -> tuple[str, float]:
        """批量矩阵 cosine similarity。

        将所有模板堆叠为 [N, 256] 矩阵，一次矩阵乘法计算所有分数，
        返回最高分及其对应模板 ID。

        Args:
            embedding: 测试音频的 256 维 embedding。

        Returns:
            (best_id, max_score) 最高分模板 ID 和分数。

        Raises:
            ValueError: 模板库为空。
        """
        if not self._templates:
            raise ValueError("模板库为空，请先调用 load()")

        emb_matrix = np.stack(list(self._templates.values()))  # [N, 256]
        audio_vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)  # [1, 256]

        norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(audio_vec)
        scores = (emb_matrix @ audio_vec.T).flatten() / norms  # [N]

        best_idx = int(np.argmax(scores))
        best_id = list(self._templates.keys())[best_idx]
        return best_id, float(scores[best_idx])
