"""Data export and reporting utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class JsonDataExporter:
    """JSON 数据导出器.

    将数据导出为 JSON 文件，支持自动转换 torch.Tensor 和 numpy.ndarray
    为 JSON 可序列化格式。

    Attributes:
        output_dir: 输出目录路径

    Example:
        >>> exporter = JsonDataExporter(Path("output"))
        >>> data = {"embedding": torch.tensor([1, 2, 3])}
        >>> output_path = exporter.export(data)
    """

    output_dir: Path

    def export(self, data: dict[str, Any], timestamp: datetime | None = None) -> Path:
        """导出数据为 JSON 文件.

        Args:
            data: 要导出的数据字典
            timestamp: 可选的时间戳，用于生成文件名。默认为当前时间。

        Returns:
            导出的 JSON 文件路径

        Raises:
            OSError: 文件写入失败
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"cross_test_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_dir / filename

        # 转换 torch.Tensor 为 list
        json_ready = self._make_json_serializable(data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_ready, f, ensure_ascii=False, indent=2)

        return output_path

    def _make_json_serializable(self, data: Any) -> Any:
        """递归转换数据为 JSON 可序列化格式.

        支持转换以下类型：
        - torch.Tensor -> list
        - numpy.ndarray -> list
        - dict: 递归处理所有值
        - list/tuple: 递归处理所有元素

        Args:
            data: 任意数据

        Returns:
            JSON 可序列化的数据
        """
        import numpy as np
        import torch

        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        return data
