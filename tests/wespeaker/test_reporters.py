"""Tests for wespeaker.reporters module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.wespeaker.reporters import JsonDataExporter, MarkdownReportGenerator


def test_json_exporter_creates_file(tmp_path: Path):
    """Test that JsonDataExporter creates a JSON file."""
    data = {"meta": {"test": "data"}, "results": []}
    exporter = JsonDataExporter(tmp_path)

    output_path = exporter.export(data)

    assert output_path.exists()
    content = json.loads(output_path.read_text())
    assert content["meta"]["test"] == "data"


def test_json_exporter_with_timestamp(tmp_path: Path):
    """Test that JsonDataExporter uses provided timestamp in filename."""
    data = {"test": "data"}
    exporter = JsonDataExporter(tmp_path)
    timestamp = datetime(2026, 5, 12, 14, 30, 45)

    output_path = exporter.export(data, timestamp=timestamp)

    assert "cross_test_data_20260512_143045" in output_path.name


def test_json_exporter_creates_output_dir(tmp_path: Path):
    """Test that JsonDataExporter creates output directory if it doesn't exist."""
    data = {"test": "data"}
    output_dir = tmp_path / "nested" / "dir"
    exporter = JsonDataExporter(output_dir)

    output_path = exporter.export(data)

    assert output_dir.exists()
    assert output_path.exists()


def test_json_exporter_converts_torch_tensor(tmp_path: Path):
    """Test that JsonDataExporter converts torch.Tensor to list."""
    import torch

    data = {"tensor": torch.tensor([1.0, 2.0, 3.0])}
    exporter = JsonDataExporter(tmp_path)

    output_path = exporter.export(data)
    content = json.loads(output_path.read_text())

    assert content["tensor"] == [1.0, 2.0, 3.0]


def test_json_exporter_converts_numpy_array(tmp_path: Path):
    """Test that JsonDataExporter converts numpy.ndarray to list."""
    import numpy as np

    data = {"array": np.array([4.0, 5.0, 6.0])}
    exporter = JsonDataExporter(tmp_path)

    output_path = exporter.export(data)
    content = json.loads(output_path.read_text())

    assert content["array"] == [4.0, 5.0, 6.0]


def test_json_exporter_handles_nested_data(tmp_path: Path):
    """Test that JsonDataExporter recursively converts nested structures."""
    import numpy as np
    import torch

    data = {
        "level1": {
            "tensor": torch.tensor([1, 2]),
            "level2": {
                "array": np.array([3, 4]),
                "list": [torch.tensor([5, 6]), "string"],
            },
        }
    }
    exporter = JsonDataExporter(tmp_path)

    output_path = exporter.export(data)
    content = json.loads(output_path.read_text())

    assert content["level1"]["tensor"] == [1, 2]
    assert content["level1"]["level2"]["array"] == [3, 4]
    assert content["level1"]["level2"]["list"][0] == [5, 6]
    assert content["level1"]["level2"]["list"][1] == "string"


def test_json_exporter_unicode_support(tmp_path: Path):
    """Test that JsonDataExporter handles Unicode characters correctly."""
    data = {"chinese": "声纹识别", "emoji": "🎤"}
    exporter = JsonDataExporter(tmp_path)

    output_path = exporter.export(data)
    content = json.loads(output_path.read_text())

    assert content["chinese"] == "声纹识别"
    assert content["emoji"] == "🎤"


def test_markdown_generator_creates_file(tmp_path: Path):
    """Test that MarkdownReportGenerator creates a markdown file."""
    gen = MarkdownReportGenerator(tmp_path)

    data = {
        "meta": {
            "timestamp": datetime(2025, 1, 12, 14, 30),
            "threshold": 0.55,
            "snr_levels": [20, 15, 10, 5, 0],
            "speakers": ["John", "Frank"],
        },
        "registration": {},
        "recognition": {"errors": {}},
    }

    output_path = gen.generate(data)

    assert output_path.exists()
    content = output_path.read_text()
    assert "# 声纹交叉测试诊断报告" in content
    assert "0.55" in content


def test_markdown_generator_with_full_data(tmp_path: Path):
    """Test that MarkdownReportGenerator generates complete report with all sections."""
    gen = MarkdownReportGenerator(tmp_path)

    data = {
        "meta": {
            "timestamp": datetime(2025, 1, 12, 14, 30),
            "threshold": 0.55,
            "snr_levels": [20, 15, 10, 5, 0],
            "speakers": ["John", "Frank"],
        },
        "registration": {
            "John": {
                "num_segments": 5,
                "total_embeddings": 25,
                "embedding_dim": 256,
                "quality_metrics": {
                    "l2_norms": {"min": 0.95, "max": 1.05, "mean": 1.0},
                    "cosine_distances": {"min": 0.1, "max": 0.3, "std": 0.05},
                    "within_class_compactness": 0.85,
                },
                "noise_effects": [
                    {"target_snr": 20, "original_rms": 0.1, "mixed_rms": 0.12},
                    {"target_snr": 10, "original_rms": 0.1, "mixed_rms": 0.15},
                ],
            },
            "Frank": {
                "num_segments": 4,
                "total_embeddings": 20,
                "embedding_dim": 256,
                "quality_metrics": {
                    "l2_norms": {"min": 0.92, "max": 1.08, "mean": 1.0},
                },
            },
        },
        "recognition": {
            "performance": {
                "avg_recognition_time": 0.15,
                "total_time": 5.5,
                "timings": {
                    "embedding_extraction": {"avg_time": 0.1, "count": 50},
                    "similarity_calculation": {"avg_time": 0.05, "count": 50},
                },
            },
            "errors": {
                "false_accepts": [
                    {
                        "test_speaker": "Frank",
                        "mistaken_as": "John",
                        "score": 0.58,
                        "threshold_distance": 0.03,
                    }
                ],
                "false_rejects": [
                    {
                        "test_speaker": "John",
                        "test_variant": "noisy",
                        "score": 0.52,
                        "threshold_distance": -0.03,
                    }
                ],
            },
            "test_cases": [
                {"test_variant": "clean", "confidence": 0.85},
                {"test_variant": "clean", "confidence": 0.90},
                {"test_variant": "noisy", "confidence": 0.65},
                {"test_variant": "noisy", "confidence": 0.70},
            ],
        },
    }

    output_path = gen.generate(data, timestamp=datetime(2025, 1, 12, 14, 30))

    assert output_path.exists()
    content = output_path.read_text()

    # Check main sections
    assert "# 声纹交叉测试诊断报告" in content
    assert "## 测试配置" in content
    assert "## 注册阶段分析" in content
    assert "## 识别阶段分析" in content
    assert "## 结论与建议" in content

    # Check registration data
    assert "### John" in content
    assert "片段数量: 5" in content
    assert "总 embedding 数: 25" in content
    assert "L2 范数: min=0.9500" in content
    assert "类内紧密度: 0.8500" in content
    assert "SNR 20dB:" in content
    assert "原始RMS=0.1000, 混合RMS=0.1200" in content

    # Check performance data
    assert "### 性能统计" in content
    assert "平均识别时间: 0.1500s" in content
    assert "总执行时间: 5.50s" in content
    assert "embedding_extraction: 0.1000s (x50)" in content

    # Check error cases
    assert "### 错误案例分析" in content
    assert "误接受 (1 例):" in content
    assert "Frank 被误认为 John: 得分=0.5800" in content
    assert "误拒绝 (1 例):" in content
    assert "John (noisy): 得分=0.5200" in content

    # Check variant analysis
    assert "### 音频变体性能分析" in content
    assert "**clean:**" in content
    assert "**noisy:**" in content
    assert "平均得分: 0.8750" in content  # clean average

    # Check conclusions
    assert "存在 1 例误接受" in content
    assert "存在 1 例误拒绝" in content
    assert "建议阈值范围:" in content
