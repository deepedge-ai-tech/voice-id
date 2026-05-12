"""Tests for wespeaker.reporters module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.wespeaker.reporters import JsonDataExporter


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
