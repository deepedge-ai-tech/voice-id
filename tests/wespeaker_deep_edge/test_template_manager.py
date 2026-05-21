"""TemplateManager 单元测试。"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_wespeaker() -> MagicMock:
    mock = MagicMock()
    mock.load.side_effect = lambda p: np.random.randn(256).astype(np.float32)
    return mock


@pytest.fixture
def template_manager(mock_wespeaker, tmp_path) -> tuple:
    vp_dir = tmp_path / "voiceprints"
    vp_dir.mkdir()
    st_dir = tmp_path / "storage"
    st_dir.mkdir()

    from src.wespeaker_deep_edge.server.template_manager import TemplateManager

    mgr = TemplateManager(mock_wespeaker, str(vp_dir), str(st_dir))
    return mgr, vp_dir, st_dir, mock_wespeaker


# ============================================================================ #
#  load() 测试
# ============================================================================ #


class TestTemplateManagerLoad:
    def test_load_preset_ids(self, template_manager) -> None:
        mgr, vp_dir, _, mock_wespeaker = template_manager
        emb = np.random.randn(256).astype(np.float32)
        for name in ["voice_john.pkl", "voice_frank.pkl"]:
            with open(vp_dir / name, "wb") as f:
                pickle.dump(emb, f)
        mock_wespeaker.load.return_value = emb

        loaded = mgr.load(["preset_john", "preset_frank"])
        assert loaded == ["preset_john", "preset_frank"]
        assert mgr.template_count == 2

    def test_load_user_ids(self, template_manager) -> None:
        mgr, _, st_dir, mock_wespeaker = template_manager
        emb = np.random.randn(256).astype(np.float32)
        with open(st_dir / "user_001.pkl", "wb") as f:
            pickle.dump(emb, f)
        mock_wespeaker.load.return_value = emb

        loaded = mgr.load(["user_001"])
        assert loaded == ["user_001"]
        assert mgr.template_count == 1

    def test_load_nonexistent_raises(self, template_manager) -> None:
        mgr, _, _, _ = template_manager
        with pytest.raises(FileNotFoundError):
            mgr.load(["nonexistent"])

    def test_load_unknown_preset_skipped(self, template_manager) -> None:
        mgr, vp_dir, _, mock_wespeaker = template_manager
        emb = np.random.randn(256).astype(np.float32)
        with open(vp_dir / "voice_john.pkl", "wb") as f:
            pickle.dump(emb, f)
        mock_wespeaker.load.return_value = emb

        loaded = mgr.load(["preset_john", "preset_unknown_xyz"])
        assert loaded == ["preset_john"]  # unknown preset skipped
        assert mgr.template_count == 1


# ============================================================================ #
#  recognize() 测试
# ============================================================================ #


class TestTemplateManagerRecognize:
    def test_recognize_returns_best_match(self, template_manager) -> None:
        mgr, vp_dir, _, mock_wespeaker = template_manager
        emb_john = np.array([1.0] * 256, dtype=np.float32)
        emb_frank = np.array([-1.0] * 256, dtype=np.float32)
        for name, data in [("voice_john.pkl", emb_john), ("voice_frank.pkl", emb_frank)]:
            with open(vp_dir / name, "wb") as f:
                pickle.dump(data, f)
        mock_wespeaker.load.side_effect = lambda p: (
            emb_john if "voice_john" in str(p) else emb_frank
        )
        mgr.load(["preset_john", "preset_frank"])

        best_id, score = mgr.recognize(np.array([0.9] * 256, dtype=np.float32))
        assert best_id == "preset_john"
        assert score > 0.9

    def test_recognize_with_frank_match(self, template_manager) -> None:
        mgr, vp_dir, _, mock_wespeaker = template_manager
        emb_john = np.array([1.0] * 256, dtype=np.float32)
        emb_frank = np.array([-1.0] * 256, dtype=np.float32)
        for name, data in [("voice_john.pkl", emb_john), ("voice_frank.pkl", emb_frank)]:
            with open(vp_dir / name, "wb") as f:
                pickle.dump(data, f)
        mock_wespeaker.load.side_effect = lambda p: (
            emb_john if "voice_john" in str(p) else emb_frank
        )
        mgr.load(["preset_john", "preset_frank"])

        best_id, score = mgr.recognize(np.array([-0.9] * 256, dtype=np.float32))
        assert best_id == "preset_frank"
        assert score > 0.9

    def test_recognize_empty_raises(self, mock_wespeaker, tmp_path) -> None:
        from src.wespeaker_deep_edge.server.template_manager import TemplateManager

        mgr = TemplateManager(mock_wespeaker, str(tmp_path), str(tmp_path))
        with pytest.raises(ValueError, match="模板库为空"):
            mgr.recognize(np.random.randn(256).astype(np.float32))


# ============================================================================ #
#  properties
# ============================================================================ #


class TestTemplateManagerProperties:
    def test_template_ids_after_load(self, template_manager) -> None:
        mgr, vp_dir, _, mock_wespeaker = template_manager
        emb = np.random.randn(256).astype(np.float32)
        with open(vp_dir / "voice_john.pkl", "wb") as f:
            pickle.dump(emb, f)
        with open(vp_dir / "voice_frank.pkl", "wb") as f:
            pickle.dump(emb, f)
        mock_wespeaker.load.return_value = emb

        mgr.load(["preset_john", "preset_frank"])
        assert mgr.template_ids == ["preset_john", "preset_frank"]

    def test_initial_state(self, mock_wespeaker, tmp_path) -> None:
        from src.wespeaker_deep_edge.server.template_manager import TemplateManager

        mgr = TemplateManager(mock_wespeaker, str(tmp_path), str(tmp_path))
        assert mgr.template_count == 0
        assert mgr.template_ids == []
