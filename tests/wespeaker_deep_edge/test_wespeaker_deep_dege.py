"""WeSpeaker Deep 核心功能测试。

测试 DeepConfig 数据类和 WespeakerDeep 类。
通过 mock wespeaker.load_model 来跳过实际模型加载。
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.wespeaker_deep_edge.wespeaker_deep_dege import DeepConfig, WespeakerDeep

# ============================================================================ #
#  DeepConfig Tests
# ============================================================================ #


class TestDeepConfigDefaults:
    """DeepConfig 默认值测试。"""

    def test_default_sim_threshold(self) -> None:
        assert DeepConfig().sim_threshold == 0.70

    def test_default_package_pk_index(self) -> None:
        assert DeepConfig().package_pk_index is None

    def test_custom_value_should_override_default(self) -> None:
        cfg = DeepConfig(sim_threshold=0.50, package_pk_index=3)
        assert cfg.sim_threshold == 0.50
        assert cfg.package_pk_index == 3


# ============================================================================ #
#  工具: 创建已 mock 的 WespeakerDeep 实例
# ============================================================================ #

FAKE_EMBEDDING = torch.randn(256)
FAKE_EMBEDDING_NORM = FAKE_EMBEDDING.numpy()


def _make_mock_model() -> MagicMock:
    """创建一个 fake _model，模拟官方 Speaker 的基本行为。"""
    mock = MagicMock()
    mock.extract_embedding.return_value = FAKE_EMBEDDING
    mock.cosine_similarity.return_value = 0.85
    return mock


def _make_deep(config: DeepConfig | None = None) -> WespeakerDeep:
    """创建 WespeakerDeep 并注入 fake _model，跳过模型加载。"""
    deep = WespeakerDeep.__new__(WespeakerDeep)
    deep._model = _make_mock_model()
    deep.sample_rate = 16000
    deep._deep_config = config if config is not None else DeepConfig()
    return deep


# ============================================================================ #
#  WespeakerDeep Tests — 初始化
# ============================================================================ #


class TestWespeakerDeepInit:
    """WespeakerDeep 初始化测试。"""

    def test_init_with_default_config(self) -> None:
        """默认 config 应为 sim_threshold=0.70。"""
        deep = _make_deep()
        assert deep._deep_config.sim_threshold == 0.70

    def test_init_with_custom_config(self) -> None:
        """传入自定义 config 应生效。"""
        cfg = DeepConfig(sim_threshold=0.80)
        deep = _make_deep(cfg)
        assert deep._deep_config.sim_threshold == 0.80

    def test_init_package_pk_index_override(self) -> None:
        """config 的 package_pk_index 应生效。"""
        cfg = DeepConfig(package_pk_index=2)
        deep = _make_deep(cfg)
        assert deep._deep_config.package_pk_index == 2

    def test_deep_config_property_returns_config(self) -> None:
        cfg = DeepConfig(sim_threshold=0.42)
        deep = _make_deep(cfg)
        assert deep.deep_config is cfg
        assert deep.deep_config.sim_threshold == 0.42

    def test_deep_config_default_is_mutable(self) -> None:
        deep = _make_deep()
        deep.deep_config.sim_threshold = 0.80
        assert deep.deep_config.sim_threshold == 0.80


# ============================================================================ #
#  WespeakerDeep Tests — _load_raw / load
# ============================================================================ #


class TestWespeakerDeepLoadRaw:
    """_load_raw() 测试。"""

    def test_load_raw_valid_file(self) -> None:
        pk_path = "/tmp/test_load_raw.pkl"
        data = {"key": "value"}
        with open(pk_path, "wb") as f:
            pickle.dump(data, f)
        result = WespeakerDeep._load_raw(pk_path)
        assert result == data

    def test_load_raw_nonexistent_file_should_raise(self) -> None:
        with pytest.raises(FileNotFoundError, match="声纹文件不存在"):
            WespeakerDeep._load_raw("/tmp/nonexistent_xyz.pkl")


class TestWespeakerDeepLoad:
    """load() 测试。"""

    def test_load_numpy_array(self) -> None:
        ref = FAKE_EMBEDDING_NORM
        pk_path = "/tmp/test_load.pkl"
        with open(pk_path, "wb") as f:
            pickle.dump(ref, f)
        deep = _make_deep()
        result = deep.load(pk_path)
        assert result.shape == (256,)
        assert isinstance(result, np.ndarray)

    def test_load_tensor(self) -> None:
        """兼容从 tensor 保存的 .pkl。"""
        ref_tensor = FAKE_EMBEDDING
        pk_path = "/tmp/test_load_tensor.pkl"
        with open(pk_path, "wb") as f:
            pickle.dump(ref_tensor.numpy(), f)
        deep = _make_deep()
        result = deep.load(pk_path)
        assert result.shape == (256,)


# ============================================================================ #
#  WespeakerDeep Tests — enroll()
# ============================================================================ #


class TestWespeakerDeepEnroll:
    """enroll() 测试。"""

    def test_enroll_nonexistent_audio_should_return_error(self) -> None:
        deep = _make_deep()
        result = deep.enroll("/tmp/nonexistent_audio.wav")
        assert result["ok"] is False
        assert "文件不存在" in result["error"]

    def test_enroll_no_speech_should_return_error(self) -> None:
        deep = _make_deep()
        deep._model.extract_embedding.return_value = None

        with patch("pathlib.Path.is_file", return_value=True):
            result = deep.enroll("/tmp/silence.wav", "/tmp/out.pkl")

        assert result["ok"] is False
        assert "未检测到有效语音" in result["error"]

    def test_enroll_happy_path(self) -> None:
        import os
        import tempfile

        deep = _make_deep()
        deep._model.extract_embedding.return_value = FAKE_EMBEDDING

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "enroll.wav")
            with open(wav_path, "w") as f:
                f.write("dummy")

            pk_path = os.path.join(tmpdir, "voice.pkl")
            with patch("pathlib.Path.is_file", return_value=True):
                result = deep.enroll(wav_path, pk_path=pk_path)

            assert result["ok"] is True
            assert result["embedding_dim"] == 256
            assert Path(pk_path).is_file()

            # 验证 .pkl 内容
            loaded = deep.load(pk_path)
            assert loaded.shape == (256,)

    def test_enroll_output_directory_created(self) -> None:
        deep = _make_deep()
        deep._model.extract_embedding.return_value = FAKE_EMBEDDING

        with patch("pathlib.Path.is_file", return_value=True):
            result = deep.enroll("/tmp/test.wav", "/tmp/new_dir/sub/voice.pkl")

        assert result["ok"] is True
        assert Path("/tmp/new_dir/sub/voice.pkl").is_file()


# ============================================================================ #
#  WespeakerDeep Tests — recognize()
# ============================================================================ #


class TestWespeakerDeepRecognize:
    """recognize() 测试。"""

    def test_recognize_nonexistent_audio_should_return_error(self) -> None:
        deep = _make_deep()
        result = deep.recognize("/tmp/nonexistent_audio.wav", FAKE_EMBEDDING_NORM)
        assert result["is_recognized"] is False
        assert result["confidence"] == 0.0
        assert "文件不存在" in result["error"]

    def test_recognize_no_speech_should_return_error(self) -> None:
        deep = _make_deep()
        deep._model.extract_embedding.return_value = None

        with patch("pathlib.Path.is_file", return_value=True):
            result = deep.recognize("/tmp/test.wav", FAKE_EMBEDDING_NORM)

        assert result["is_recognized"] is False
        assert result["confidence"] == 0.0
        assert "未检测到有效语音" in result["error"]

    def test_recognize_with_numpy_voiceprint_above_threshold(self) -> None:
        """分数高于阈值时应返回 is_recognized=True。"""
        deep = _make_deep()
        deep._model.cosine_similarity.return_value = 0.85

        with patch("pathlib.Path.is_file", return_value=True):
            result = deep.recognize("/tmp/test.wav", FAKE_EMBEDDING_NORM)

        assert result["is_recognized"] is True
        assert result["confidence"] == 0.85
        assert result["threshold"] == 0.70

    def test_recognize_below_threshold_should_reject(self) -> None:
        """分数低于阈值时应返回 is_recognized=False。"""
        cfg = DeepConfig(sim_threshold=0.90)
        deep = _make_deep(cfg)
        deep._model.cosine_similarity.return_value = 0.50

        with patch("pathlib.Path.is_file", return_value=True):
            result = deep.recognize("/tmp/test.wav", FAKE_EMBEDDING_NORM)

        assert result["is_recognized"] is False
        assert result["confidence"] == 0.50

    def test_recognize_cosine_similarity_called_with_correct_args(self) -> None:
        """验证 cosine_similarity 被正确调用。"""
        deep = _make_deep()
        deep._model.cosine_similarity.return_value = 0.75

        with patch("pathlib.Path.is_file", return_value=True):
            deep.recognize("/tmp/test.wav", FAKE_EMBEDDING_NORM)

        # 应该用 test_emb 和 ref_emb 调用 cosine_similarity
        deep._model.cosine_similarity.assert_called_once()
        args = deep._model.cosine_similarity.call_args[0]
        assert len(args) == 2
        # args[0] 是 test_emb（FAKE_EMBEDDING），args[1] 是 ref_emb
        assert torch.equal(args[0], FAKE_EMBEDDING)

    def test_recognize_with_pkl_path(self) -> None:
        """从 .pkl 文件加载声纹。"""
        deep = _make_deep()
        deep._model.cosine_similarity.return_value = 0.90

        # 创建声纹 .pkl
        vp_path = "/tmp/test_vp_recognize.pkl"
        with open(vp_path, "wb") as f:
            pickle.dump(FAKE_EMBEDDING_NORM, f)

        with patch("pathlib.Path.is_file", return_value=True):
            result = deep.recognize("/tmp/test.wav", vp_path)

        assert result["is_recognized"] is True
        assert result["confidence"] == 0.90

    def test_recognize_with_none_voiceprint_uses_builtin(self) -> None:
        """voiceprint=None 时应使用内置声纹。"""
        deep = _make_deep()
        deep._model.cosine_similarity.return_value = 0.80

        with (
            patch("pathlib.Path.is_file", return_value=True) as mock_isfile,
            patch(
                "src.wespeaker_deep_edge._voiceprints.get_voiceprint_path",
                return_value="/fake/builtin.pkl",
            ) as mock_gvp,
            patch.object(deep, "load", return_value=FAKE_EMBEDDING_NORM) as mock_load,
        ):
            # 让 is_file 对测试音频和内置声纹都返回 True
            mock_isfile.return_value = True
            result = deep.recognize("/tmp/test.wav", voiceprint=None)

        assert result["is_recognized"] is True
        mock_gvp.assert_called_once_with(0)  # 默认 index 0
        mock_load.assert_called_once_with("/fake/builtin.pkl")

    def test_recognize_with_package_pk_index(self) -> None:
        """package_pk_index 应覆盖默认内置声纹索引。"""
        cfg = DeepConfig(package_pk_index=3)
        deep = _make_deep(cfg)
        deep._model.cosine_similarity.return_value = 0.80

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch(
                "src.wespeaker_deep_edge._voiceprints.get_voiceprint_path",
                return_value="/fake/index3.pkl",
            ) as mock_gvp,
            patch.object(deep, "load", return_value=FAKE_EMBEDDING_NORM),
        ):
            result = deep.recognize("/tmp/test.wav", voiceprint=None)

        assert result["is_recognized"] is True
        mock_gvp.assert_called_once_with(3)

    def test_recognize_returns_correct_keys(self) -> None:
        """返回值应包含所有必要字段。"""
        deep = _make_deep()
        deep._model.cosine_similarity.return_value = 0.75

        with patch("pathlib.Path.is_file", return_value=True):
            result = deep.recognize("/tmp/test.wav", FAKE_EMBEDDING_NORM)

        assert set(result.keys()) == {"is_recognized", "confidence", "threshold"}


class TestWespeakerDeepRecognizeMulti:
    """WespeakerDeep.recognize_multi 测试。"""

    FAKE_EMBED_1 = np.random.randn(256).astype(np.float32)
    FAKE_EMBED_2 = np.random.randn(256).astype(np.float32)
    FAKE_EMBED_3 = np.random.randn(256).astype(np.float32)
    FAKE_TEST_EMB = np.random.randn(256).astype(np.float32)
    FAKE_TEST_TENSOR = torch.from_numpy(FAKE_TEST_EMB)

    def test_recognize_multi_nonexistent_audio_should_return_error(self) -> None:
        """传入不存在的音频应返回 error。"""
        deep = _make_deep()

        with patch("pathlib.Path.is_file", return_value=False):
            result = deep.recognize_multi("/nonexistent.wav", [0, 1])

        assert result["is_recognized"] is False
        assert "error" in result

    def test_recognize_multi_returns_best_match(self) -> None:
        """应在多个声纹中返回最佳匹配。"""
        deep = _make_deep()
        deep._model.extract_embedding.return_value = self.FAKE_TEST_TENSOR

        embeds = [self.FAKE_EMBED_1, self.FAKE_EMBED_2, self.FAKE_EMBED_3]
        names = ["john", "frank", "michael"]

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("src.wespeaker_deep_edge._voiceprints.get_voiceprint_path"),
            patch("src.wespeaker_deep_edge._voiceprints.get_voiceprint_name", side_effect=lambda i: names[i]),
            patch.object(WespeakerDeep, "load", side_effect=lambda _: embeds.pop(0)),
        ):
            result = deep.recognize_multi("/tmp/test.wav", [0, 1, 2])

        assert set(result.keys()) == {"is_recognized", "confidence", "name", "index", "threshold"}
        assert result["index"] in [0, 1, 2]
        assert isinstance(result["name"], str)
        assert isinstance(result["confidence"], float)
