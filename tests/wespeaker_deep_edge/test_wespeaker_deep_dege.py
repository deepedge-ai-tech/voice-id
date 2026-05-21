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

from src.wespeaker_deep_edge.wespeaker_deep_dege import DeepConfig, RecognitionResult, WespeakerDeep

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

    def test_default_enable_asnorm(self) -> None:
        assert DeepConfig().enable_asnorm is False

    def test_default_asnorm_top_k(self) -> None:
        assert DeepConfig().asnorm_top_k == 300

    def test_default_asnorm_cohort_path(self) -> None:
        assert DeepConfig().asnorm_cohort_path == "asset/cohort/cohort_embeddings.npy"


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
    deep._template_matrix = None
    deep._template_names = []
    deep._template_norms = None
    deep._cohort_cache = None
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
    """WespeakerDeep 双阶段多人声纹测试。"""

    FAKE_TEST_TENSOR = torch.from_numpy(np.random.randn(256).astype(np.float32))

    @staticmethod
    def _make_loaded() -> WespeakerDeep:
        """创建已预载模板的 WespeakerDeep 实例。"""
        deep = _make_deep()
        embs = [
            np.random.randn(256).astype(np.float32),
            np.random.randn(256).astype(np.float32),
            np.random.randn(256).astype(np.float32),
        ]
        deep._template_matrix = np.stack(embs)
        deep._template_names = ["john", "frank", "michael"]
        deep._template_norms = np.linalg.norm(deep._template_matrix, axis=1)
        return deep

    def test_empty_cache_should_raise(self) -> None:
        """未加载模板时调用应抛 RuntimeError。"""
        deep = _make_deep()
        with (
            patch("pathlib.Path.is_file", return_value=True),
        ):
            with pytest.raises(RuntimeError, match="模板为空"):
                deep.recognize_multi("/tmp/test.wav")

    def test_nonexistent_audio_should_raise(self) -> None:
        """不存在的音频应抛 FileNotFoundError。"""
        deep = self._make_loaded()
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            deep.recognize_multi("/nonexistent.wav")

    def test_recognize_multi_returns_recognition_result(self) -> None:
        """正常识别应返回 RecognitionResult。"""
        deep = self._make_loaded()
        deep._model.extract_embedding.return_value = self.FAKE_TEST_TENSOR

        with patch("pathlib.Path.is_file", return_value=True):
            result = deep.recognize_multi("/tmp/test.wav")

        assert isinstance(result, RecognitionResult)
        assert isinstance(result.name, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.is_recognized, bool)

    def test_recognize_multi_pcm_returns_recognition_result(self) -> None:
        """PCM 识别应返回 RecognitionResult。"""
        deep = self._make_loaded()
        deep._model.extract_embedding_from_pcm.return_value = self.FAKE_TEST_TENSOR

        pcm = np.random.randint(-32768, 32767, (1, 16000), dtype=np.int16)
        result = deep.recognize_multi_pcm(pcm, sample_rate=16000)

        assert isinstance(result, RecognitionResult)
        assert isinstance(result.name, str)

    def test_recognize_multi_pcm_int16_conversion(self) -> None:
        """int16 PCM 应被正确归一化。"""
        deep = self._make_loaded()
        deep._model.extract_embedding_from_pcm.return_value = self.FAKE_TEST_TENSOR
        mock = deep._model.extract_embedding_from_pcm

        pcm = np.array([[0, 16384, -32768, 32767]], dtype=np.int16)
        deep.recognize_multi_pcm(pcm, 16000)

        called_pcm = mock.call_args[0][0]
        assert called_pcm.dtype == torch.float32
        assert called_pcm[0, 0].item() == 0.0
        assert called_pcm[0, 2].item() == -1.0
        assert called_pcm[0, 3].item() == pytest.approx(1.0, abs=1e-3)

    def test_recognize_multi_returns_all_scores(self) -> None:
        """recognize_multi 应返回包含 all_scores 的 RecognitionResult。"""
        deep = self._make_loaded()
        deep._model.extract_embedding.return_value = self.FAKE_TEST_TENSOR

        with patch("pathlib.Path.is_file", return_value=True):
            result = deep.recognize_multi("/tmp/test.wav")

        assert result.all_scores is not None
        assert isinstance(result.all_scores, dict)
        assert "john" in result.all_scores
        assert "frank" in result.all_scores
        assert "michael" in result.all_scores
        # best confidence matches the dict value for the recognized name
        assert result.all_scores[result.name] == result.confidence

    def test_recognize_multi_pcm_returns_all_scores(self) -> None:
        """recognize_multi_pcm 应返回包含 all_scores 的 RecognitionResult。"""
        deep = self._make_loaded()
        deep._model.extract_embedding_from_pcm.return_value = self.FAKE_TEST_TENSOR

        pcm = np.random.randint(-32768, 32767, (1, 16000), dtype=np.int16)
        result = deep.recognize_multi_pcm(pcm, sample_rate=16000)

        assert result.all_scores is not None
        assert isinstance(result.all_scores, dict)
        assert set(result.all_scores.keys()) == {"john", "frank", "michael"}

    def test_recognize_multi_pcm_asnorm_no_cohort_returns_raw(self) -> None:
        """AS-Norm 启用但 cohort 未加载时返回原始结果。"""
        deep = self._make_loaded()
        deep._deep_config.enable_asnorm = True
        # _cohort_cache is None by default in _make_loaded
        deep._model.extract_embedding_from_pcm.return_value = self.FAKE_TEST_TENSOR

        pcm = np.random.randint(-32768, 32767, (1, 16000), dtype=np.int16)
        result = deep.recognize_multi_pcm(pcm, sample_rate=16000)

        assert result.all_scores is not None
        assert isinstance(result, RecognitionResult)

    def test_recognize_multi_pcm_asnorm_enabled_with_cohort(self) -> None:
        """AS-Norm 启用且 cohort 已加载时应用规范化。"""
        deep = self._make_loaded()
        deep._deep_config.enable_asnorm = True
        deep._model.extract_embedding_from_pcm.return_value = self.FAKE_TEST_TENSOR

        # Create mock cohort cache with precomputed stats
        num_cohort = 500
        mock_cache = MagicMock()
        mock_cache._enroll_mu = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        mock_cache._enroll_sigma = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        norm_scores = np.array([0.6, 0.4, 0.3], dtype=np.float32)
        mock_cache.apply.return_value = (norm_scores, mock_cache._enroll_mu, mock_cache._enroll_sigma)
        deep._cohort_cache = mock_cache

        pcm = np.random.randint(-32768, 32767, (1, 16000), dtype=np.int16)
        result = deep.recognize_multi_pcm(pcm, sample_rate=16000)

        assert isinstance(result, RecognitionResult)
        assert isinstance(result.is_recognized, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.name, str)
        assert result.all_scores is not None
        assert set(result.all_scores.keys()) == {"john", "frank", "michael"}


# ============================================================================ #
#  WespeakerDeep Tests — load_cohort()
# ============================================================================ #


class TestWespeakerDeepLoadCohort:
    """load_cohort() 测试。"""

    def test_load_cohort_without_asnorm_does_nothing(self) -> None:
        """enable_asnorm=False 时 load_cohort() 不执行任何操作。"""
        deep = _make_deep()  # enable_asnorm=False by default

        with patch(
            "src.wespeaker_deep_edge.wespeaker_deep_dege.CohortCache.load",
        ) as mock_load:
            deep.load_cohort()
            mock_load.assert_not_called()

        assert deep._cohort_cache is None

    def test_load_cohort_file_not_found(self) -> None:
        """文件不存在时应记录警告并设 _cohort_cache = None。"""
        cfg = DeepConfig(enable_asnorm=True)
        deep = _make_deep(cfg)

        with patch(
            "src.wespeaker_deep_edge.wespeaker_deep_dege.CohortCache.load",
            side_effect=FileNotFoundError("Not found"),
        ):
            deep.load_cohort()

        assert deep._cohort_cache is None

    def test_load_cohort_success(self) -> None:
        """正常加载应设置 _cohort_cache。"""
        cfg = DeepConfig(enable_asnorm=True)
        deep = _make_deep(cfg)

        mock_cache = MagicMock()
        mock_cache.size = 500

        with patch(
            "src.wespeaker_deep_edge.wespeaker_deep_dege.CohortCache.load",
            return_value=mock_cache,
        ):
            deep.load_cohort()

        assert deep._cohort_cache is not None
        assert deep._cohort_cache.size == 500
