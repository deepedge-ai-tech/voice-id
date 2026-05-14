"""WeSpeaker Deep 核心功能测试。

测试 DeepConfig 数据类 和 WespeakerDeep 类。
由于父类初始化需要模型文件（pyannote.audio），所有 WespeakerDeep 测试
通过 mock _client 来绕过模型加载。
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.wespeaker_deep_edge.best import BestConfig
from src.wespeaker_deep_edge.wespeaker_deep_dege import DeepConfig, WespeakerDeep

# ============================================================================ #
#  DeepConfig Tests
# ============================================================================ #


class TestDeepConfigDefaults:
    """DeepConfig 默认值测试。"""

    def test_default_sim_threshold(self) -> None:
        assert DeepConfig().sim_threshold == 0.50

    def test_default_verify_crop_mode(self) -> None:
        assert DeepConfig().verify_crop_mode == "head_window"

    def test_default_verify_window_secs(self) -> None:
        assert DeepConfig().verify_window_secs == 0.4

    def test_default_enrollment_segment_secs(self) -> None:
        assert DeepConfig().enrollment_segment_secs == 0.6

    def test_default_enable_vad(self) -> None:
        assert DeepConfig().enable_vad is False

    def test_default_enable_score_compensation(self) -> None:
        assert DeepConfig().enable_score_compensation is True

    def test_default_score_compensation_mode(self) -> None:
        assert DeepConfig().score_compensation_mode == "sqrt"

    def test_default_noise_injection_snrs(self) -> None:
        assert DeepConfig().noise_injection_snrs == ()

    def test_default_enroll_skip_vad(self) -> None:
        assert DeepConfig().enroll_skip_vad is True

    def test_default_enroll_clean_only(self) -> None:
        assert DeepConfig().enroll_clean_only is True

    def test_default_enable_multi_template(self) -> None:
        assert DeepConfig().enable_multi_template is True

    def test_default_enable_sliding_window_test(self) -> None:
        assert DeepConfig().enable_sliding_window_test is False

    def test_default_short_audio_max_duration(self) -> None:
        assert DeepConfig().short_audio_max_duration == 1.5

    def test_custom_value_should_override_default(self) -> None:
        cfg = DeepConfig(sim_threshold=0.70, enable_vad=True)
        assert cfg.sim_threshold == 0.70
        assert cfg.enable_vad is True
        # 未修改的保持默认
        assert cfg.enroll_skip_vad is True


class TestDeepConfigToDict:
    """DeepConfig.to_dict() 测试。"""

    def test_to_dict_contains_all_keys(self) -> None:
        d = DeepConfig().to_dict()
        assert isinstance(d, dict)
        assert d["sim_threshold"] == 0.50
        assert d["verify_crop_mode"] == "head_window"
        assert d["enroll_clean_only"] is True
        assert d["enable_multi_template"] is True
        assert d["enable_sliding_window_test"] is False
        assert d["short_audio_max_duration"] == 1.5

    def test_to_dict_noise_injection_snrs_is_list(self) -> None:
        cfg = DeepConfig(noise_injection_snrs=(20, 15, 10))
        d = cfg.to_dict()
        assert d["noise_injection_snrs"] == [20, 15, 10]


class TestDeepConfigFromDict:
    """DeepConfig.from_dict() 测试。"""

    def test_from_dict_preserves_values(self) -> None:
        data = {
            "sim_threshold": 0.45,
            "verify_crop_mode": "tail_window",
            "enroll_clean_only": False,
            "enable_multi_template": False,
        }
        cfg = DeepConfig.from_dict(data)
        assert cfg.sim_threshold == 0.45
        assert cfg.verify_crop_mode == "tail_window"
        assert cfg.enroll_clean_only is False
        assert cfg.enable_multi_template is False

    def test_from_dict_converts_noise_list_to_tuple(self) -> None:
        cfg = DeepConfig.from_dict({"noise_injection_snrs": [10, 5, 0]})
        assert cfg.noise_injection_snrs == (10, 5, 0)

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = DeepConfig.from_dict({"unknown_field": 999, "sim_threshold": 0.42})
        assert cfg.sim_threshold == 0.42
        # unknown_field 不会影响构造
        assert not hasattr(cfg, "unknown_field")


class TestDeepConfigFromBestConfig:
    """DeepConfig.from_best_config() 测试。"""

    def test_from_best_config_copies_shared_fields(self) -> None:
        bc = BestConfig(
            sim_threshold=0.55,
            verify_crop_mode="full_utterance",
            verify_buffer_keep_secs=60.0,
            verify_window_secs=1.0,
            enrollment_segment_secs=1.0,
            enable_vad=False,
            vad_rms_threshold=0.002,
            noise_injection_snrs=(20, 15, 10, 5, 0),
            sliding_window_secs=0.6,
            sliding_hop_secs=0.2,
        )
        dc = DeepConfig.from_best_config(bc)
        assert dc.sim_threshold == 0.55
        assert dc.verify_crop_mode == "full_utterance"
        assert dc.verify_buffer_keep_secs == 60.0
        assert dc.noise_injection_snrs == (20, 15, 10, 5, 0)

    def test_from_best_config_uses_deep_defaults_for_new_fields(self) -> None:
        bc = BestConfig(sim_threshold=0.60)
        dc = DeepConfig.from_best_config(bc)
        # spread 字段来自 BestConfig
        assert dc.sim_threshold == 0.60
        # 新增字段使用 DeepConfig 默认值
        assert dc.enroll_skip_vad is True
        assert dc.enroll_clean_only is True
        assert dc.enable_multi_template is True
        assert dc.enable_sliding_window_test is False


# ============================================================================ #
#  工具: 创建已 mock 的 WespeakerDeep 实例
# ============================================================================ #

FAKE_EMBEDDING = F.normalize(torch.randn(256), dim=0).cpu().numpy()


def _make_client_mock() -> MagicMock:
    """创建一个 fake _client，模拟 WespeakerClient 的基本行为。"""
    mock = MagicMock()
    mock.sample_rate = 16000
    mock._model = MagicMock()
    mock._ensure_model = MagicMock(return_value=None)
    return mock


def _make_deep(config: DeepConfig | None = None) -> WespeakerDeep:
    """创建 WespeakerDeep 并注入 fake _client，跳过模型加载。"""
    deep = WespeakerDeep.__new__(WespeakerDeep)
    deep._deep_config = config if config is not None else DeepConfig()
    deep._client = _make_client_mock()
    return deep


# ============================================================================ #
#  WespeakerDeep Tests — 初始化
# ============================================================================ #


class TestWespeakerDeepInit:
    """WespeakerDeep 初始化测试。"""

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
#  WespeakerDeep Tests — _load_raw / load / load_full
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

    def test_load_new_dict_format(self) -> None:
        ref = FAKE_EMBEDDING
        pk_path = "/tmp/test_load_dict.pkl"
        with open(pk_path, "wb") as f:
            pickle.dump({"reference": ref, "version": 1, "templates": []}, f)
        deep = _make_deep()
        result = deep.load(pk_path)
        assert result.shape == (256,)

    def test_load_old_array_format(self) -> None:
        ref = FAKE_EMBEDDING
        pk_path = "/tmp/test_load_array.pkl"
        with open(pk_path, "wb") as f:
            pickle.dump(ref, f)
        deep = _make_deep()
        result = deep.load(pk_path)
        assert result.shape == (256,)


class TestWespeakerDeepLoadFull:
    """load_full() 测试。"""

    def test_load_full_new_format(self) -> None:
        data = {"version": 1, "templates": [FAKE_EMBEDDING], "reference": FAKE_EMBEDDING}
        pk_path = "/tmp/test_load_full_dict.pkl"
        with open(pk_path, "wb") as f:
            pickle.dump(data, f)
        deep = _make_deep()
        result = deep.load_full(pk_path)
        assert result["version"] == 1
        assert len(result["templates"]) == 1

    def test_load_full_old_format_wraps_to_dict(self) -> None:
        old_arr = FAKE_EMBEDDING
        pk_path = "/tmp/test_load_full_old.pkl"
        with open(pk_path, "wb") as f:
            pickle.dump(old_arr, f)
        deep = _make_deep()
        result = deep.load_full(pk_path)
        assert result["version"] == 0
        assert len(result["templates"]) == 1
        np.testing.assert_array_equal(result["reference"], old_arr)


# ============================================================================ #
#  WespeakerDeep Tests — _apply_score_compensation
# ============================================================================ #


class TestApplyScoreCompensation:
    """_apply_score_compensation() 测试。"""

    def test_compensation_disabled_returns_raw_and_factor_1(self) -> None:
        cfg = DeepConfig(enable_score_compensation=False)
        deep = _make_deep(cfg)
        result, factor = deep._apply_score_compensation(0.50, 0.5)
        assert result == 0.50
        assert factor == 1.0

    def test_sqrt_mode_short_audio_boosts_score(self) -> None:
        cfg = DeepConfig(
            enable_score_compensation=True,
            score_compensation_mode="sqrt",
            score_compensation_target_duration=2.0,
        )
        deep = _make_deep(cfg)
        result, factor = deep._apply_score_compensation(0.50, 0.5)
        # duration=0.5, target=2.0, factor = sqrt(2.0/0.5) = 2.0
        assert factor == 2.0
        assert result == 1.0  # clamped to 1.0

    def test_sqrt_mode_long_audio_no_boost(self) -> None:
        cfg = DeepConfig(
            enable_score_compensation=True,
            score_compensation_mode="sqrt",
            score_compensation_target_duration=2.0,
        )
        deep = _make_deep(cfg)
        result, factor = deep._apply_score_compensation(0.80, 4.0)
        # duration=4.0, target=2.0, factor = sqrt(2.0/4.0) = sqrt(0.5) ≈ 0.707 < 1
        # but the formula is: factor = min((target/effective_dur)^0.5, 2.0)
        # so factor = 0.707... no clamping to 1.0 from below
        expected_factor = (2.0 / 4.0) ** 0.5
        assert factor == pytest.approx(expected_factor)
        assert result == pytest.approx(0.80 * expected_factor)

    def test_sqrt_mode_min_effective_duration_03(self) -> None:
        cfg = DeepConfig(
            enable_score_compensation=True,
            score_compensation_mode="sqrt",
            score_compensation_target_duration=2.0,
        )
        deep = _make_deep(cfg)
        # duration=0.1, effective = max(0.1, 0.3) = 0.3
        # factor = min((2.0/0.3)^0.5, 2.0) = min(2.582, 2.0) = 2.0
        result, factor = deep._apply_score_compensation(0.50, 0.1)
        assert factor == 2.0

    def test_linear_mode_uses_get_score_compensation_factor(self) -> None:
        cfg = DeepConfig(
            enable_score_compensation=True,
            score_compensation_mode="linear",
            score_compensation_target_duration=2.0,
        )
        deep = _make_deep(cfg)
        with patch(
            "src.wespeaker_deep_edge.wespeaker_deep_dege.get_score_compensation_factor",
            return_value=1.3,
        ) as mock_gs:
            result, factor = deep._apply_score_compensation(0.50, 1.0)
            assert factor == 1.3
            assert result == pytest.approx(0.65)
            mock_gs.assert_called_once_with(1.0, 2.0)

    def test_none_mode_returns_factor_1(self) -> None:
        cfg = DeepConfig(
            enable_score_compensation=True,
            score_compensation_mode="none",
        )
        deep = _make_deep(cfg)
        result, factor = deep._apply_score_compensation(0.50, 0.5)
        assert factor == 1.0
        assert result == 0.50


# ============================================================================ #
#  WespeakerDeep Tests — enroll()
# ============================================================================ #


class TestWespeakerDeepEnroll:
    """enroll() 测试。"""

    def test_enroll_empty_directory_should_raise(self) -> None:
        deep = _make_deep()
        with pytest.raises(FileNotFoundError, match="注册目录无有效音频文件"):
            deep.enroll("/tmp/empty_deep_enroll_dir")

    def test_enroll_clean_only_ignores_noise_with_warning(self, caplog) -> None:
        """clean_only=True 时，传入 noise_profile 应打 warning 并忽略。"""
        deep = _make_deep()
        # 创建一个有 wav 文件的临时目录
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # 写入一个简单的 WAV 文件（44 字节 header + 数据）
            import struct
            import wave

            wav_path = os.path.join(tmpdir, "test.wav")
            sample_rate = 16000
            duration_secs = 0.5
            n_samples = int(sample_rate * duration_secs)
            audio_data = (np.sin(2 * np.pi * 440 * np.arange(n_samples) / sample_rate) * 0.5).astype(np.float32)

            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            fake_emb = torch.randn(256)

            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.from_numpy(audio_data),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=fake_emb,
            ):
                noise = np.zeros(16000, dtype=np.float32)
                result = deep.enroll(tmpdir, noise_profile=noise, pk_path=os.path.join(tmpdir, "out.pkl"))

            assert result["ok"] is True
            assert result["num_segments"] == 1
            # noise_profile 传入但被忽略，应有 warning
            assert "noise_profile" in caplog.text.lower()

    def test_enroll_happy_path_returns_expected_result(self) -> None:
        import os
        import struct
        import tempfile
        import wave

        deep = _make_deep()

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建 3 个 wav 文件
            for i in range(3):
                wav_path = os.path.join(tmpdir, f"seg_{i}.wav")
                audio_data = np.random.randn(16000).astype(np.float32)
                with wave.open(wav_path, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            pk_path = os.path.join(tmpdir, "voice.pkl")

            fake_emb = torch.randn(256)
            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.randn(16000),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=fake_emb,
            ):
                result = deep.enroll(tmpdir, pk_path=pk_path)

            assert result["ok"] is True
            assert result["num_segments"] == 3
            assert result["num_templates"] == 3
            assert result["embedding_dim"] == 256
            assert os.path.exists(pk_path)

            # 验证 pkl 是新格式
            loaded = deep.load_full(pk_path)
            assert loaded["version"] == 1
            assert len(loaded["templates"]) == 3

    def test_enroll_no_wav_files_finds_any_audio(self) -> None:
        """目录无 .wav 时 fallback 到任意文件。"""
        import os
        import tempfile

        deep = _make_deep()

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建非 .wav 文件
            path = os.path.join(tmpdir, "audio.pcm")
            with open(path, "wb") as f:
                f.write(b"dummy")

            fake_emb = torch.randn(256)
            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.randn(16000),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=fake_emb,
            ):
                result = deep.enroll(tmpdir, pk_path=os.path.join(tmpdir, "out.pkl"))

            assert result["ok"] is True
            assert result["num_segments"] == 1

    def test_enroll_with_vad_calls_silero_vad(self) -> None:
        """enroll_skip_vad=False 时应调用 _apply_silero_vad。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(enroll_skip_vad=False)
        deep = _make_deep(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            audio_data = np.random.randn(16000).astype(np.float32)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            fake_emb = torch.randn(256)
            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.randn(16000),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._apply_silero_vad",
            ) as mock_vad, patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=fake_emb,
            ):
                deep.enroll(tmpdir, pk_path=os.path.join(tmpdir, "out.pkl"))
                mock_vad.assert_called()


# ============================================================================ #
#  WespeakerDeep Tests — recognize()
# ============================================================================ #


class TestWespeakerDeepRecognize:
    """recognize() 测试。"""

    def test_recognize_nonexistent_audio_should_return_error(self) -> None:
        deep = _make_deep()
        result = deep.recognize("/tmp/nonexistent_audio.wav", FAKE_EMBEDDING)
        assert result["is_recognized"] is False
        assert result["confidence"] == 0.0
        assert "文件不存在" in result["error"]

    def test_recognize_with_numpy_voiceprint(self) -> None:
        """直接传入 numpy 数组作为声纹。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(sim_threshold=0.50, enable_score_compensation=False)
        deep = _make_deep(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            audio_data = np.random.randn(32000).astype(np.float32)  # 2s
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            # 返回一个与 voiceprint 相同的 embedding 来保证高相似度
            ref_arr = FAKE_EMBEDDING
            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.from_numpy(audio_data),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=torch.from_numpy(ref_arr),
            ):
                result = deep.recognize(wav_path, ref_arr)

            # 同向量点积 = 1.0
            assert result["is_recognized"] is True
            assert result["confidence"] == pytest.approx(1.0, abs=0.01)
            assert result["num_templates_used"] == 1
            assert result["sliding_windows_used"] == 0

    def test_recognize_with_pkl_path(self) -> None:
        """从 .pkl 文件加载声纹（多模板）。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(sim_threshold=0.50, enable_score_compensation=False)
        deep = _make_deep(cfg)

        # 创建声纹 pkl（新格式：3 个 templates）
        ref = FAKE_EMBEDDING
        pkl_data = {
            "version": 1,
            "templates": [ref, ref, ref],
            "reference": ref,
        }
        vp_path = "/tmp/test_voiceprint.pkl"
        with open(vp_path, "wb") as f:
            pickle.dump(pkl_data, f)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            audio_data = np.random.randn(32000).astype(np.float32)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.from_numpy(audio_data),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=torch.from_numpy(ref),
            ):
                result = deep.recognize(wav_path, vp_path)

            assert result["num_templates_used"] == 3
            assert result["sliding_windows_used"] == 0

    def test_recognize_with_multi_template_disabled(self) -> None:
        """enable_multi_template=False 时应只用 reference。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(enable_multi_template=False, enable_score_compensation=False)
        deep = _make_deep(cfg)

        ref = FAKE_EMBEDDING
        pkl_data = {
            "version": 1,
            "templates": [ref, ref, ref],
            "reference": ref,
        }
        vp_path = "/tmp/test_voiceprint_single.pkl"
        with open(vp_path, "wb") as f:
            pickle.dump(pkl_data, f)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            audio_data = np.random.randn(32000).astype(np.float32)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.from_numpy(audio_data),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=torch.from_numpy(ref),
            ):
                result = deep.recognize(wav_path, vp_path)

            assert result["num_templates_used"] == 1

    def test_recognize_with_sliding_window(self) -> None:
        """短音频 + enable_sliding_window_test=True 应使用滑动窗口。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(
            enable_sliding_window_test=True,
            enable_score_compensation=False,
            short_audio_max_duration=1.5,
            sliding_window_secs=0.4,
            sliding_hop_secs=0.15,
        )
        deep = _make_deep(cfg)

        ref = FAKE_EMBEDDING
        vp_path = "/tmp/test_voiceprint_slide.pkl"
        pkl_data = {
            "version": 1,
            "templates": [ref],
            "reference": ref,
        }
        with open(vp_path, "wb") as f:
            pickle.dump(pkl_data, f)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test_short.wav")
            # 1.0s 音频（> 0.4s 窗口最小长度，< 1.5s 短音频阈值）
            audio_data = np.random.randn(16000).astype(np.float32)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.from_numpy(audio_data),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=torch.from_numpy(ref),
            ):
                result = deep.recognize(wav_path, vp_path)

            # 验证滑动窗口被使用
            assert result["sliding_windows_used"] > 0

    def test_recognize_with_vad_enabled(self) -> None:
        """enable_vad=True 时应调用 VAD 处理。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(enable_vad=True, enable_score_compensation=False)
        deep = _make_deep(cfg)

        ref = FAKE_EMBEDDING
        vp_path = "/tmp/test_vp_vad.pkl"
        pkl_data = {
            "version": 1,
            "templates": [ref],
            "reference": ref,
        }
        with open(vp_path, "wb") as f:
            pickle.dump(pkl_data, f)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test_vad.wav")
            audio_data = np.random.randn(32000).astype(np.float32)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            fake_vad_result = [torch.randn(8000), torch.randn(8000)]

            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.from_numpy(audio_data),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker._vad_segments",
                return_value=fake_vad_result,
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=torch.from_numpy(ref),
            ):
                result = deep.recognize(wav_path, vp_path)

            assert result["is_recognized"] is True

    def test_recognize_with_head_window_crop(self) -> None:
        """verify_crop_mode=head_window 且音频过长时应裁剪头部。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(
            verify_crop_mode="head_window",
            verify_buffer_keep_secs=1.0,
            enable_score_compensation=False,
        )
        deep = _make_deep(cfg)

        ref = FAKE_EMBEDDING

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test_long.wav")
            # 3s 音频（超出 1s 限制）
            audio_data = np.random.randn(48000).astype(np.float32)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            captured_waveform = []

            def fake_load_audio(path, sr):
                t = torch.from_numpy(audio_data)
                return t

            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                side_effect=fake_load_audio,
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
            ) as mock_extract:
                mock_extract.return_value = torch.from_numpy(ref)
                deep.recognize(wav_path, ref)

            # 提取的 embedding 输入应该只有 1s（16000 samples）
            args, _ = mock_extract.call_args
            waveform_arg = args[1]
            assert waveform_arg.numel() == 16000

    def test_recognize_score_compensation_applied(self) -> None:
        """分值补偿应在结果中反映。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(
            enable_score_compensation=True,
            score_compensation_target_duration=2.0,
            score_compensation_mode="sqrt",
        )
        deep = _make_deep(cfg)

        ref = FAKE_EMBEDDING

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test_comp.wav")
            # 1s 音频（短于 2s 目标 → 应补偿）
            audio_data = np.random.randn(16000).astype(np.float32)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.from_numpy(audio_data),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=torch.from_numpy(ref),
            ):
                result = deep.recognize(wav_path, ref)

            # 1s duration，2s target，sqrt mode → factor = sqrt(2/1) ≈ 1.414
            # raw = 1.0 (same vector), compensated = min(1.0 * 1.414, 1.0) = 1.0
            assert result["score_compensation_factor"] > 1.0

    def test_recognize_below_threshold_should_reject(self) -> None:
        """相似度低于阈值时应返回 is_recognized=False。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(sim_threshold=0.99, enable_score_compensation=False)
        deep = _make_deep(cfg)

        ref = FAKE_EMBEDDING
        # 正交向量，点积 ≈ 0
        different_emb = F.normalize(torch.randn(256), dim=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test_low.wav")
            audio_data = np.random.randn(32000).astype(np.float32)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.from_numpy(audio_data),
            ), patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._extract_embedding",
                return_value=different_emb,
            ):
                result = deep.recognize(wav_path, ref)

            assert result["is_recognized"] is False
            assert result["raw_confidence"] < 0.99

    def test_recognize_zero_length_audio_should_return_error(self) -> None:
        """零长度音频应返回错误。"""
        import os
        import tempfile
        import wave

        cfg = DeepConfig(enable_vad=False, enable_score_compensation=False)
        deep = _make_deep(cfg)

        ref = FAKE_EMBEDDING

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test_zero.wav")
            audio_data = np.random.randn(1000).astype(np.float32)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            with patch(
                "src.wespeaker_deep_edge.wespeaker_deep_dege._load_audio",
                return_value=torch.zeros(0),  # 0 samples
            ):
                result = deep.recognize(wav_path, ref)

            assert result["is_recognized"] is False
            assert "音频太短" in result["error"]
