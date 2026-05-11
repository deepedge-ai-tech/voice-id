"""WeSpeaker 核心功能测试。"""

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.wespeaker.wespeaker import (
    WespeakerClient,
    _crop_verify,
    _extract_embedding,
    _load_audio,
)


class TestLoadAudio:
    """测试音频加载功能。"""

    def test_load_audio_with_invalid_path_should_fail(self):
        """测试：无效路径应该抛出异常。"""
        with pytest.raises(RuntimeError, match="无法加载音频"):
            _load_audio("/nonexistent/path.wav")


class TestCropVerify:
    """测试裁剪策略。"""

    def test_crop_verify_full_utterance_should_return_all(self):
        """测试：full_utterance 模式应返回完整波形。"""
        waveform = torch.randn(160000)  # 10s
        result = _crop_verify(waveform, "full_utterance", 1.0, 16000)
        assert result.numel() == 160000

    def test_crop_verify_tail_window_should_return_tail(self):
        """测试：tail_window 模式应返回尾部窗口。"""
        waveform = torch.arange(160000)  # 10s
        result = _crop_verify(waveform, "tail_window", 1.0, 16000)
        assert result[0].item() == 144000  # 160000 - 16000

    def test_crop_verify_head_window_should_return_head(self):
        """测试：head_window 模式应返回头部窗口。"""
        waveform = torch.arange(160000)
        result = _crop_verify(waveform, "head_window", 1.0, 16000)
        assert result[0].item() == 0

    def test_crop_verify_short_audio_should_return_all(self):
        """测试：短音频应返回全部内容。"""
        waveform = torch.randn(8000)  # 0.5s
        result = _crop_verify(waveform, "tail_window", 1.0, 16000)
        assert result.numel() == 8000


class TestWespeakerClient:
    """测试 WespeakerClient 核心类。"""

    def test_enroll_with_nonexistent_file_should_fail(self, client):
        """测试：不存在的文件应该返回错误。"""
        result = client.mp3_to_pk("/nonexistent.wav", "/tmp/test.pkl")
        assert result["ok"] is False
        assert "文件不存在" in result["error"]

    def test_recognize_with_nonexistent_audio_should_fail(self, client):
        """测试：不存在的测试音频应该返回错误。"""
        result = client.recognize("/nonexistent.wav", "/tmp/voice.pkl")
        assert result["is_recognized"] is False
        assert "文件不存在" in result["error"]

    def test_recognize_with_nonexistent_voiceprint_should_fail(self, client):
        """测试：不存在的声纹文件应该返回错误。"""
        # Create a temp audio file first
        import numpy as np
        import soundfile as sf

        sf.write("/tmp/test_audio.wav", np.zeros(16000), 16000)
        result = client.recognize("/tmp/test_audio.wav", "/nonexistent.pkl")
        assert result["is_recognized"] is False
        assert "声纹文件不存在" in result["error"]


class TestVerifyBufferKeepSecs:
    """测试 verify_buffer_keep_secs 截断逻辑。"""

    def test_default_verify_buffer_keep_secs_should_be_8(self):
        """测试：默认 buffer 保留时长应为 8.0 秒。"""
        client = WespeakerClient()
        assert client.verify_buffer_keep_secs == 8.0

    def test_default_verify_crop_mode_should_be_full_utterance(self):
        """测试：默认裁剪模式应为 full_utterance。"""
        client = WespeakerClient()
        assert client.verify_crop_mode == "full_utterance"

    def test_default_sim_threshold_should_be_055(self):
        """测试：默认相似度阈值应为 0.55。"""
        client = WespeakerClient()
        assert client.sim_threshold == 0.55


class TestEnrollMixed:
    """测试混合注册功能。"""

    def test_enroll_mixed_with_nonexistent_files_should_fail(self, client):
        """测试：所有文件都不存在时应返回错误。"""
        result = client.enroll_mixed([], [], "/tmp/mixed_test.pkl")
        assert result["ok"] is False
        assert "无有效音频片段" in result["error"]

    def test_enroll_mixed_creates_valid_pickle(self, client):
        """测试：混合注册应生成有效的 pickle 文件。"""
        import os

        import numpy as np
        import soundfile as sf

        # Create test audio files
        sf.write("/tmp/clean_test.wav", np.zeros(16000), 16000)  # 1s
        sf.write("/tmp/noisy_test.wav", np.zeros(32000), 16000)  # 2s

        if os.path.exists("/tmp/mixed_test.pkl"):
            os.remove("/tmp/mixed_test.pkl")

        result = client.enroll_mixed(
            ["/tmp/clean_test.wav"], ["/tmp/noisy_test.wav"], "/tmp/mixed_test.pkl"
        )
        assert result["ok"] is True
        assert result["num_segments"] == 3  # 1 + 2 segments
        assert os.path.exists("/tmp/mixed_test.pkl")

        # Verify the pickle file contains a valid embedding
        with open("/tmp/mixed_test.pkl", "rb") as f:
            emb = np.asarray(pickle.load(f))
        assert emb.shape == (256,)  # 256-dimensional embedding
