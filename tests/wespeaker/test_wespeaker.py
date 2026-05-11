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
        import soundfile as sf
        import numpy as np
        sf.write("/tmp/test_audio.wav", np.zeros(16000), 16000)
        result = client.recognize("/tmp/test_audio.wav", "/nonexistent.pkl")
        assert result["is_recognized"] is False
        assert "声纹文件不存在" in result["error"]
