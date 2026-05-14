"""共享 fixtures。"""

import pytest

from src.wespeaker_deep_edge.wespeaker import WespeakerClient


@pytest.fixture
def client():
    """创建 WespeakerClient 实例（CPU 模式，禁用增强）。"""
    return WespeakerClient(device="cpu", enable_augmentation=False)
