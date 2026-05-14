#!/usr/bin/env python3
"""Voice-ID 声纹识别自动研究 - 数据准备与工具模块。

此文件包含:
  - 固定常量定义
  - 说话人配置 (SPEAKERS, SAME_PERSON_GROUPS)
  - 工具函数（不修改）

设计原则:
  - 此文件不应该被 AI 代理修改
  - training.py 是唯一被代理修改的文件

注册方式:
  - 使用合并音频注册 — 将每个说话人的所有注册片段合并成一条音频进行注册
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================ #
#  固定常量
# ============================================================================ #

PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = PROJECT_ROOT / "asset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUT_ROOT / "experiments"
BEST_CONFIG_PATH = OUTPUT_ROOT / "best_config.json"
EXPERIMENT_LOG_PATH = OUTPUT_ROOT / "experiment_log.json"

# 时间预算（秒）- 每次实验的固定运行时间
TIME_BUDGET_SECONDS = 600  # 10 分钟

# 音频处理常量
SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 0.3  # 最小音频时长（秒）
MAX_AUDIO_DURATION = 60.0  # 最大音频时长（秒）

# 短音频判定阈值
# 启用 VAD 时：0.6s（VAD 会裁掉静音，有效音频更短）
# 禁用 VAD 时：1.5s（原始音频长度，需包含短音频场景）
SHORT_AUDIO_DURATION_VAD = 0.6
SHORT_AUDIO_DURATION_NO_VAD = 1.5

# 模型配置
DEFAULT_MODEL_PATH = "pyannote/wespeaker-voxceleb-resnet34-LM"
DEFAULT_DEVICE = "cpu"

# 评估指标
METRICS_NAMES = [
    "far",  # False Accept Rate
    "frr",  # False Reject Rate
    "short_audio_confidence",  # < 0.6s 音频平均置信度
    "overall_accuracy",  # 总体准确率
    "eer",  # Equal Error Rate
    "avg_true_positive_score",  # 正确匹配平均分数
    "avg_true_negative_score",  # 正确拒绝平均分数
]

# ============================================================================ #
#  说话人配置（固定数据，不修改）
# ============================================================================ #

SPEAKERS = {
    "John": {
        "register_dir": "asset/john/registration_segments",
        "test_segments_dir": "asset/john/test_segments",
    },
    "John_MeetingRoom": {
        "register_dir": "asset/john_metting_room/registration_segments",
        "test_segments_dir": "asset/john_metting_room/test_segments",
    },
    "John_D_USB": {
        "register_dir": "asset/john_d_usb/registration_segments",
        "test_segments_dir": "asset/john_d_usb/test_segments",
    },
    "John_D_USB_AEC": {
        "register_dir": "asset/john_d_usb_AEC/registration_segments",
        "test_segments_dir": "asset/john_d_usb_AEC/test_segments",
    },
    "Xixi": {
        "register_dir": "asset/xixi/registration_segments",
        "test_segments_dir": "asset/xixi/test_segments",
    },
    "Frank": {
        "register_dir": "asset/frank/registration_segments",
        "test_segments_dir": "asset/frank/test_segments",
    },
    "Qingqing": {
        "register_dir": "asset/qingqing/registration_segments",
        "test_segments_dir": "asset/qingqing/test_segments",
    },
    "Zhong": {
        "register_dir": "asset/zhong/registration_segments",
        "test_segments_dir": "asset/zhong/test_segments",
    },
    "Zhong_D_USB": {
        "register_dir": "asset/zhong_d_usb/registration_segments",
        "test_segments_dir": "asset/zhong_d_usb/test_segments",
    },
}

# 同一人组映射（用于评估）
SAME_PERSON_GROUPS: dict[str, set[str]] = {
    "John": {"John", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_MeetingRoom": {"John", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_D_USB": {"John", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_D_USB_AEC": {"John", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "Zhong": {"Zhong", "Zhong_D_USB"},
    "Zhong_D_USB": {"Zhong", "Zhong_D_USB"},
}

SPEAKER_ORDER = [
    "John",
    "John_MeetingRoom",
    "John_D_USB",
    "John_D_USB_AEC",
    "Zhong",
    "Zhong_D_USB",
    "Xixi",
    "Frank",
    "Qingqing",
]

# ============================================================================ #
#  工具函数
# ============================================================================ #


def is_same_person(speaker1: str, speaker2: str) -> bool:
    """判断两个说话人是否属于同一人（考虑同一人的不同录制条件）."""
    if speaker1 == speaker2:
        return True
    group1 = SAME_PERSON_GROUPS.get(speaker1, {speaker1})
    return speaker2 in group1


def setup_output_dirs() -> None:
    """创建输出目录结构."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentMetrics:
    """实验评估指标."""

    far: float = 0.0  # False Accept Rate = false_accepts / total_negatives
    frr: float = 0.0  # False Reject Rate = false_rejects / total_positives
    short_audio_confidence: float = 0.0  # 短音频平均置信度（所有样本）：VAD启用时<0.6s，禁用时<1.5s
    short_audio_genuine_confidence: float = 0.0  # 短音频同人平均置信度
    overall_accuracy: float = 0.0  # (TP + TN) / total
    eer: float = 0.0  # Equal Error Rate
    avg_true_positive_score: float = 0.0  # 正确匹配平均分数
    avg_true_negative_score: float = 0.0  # 正确拒绝平均分数（越低越好）

    # 总置信度（所有样本平均）
    total_avg_confidence: float = 0.0
    genuine_avg_confidence: float = 0.0  # 同人匹配平均置信度

    # 统计信息
    total_tests: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    short_audio_count: int = 0

    def to_dict(self) -> dict:
        """转换为字典."""
        return {
            "far": self.far,
            "frr": self.frr,
            "short_audio_confidence": self.short_audio_confidence,
            "short_audio_genuine_confidence": self.short_audio_genuine_confidence,
            "overall_accuracy": self.overall_accuracy,
            "eer": self.eer,
            "avg_true_positive_score": self.avg_true_positive_score,
            "avg_true_negative_score": self.avg_true_negative_score,
            "total_avg_confidence": self.total_avg_confidence,
            "genuine_avg_confidence": self.genuine_avg_confidence,
            "total_tests": self.total_tests,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "short_audio_count": self.short_audio_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentMetrics":
        """从字典创建指标."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_results(cls, results: list[dict], threshold: float, short_audio_duration: float = 0.6) -> "ExperimentMetrics":
        """从测试结果计算指标.

        Args:
            results: 测试用例列表
            threshold: 相似度阈值
            short_audio_duration: 短音频判定阈值（秒），启用 VAD 默认 0.6，禁用 VAD 建议 1.5
        """
        metrics = cls()

        total_positives = 0
        total_negatives = 0
        short_scores = []
        short_genuine_scores = []

        for r in results:
            test_speaker = r.get("test_speaker", "")
            ref_speaker = r.get("ref_speaker", "")
            score = r.get("score", 0.0)
            vad_duration = r.get("vad_duration", 0.0)
            is_match = r.get("is_match", False)

            metrics.total_tests += 1

            # 收集短音频分数
            if vad_duration < short_audio_duration:
                short_scores.append(score)
                metrics.short_audio_count += 1
                if is_same_person(test_speaker, ref_speaker):
                    short_genuine_scores.append(score)

            # 判断是正样本还是负样本
            same_person = is_same_person(test_speaker, ref_speaker)

            if same_person:
                total_positives += 1
                if is_match:
                    metrics.true_positives += 1
                else:
                    metrics.false_negatives += 1
            else:
                total_negatives += 1
                if is_match:
                    metrics.false_positives += 1
                else:
                    metrics.true_negatives += 1

        # 计算各项指标
        if total_positives > 0:
            metrics.frr = metrics.false_negatives / total_positives
        if total_negatives > 0:
            metrics.far = metrics.false_positives / total_negatives
        if metrics.total_tests > 0:
            metrics.overall_accuracy = (metrics.true_positives + metrics.true_negatives) / metrics.total_tests

        # 短音频平均置信度
        if short_scores:
            metrics.short_audio_confidence = np.mean(short_scores)
        if short_genuine_scores:
            metrics.short_audio_genuine_confidence = np.mean(short_genuine_scores)

        # EER 简化估计（FAR 和 FRR 的平均值）
        metrics.eer = (metrics.far + metrics.frr) / 2

        return metrics


@dataclass
class ExperimentConfig:
    """实验配置."""

    # 识别参数
    sim_threshold: float = 0.55
    verify_crop_mode: str = "full_utterance"  # full_utterance / tail_window / head_window
    verify_buffer_keep_secs: float = 8.0
    verify_window_secs: float = 1.0

    # 注册参数
    enrollment_segment_secs: float = 1.0

    # VAD 参数
    enable_vad: bool = True
    vad_rms_threshold: float = 0.005

    # 分数补偿（固定参数，不修改）
    enable_score_compensation: bool = False
    score_compensation_target_duration: float = 2.0

    # 噪声注入
    noise_injection_snrs: list[float] = field(default_factory=lambda: [20.0, 15.0, 10.0, 5.0, 0.0])
    noise_path: str = "asset/john/嘈杂环境测试.m4a"

    # 滑动窗口
    enable_sliding_window: bool = True
    sliding_window_secs: float = 0.6
    sliding_hop_secs: float = 0.2

    # 输出控制
    verbose: bool = False
    debug: bool = False

    def to_dict(self) -> dict:
        """转换为字典."""
        return {
            "sim_threshold": self.sim_threshold,
            "verify_crop_mode": self.verify_crop_mode,
            "verify_buffer_keep_secs": self.verify_buffer_keep_secs,
            "verify_window_secs": self.verify_window_secs,
            "enrollment_segment_secs": self.enrollment_segment_secs,
            "enable_vad": self.enable_vad,
            "vad_rms_threshold": self.vad_rms_threshold,
            "enable_score_compensation": self.enable_score_compensation,
            "score_compensation_target_duration": self.score_compensation_target_duration,
            "noise_injection_snrs": self.noise_injection_snrs,
            "noise_path": self.noise_path,
            "enable_sliding_window": self.enable_sliding_window,
            "sliding_window_secs": self.sliding_window_secs,
            "sliding_hop_secs": self.sliding_hop_secs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """从字典创建配置."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        """保存配置到 JSON 文件."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """从 JSON 文件加载配置."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ExperimentResult:
    """单次实验结果."""

    experiment_id: str
    timestamp: str
    config: ExperimentConfig
    metrics: ExperimentMetrics
    scores_matrix: Optional[list[list[float]]] = None
    row_labels: Optional[list[str]] = None

    def to_dict(self) -> dict:
        """转换为字典."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


def load_experiment_log() -> list[dict]:
    """加载实验历史记录."""
    if EXPERIMENT_LOG_PATH.exists():
        with open(EXPERIMENT_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_experiment_log(log: list[dict]) -> None:
    """保存实验历史记录."""
    with open(EXPERIMENT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def load_best_config() -> Optional[ExperimentConfig]:
    """加载当前最佳配置."""
    if BEST_CONFIG_PATH.exists():
        try:
            return ExperimentConfig.load(BEST_CONFIG_PATH)
        except Exception:
            return None
    return None


def save_best_config(config: ExperimentConfig, metrics: ExperimentMetrics) -> None:
    """保存最佳配置."""
    data = {
        "config": config.to_dict(),
        "metrics": metrics.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_improvement(old_metrics: ExperimentMetrics, new_metrics: ExperimentMetrics) -> dict[str, float]:
    """计算指标改善情况."""
    return {
        "far_delta": old_metrics.far - new_metrics.far,  # 正值表示改善
        "frr_delta": old_metrics.frr - new_metrics.frr,
        "short_audio_conf_delta": new_metrics.short_audio_confidence - old_metrics.short_audio_confidence,
        "accuracy_delta": new_metrics.overall_accuracy - old_metrics.overall_accuracy,
        "eer_delta": old_metrics.eer - new_metrics.eer,
    }


def format_improvement_summary(improvement: dict) -> str:
    """格式化改善摘要."""
    lines = ["指标变化:"]
    for key, value in improvement.items():
        symbol = "↑" if value > 0 else "↓" if value < 0 else "="
        lines.append(f"  {key}: {value:+.4f} {symbol}")
    return "\n".join(lines)


# ============================================================================ #
#  数据验证
# ============================================================================ #


def validate_speaker_data() -> bool:
    """验证说话人数据目录是否存在."""
    all_valid = True
    for name, paths in SPEAKERS.items():
        reg_dir = Path(paths["register_dir"])
        test_dir = Path(paths["test_segments_dir"])

        if not reg_dir.is_dir():
            logger.warning(f"注册目录不存在: {reg_dir}")
            all_valid = False
        if not test_dir.is_dir():
            logger.warning(f"测试目录不存在: {test_dir}")
            all_valid = False
        else:
            wav_files = list(test_dir.glob("*.wav"))
            if not wav_files:
                logger.warning(f"测试目录中没有 .wav 文件: {test_dir}")
                all_valid = False

    return all_valid


def validate_noise_file(noise_path: str) -> bool:
    """验证噪声音频文件是否存在."""
    path = Path(noise_path)
    if not path.is_file():
        logger.warning(f"噪声音频不存在: {noise_path}")
        return False
    return True


# ============================================================================ #
#  主函数 - 一次性数据准备
# ============================================================================ #


def main() -> None:
    """执行数据准备和验证."""
    print("=" * 60)
    print("Voice-ID 数据准备")
    print("=" * 60)

    # 创建输出目录
    setup_output_dirs()
    print(f"✓ 输出目录已创建: {OUTPUT_ROOT}")

    # 验证说话人数据
    print("\n验证说话人数据...")
    speakers_valid = validate_speaker_data()
    if speakers_valid:
        print("✓ 所有说话人数据目录验证通过")
    else:
        print("⚠ 部分说话人数据目录缺失，请检查")

    # 验证噪声音频
    print("\n验证噪声音频...")
    noise_valid = validate_noise_file("asset/john/嘈杂环境测试.m4a")
    if noise_valid:
        print("✓ 噪声音频验证通过")
    else:
        print("⚠ 噪声音频文件不存在")

    # 创建初始配置
    print("\n创建默认配置...")
    default_config = ExperimentConfig()
    if not BEST_CONFIG_PATH.exists():
        save_best_config(default_config, ExperimentMetrics())
        print(f"✓ 默认配置已创建: {BEST_CONFIG_PATH}")

    # 初始化实验日志
    if not EXPERIMENT_LOG_PATH.exists():
        save_experiment_log([])
        print(f"✓ 实验日志已初始化: {EXPERIMENT_LOG_PATH}")

    print("\n" + "=" * 60)
    print("数据准备完成！现在可以运行实验:")
    print("  uv run training.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
