#!/usr/bin/env python3
"""AEC 风格音频处理 — 模拟声学回声消除的"失真压制"效果。

处理方法：
  1. 降噪（noisereduce）- 去除背景噪声
  2. 动态压缩（pedalboard）- 压制峰值，提升一致性
  3. 高通滤波 - 去除低频噪声
  4. 间歇性"水下"失真 - 随机丢失高频，产生沉闷感

用法:
    # 处理单个目录
    uv run python scripts/apply_aec_processing.py \\
        --source asset/john_d_usb/test_segments \\
        --target asset/john_d_usb_AEC/test_segments

    # 调整降噪强度
    uv run python scripts/apply_aec_processing.py \\
        --source asset/john_d_usb/test_segments \\
        --target asset/john_d_usb_AEC/test_segments \\
        --noise-reduction 0.8

    # 处理注册片段
    uv run python scripts/apply_aec_processing.py \\
        --source asset/john_d_usb/registration_segments \\
        --target asset/john_d_usb_AEC/registration_segments

    # 启用间歇性"水下"失真（模拟 AEC 高频丢失）
    uv run python scripts/apply_aec_processing.py \\
        --source asset/john_d_usb/test_segments \\
        --target asset/john_d_usb_AEC/test_segments \\
        --distortion-prob 0.2 --distortion-min-ms 150 --distortion-max-ms 600 --lowpass-cutoff 700
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 尝试导入可选库
try:
    import noisereduce as nr

    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    logging.warning("noisereduce 未安装，将跳过降噪步骤")

try:
    from pedalboard import (
        Compressor,
        HighpassFilter,
        LowpassFilter,
        Pedalboard,
    )

    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False
    logging.warning("pedalboard 未安装，将跳过压缩和滤波步骤")


def add_underwater_distortion(
    audio: np.ndarray,
    sample_rate: int,
    distortion_prob: float = 0.1,
    distortion_min_ms: float = 100,
    distortion_max_ms: float = 500,
    lowpass_cutoff: float = 800,
) -> np.ndarray:
    """添加间歇性"水下"失真效果，模拟 AEC 导致的高频丢失。

    产生类似"在水里说话"的沉闷感：
    - 使用低通滤波去除高频
    - 添加轻微的调制效果

    Args:
        audio: 输入音频
        sample_rate: 采样率
        distortion_prob: 每秒发生失真的概率（0-1）
        distortion_min_ms: 最短失真时长（毫秒）
        distortion_max_ms: 最长失真时长（毫秒）
        lowpass_cutoff: 低通滤波截止频率

    Returns:
        添加了间歇性失真的音频
    """
    if distortion_prob <= 0:
        return audio

    from scipy.signal import butter, lfilter

    def lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
        """低通滤波器."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return lfilter(b, a, data)

    audio_length = len(audio)
    samples_per_ms = sample_rate / 1000
    min_distortion_samples = int(distortion_min_ms * samples_per_ms)
    max_distortion_samples = int(distortion_max_ms * samples_per_ms)

    result = audio.copy()
    pos = 0
    window_size = int(sample_rate)  # 每秒检查一次

    while pos < audio_length:
        # 在每个 1 秒窗口内，随机决定是否发生失真
        if np.random.random() < distortion_prob:
            # 随机选择失真时长
            distortion_duration = np.random.randint(
                min_distortion_samples, max_distortion_samples + 1
            )
            end_pos = min(pos + distortion_duration, audio_length)

            # 渐变淡出和淡入，避免突变
            fade_samples = int(0.02 * sample_rate)  # 20ms 淡入淡出
            fade_start = max(pos - fade_samples, 0)
            fade_end = min(end_pos + fade_samples, audio_length)

            # 淡出：过渡到失真段
            if fade_start < pos:
                fade_out_curve = np.linspace(1, 0, pos - fade_start)
                result[fade_start:pos] *= fade_out_curve

            # 失真段：应用低通滤波 + 轻微衰减
            distorted_segment = audio[pos:end_pos].copy()
            # 低通滤波，产生"沉闷"效果
            distorted_segment = lowpass_filter(
                distorted_segment, lowpass_cutoff, sample_rate, order=4
            )
            # 轻微衰减，模拟"被压制"的感觉
            distorted_segment *= 0.7
            result[pos:end_pos] = distorted_segment

            # 淡入：从失真段恢复
            if end_pos < fade_end:
                fade_in_curve = np.linspace(0, 1, min(fade_end - end_pos, fade_samples))
                # 淡入时从低频逐渐恢复到全频
                for i, fade_val in enumerate(fade_in_curve):
                    sample_idx = end_pos + i
                    if sample_idx < audio_length:
                        # 混合原始信号和恢复信号
                        result[sample_idx] = (
                            result[sample_idx] * (1 - fade_val) + audio[sample_idx] * fade_val
                        )

            pos = end_pos + fade_samples
        else:
            pos += window_size

    return result


def apply_aec_processing(
    waveform: torch.Tensor,
    sample_rate: int,
    noise_reduction_strength: float = 0.7,
    compressor_threshold: float = -20.0,
    compressor_ratio: float = 4.0,
    distortion_prob: float = 0.0,
    distortion_min_ms: float = 100,
    distortion_max_ms: float = 500,
    lowpass_cutoff: float = 800,
) -> torch.Tensor:
    """应用 AEC 风格的音频处理.

    Args:
        waveform: 输入音频波形 (channels, samples)
        sample_rate: 采样率
        noise_reduction_strength: 降噪强度 0-1，越高降噪越多
        compressor_threshold: 压缩器阈值 (dB)
        compressor_ratio: 压缩比率
        distortion_prob: 间歇性失真概率（每秒）
        distortion_min_ms: 最短失真时长（毫秒）
        distortion_max_ms: 最长失真时长（毫秒）
        lowpass_cutoff: 低通滤波截止频率

    Returns:
        处理后的音频波形
    """
    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio = waveform.squeeze(0).numpy()

    # 1. 降噪 - 如果可用
    if HAS_NOISEREDUCE and noise_reduction_strength > 0:
        # 使用静态噪声估计（简化方案）
        # 实际应用中可以提取噪声片段作为噪声样本
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=False,  # 非平稳噪声
            prop_decrease=noise_reduction_strength,  # 降噪强度
        )
        audio = reduced_noise

    # 2. 动态范围压缩和滤波 - 如果可用
    if HAS_PEDALBOARD:
        # 构建 pedalboard 处理链
        board = Pedalboard(
            [
                # 高通滤波 - 去除低频噪声
                HighpassFilter(cutoff_frequency_hz=80),
                # 压缩器 - 压制峰值
                Compressor(
                    threshold_db=compressor_threshold,
                    ratio=compressor_ratio,
                    attack_ms=5.0,
                    release_ms=100.0,
                ),
            ]
        )

        # 应用处理
        audio = board(audio, sample_rate)

    # 4. 间歇性"水下"失真（模拟 AEC 导致的高频丢失）
    if distortion_prob > 0:
        audio = add_underwater_distortion(
            audio,
            sample_rate,
            distortion_prob=distortion_prob,
            distortion_min_ms=distortion_min_ms,
            distortion_max_ms=distortion_max_ms,
            lowpass_cutoff=lowpass_cutoff,
        )

    # 归一化防止削波
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.95

    return torch.from_numpy(audio).unsqueeze(0)


def process_directory(
    source_dir: Path,
    target_dir: Path,
    noise_reduction_strength: float = 0.7,
    compressor_threshold: float = -20.0,
    compressor_ratio: float = 4.0,
    distortion_prob: float = 0.0,
    distortion_min_ms: float = 100,
    distortion_max_ms: float = 500,
    lowpass_cutoff: float = 800,
) -> None:
    """处理目录中的所有音频文件.

    Args:
        source_dir: 源目录
        target_dir: 目标目录
        noise_reduction_strength: 降噪强度
        compressor_threshold: 压缩器阈值
        compressor_ratio: 压缩比率
        distortion_prob: 间歇性失真概率
        distortion_min_ms: 最短失真时长
        distortion_max_ms: 最长失真时长
        lowpass_cutoff: 低通滤波截止频率
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # 支持的音频格式
    audio_extensions = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}
    audio_files = [
        f
        for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        logging.warning(f"源目录中没有音频文件: {source_dir}")
        return

    audio_files = sorted(audio_files)

    logging.info(f"找到 {len(audio_files)} 个音频文件")
    logging.info(f"处理参数:")
    logging.info(f"  - 降噪强度: {noise_reduction_strength}")
    logging.info(f"  - 压缩阈值: {compressor_threshold} dB")
    logging.info(f"  - 压缩比率: {compressor_ratio}:1")
    if distortion_prob > 0:
        logging.info(f"  - 间歇性失真: {distortion_prob * 100:.0f}% 概率/秒")
        logging.info(f"  - 失真时长: {distortion_min_ms}-{distortion_max_ms} ms")
        logging.info(f"  - 低通截止: {lowpass_cutoff} Hz")

    for i, audio_file in enumerate(audio_files, 1):
        target_file = target_dir / audio_file.name

        logging.info(f"[{i}/{len(audio_files)}] 处理: {audio_file.name}")

        try:
            # 加载音频
            waveform, sr = torchaudio.load(str(audio_file))

            # 应用处理
            processed = apply_aec_processing(
                waveform,
                sr,
                noise_reduction_strength=noise_reduction_strength,
                compressor_threshold=compressor_threshold,
                compressor_ratio=compressor_ratio,
                distortion_prob=distortion_prob,
                distortion_min_ms=distortion_min_ms,
                distortion_max_ms=distortion_max_ms,
                lowpass_cutoff=lowpass_cutoff,
            )

            # 保存结果（始终保存为 WAV）
            output_file = target_file.with_suffix(".wav")
            torchaudio.save(str(output_file), processed, sr)

            logging.info(f"  → 保存到: {output_file.name}")

        except Exception as e:
            logging.error(f"  × 处理失败: {e}")

    logging.info(f"\n完成! 处理了 {len(audio_files)} 个文件")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AEC 风格音频处理 — 模拟声学回声消除的失真压制效果"
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="源音频目录",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="目标输出目录",
    )
    parser.add_argument(
        "--noise-reduction",
        "-n",
        type=float,
        default=0.7,
        help="降噪强度 0-1 (default: 0.7)",
    )
    parser.add_argument(
        "--compressor-threshold",
        "-c",
        type=float,
        default=-20.0,
        help="压缩器阈值 dB (default: -20.0)",
    )
    parser.add_argument(
        "--compressor-ratio",
        "-r",
        type=float,
        default=4.0,
        help="压缩比率 (default: 4.0)",
    )
    parser.add_argument(
        "--distortion-prob",
        "-d",
        type=float,
        default=0.15,
        help="间歇性失真概率（每秒，0-1）(default: 0.15)",
    )
    parser.add_argument(
        "--distortion-min-ms",
        type=float,
        default=100,
        help="最短失真时长（毫秒）(default: 100)",
    )
    parser.add_argument(
        "--distortion-max-ms",
        type=float,
        default=500,
        help="最长失真时长（毫秒）(default: 500)",
    )
    parser.add_argument(
        "--lowpass-cutoff",
        type=float,
        default=800,
        help="低通滤波截止频率 Hz (default: 800)",
    )

    args = parser.parse_args()

    process_directory(
        source_dir=Path(args.source),
        target_dir=Path(args.target),
        noise_reduction_strength=args.noise_reduction,
        compressor_threshold=args.compressor_threshold,
        compressor_ratio=args.compressor_ratio,
        distortion_prob=args.distortion_prob,
        distortion_min_ms=args.distortion_min_ms,
        distortion_max_ms=args.distortion_max_ms,
        lowpass_cutoff=args.lowpass_cutoff,
    )


if __name__ == "__main__":
    main()
