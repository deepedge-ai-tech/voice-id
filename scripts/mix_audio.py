#!/usr/bin/env python3
"""音频混合工具 - 将两个音频文件按指定比例混合。

用法:
    python mix_audio.py target.wav other.wav output.wav --ratio 0.5
    python mix_audio.py target.wav other.wav output.wav --ratio 1.0  # 等音量
    python mix_audio.py target.wav other.wav output.wav --ratio 0.2  # other音量20%
"""

import argparse
import logging
from pathlib import Path

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """加载音频并重采样到目标采样率.

    Args:
        path: 音频文件路径
        target_sr: 目标采样率

    Returns:
        (channels, samples) 形状的音频张量
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")

    waveform, sr = torchaudio.load(str(path))

    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 重采样到目标采样率
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform


def mix_audios(
    target_path: str,
    other_path: str,
    output_path: str,
    other_ratio: float = 0.5,
    target_sr: int = 16000,
) -> None:
    """混合两个音频文件.

    Args:
        target_path: 主说话人音频路径（保持原始音量）
        other_path: 旁人音频路径
        output_path: 输出音频路径
        other_ratio: 旁人音量比例（相对于主说话人）
        target_sr: 目标采样率
    """
    logger.info(f"加载主说话人音频: {target_path}")
    target_waveform = load_audio(target_path, target_sr)

    logger.info(f"加载旁人音频: {other_path}")
    other_waveform = load_audio(other_path, target_sr)

    # 获取较短的长度
    min_length = min(target_waveform.shape[1], other_waveform.shape[1])
    target_waveform = target_waveform[:, :min_length]
    other_waveform = other_waveform[:, :min_length]

    logger.info(f"音频长度: {min_length / target_sr:.2f}s")
    logger.info(f"主说话人 RMS: {target_waveform.norm().item():.4f}")
    logger.info(f"旁人 RMS: {other_waveform.norm().item():.4f}")

    # 混合：主说话人保持原音量，旁人按比例缩放
    mixed_waveform = target_waveform + other_waveform * other_ratio

    # 归一化防止削波
    max_val = mixed_waveform.abs().max().item()
    if max_val > 0.95:
        mixed_waveform = mixed_waveform * (0.95 / max_val)
        logger.info(f"归一化防止削波: {max_val:.4f} → 0.95")

    logger.info(f"混合后 RMS: {mixed_waveform.norm().item():.4f}")
    logger.info(f"旁人音量比例: {other_ratio:.0%}")

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), mixed_waveform, target_sr)
    logger.info(f"✅ 混合音频已保存: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="音频混合工具")
    parser.add_argument("target", help="主说话人音频路径")
    parser.add_argument("other", help="旁人音频路径")
    parser.add_argument("output", help="输出音频路径")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="旁人音量比例（相对于主说话人）(default: 0.5)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="目标采样率 (default: 16000)",
    )
    args = parser.parse_args()

    mix_audios(args.target, args.other, args.output, args.ratio, args.sr)


if __name__ == "__main__":
    main()
