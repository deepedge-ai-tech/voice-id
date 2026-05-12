#!/usr/bin/env python3
"""音频变体生成器 — 为测试音频生成多种效果变体。

生成的变体用于 cross_test.py，模拟不同录制设备和环境场景。

变体类型:
  - eq_phone: 电话音效（频带限制 300Hz-3.5kHz）
  - reverb_hall: 大厅回音（大房间混响）
  - low_bitrate: 低码率效果（降低采样率 + 量化噪声模拟）
  - noise_hiss: 底噪/嘶声（添加背景噪声）

用法:
    uv run python scripts/generate_audio_variants.py
    uv run python scripts/generate_audio_variants.py --dry-run
    uv run python scripts/generate_audio_variants.py --speakers frank john
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pedalboard
from pedalboard import (
    Bitcrush,
    HighpassFilter,
    LowpassFilter,
    Reverb,
)
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter
from pydub.utils import make_chunks

# --------------------------------------------------------------------------- #
#  变体配置
# --------------------------------------------------------------------------- #

VARIANTS = {
    "eq_phone": {
        "name": "电话音效",
        "description": "模拟电话频段 300Hz-3.5kHz",
    },
    "reverb_hall": {
        "name": "大厅回音",
        "description": "大房间混响效果",
    },
    "low_bitrate": {
        "name": "低码率",
        "description": "模拟低码率 MP3 压缩",
    },
    "noise_hiss": {
        "name": "底噪",
        "description": "添加背景嘶声/底噪",
    },
}

SOURCE_FILES = {
    "Frank": "asset/frank/frank 测试.m4a",
    "John": "asset/john/安静环境测试测试.m4a",
    "Michael": "asset/michael/测试.wav",
    "Zhong": "asset/zhong/测试.wav",
    "Xixi": "asset/xixi/测试.wav",
    "Qingqing": "asset/qingqing/测试.wav",
}


# --------------------------------------------------------------------------- #
#  效果处理函数
# --------------------------------------------------------------------------- #


def apply_eq_phone(audio: AudioSegment) -> AudioSegment:
    """应用电话音效 — 频带限制 300Hz-3.5kHz."""
    # 首先削减低频
    audio = high_pass_filter(audio, 300)
    # 然后削减高频
    audio = low_pass_filter(audio, 3500)
    return audio


def apply_eq_phone_pedalboard(audio: np.ndarray, sr: int) -> np.ndarray:
    """使用 pedalboard 应用电话音效 EQ."""
    board = pedalboard.Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=300),
            LowpassFilter(cutoff_frequency_hz=3500),
        ]
    )
    return board(audio, sr)


def apply_reverb_hall(audio: np.ndarray, sr: int) -> np.ndarray:
    """应用大厅回音效果."""
    board = pedalboard.Pedalboard(
        [
            Reverb(
                room_size=0.9,
                damping=0.2,
                wet_level=0.4,
                dry_level=0.6,
                width=1.0,
                freeze_mode=0.0,
            )
        ]
    )
    return board(audio, sr)


def apply_low_bitrate(audio: np.ndarray, sr: int) -> np.ndarray:
    """模拟低码率 MP3 效果 — 降采样 + 量化."""
    # 先降采样到 16kHz
    from scipy.signal import resample

    target_sr = 16000
    ratio = target_sr / sr
    n_samples = int(len(audio) * ratio)
    resampled = resample(audio, n_samples)

    # 使用 bitcrush 模拟量化
    board = pedalboard.Pedalboard(
        [
            Bitcrush(bit_depth=8),  # 8-bit 量化
        ]
    )
    processed = board(resampled, target_sr)

    # 再升采样回原采样率
    n_original = int(len(processed) / ratio)
    return resample(processed, n_original)


def apply_noise_hiss(audio: np.ndarray, sr: int) -> np.ndarray:
    """添加背景底噪/嘶声."""
    # 计算音频的 RMS 动态范围
    rms = np.sqrt(np.mean(audio**2))
    # 噪声水平设为 -40dB
    noise_level = rms * 0.01
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise


# --------------------------------------------------------------------------- #
#  音频处理
# --------------------------------------------------------------------------- #


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """加载音频文件，返回 (audio_array, sample_rate)."""
    audio = AudioSegment.from_file(str(path))
    sr = audio.frame_rate
    # 转换为 numpy 数组 (单声道)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # 归一化到 [-1, 1]
    samples = samples / np.max(np.abs(samples))
    return samples, sr


def save_audio(audio: np.ndarray, sr: int, path: Path) -> None:
    """保存音频文件."""
    # 反归一化并转换为 16-bit PCM
    audio = np.clip(audio, -1, 1)
    audio_int16 = (audio * 32767).astype(np.int16)
    # 使用 pydub 保存
    audio_seg = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=audio_int16.dtype.itemsize,
        channels=1,
    )
    audio_seg.export(str(path), format="mp4")


def generate_variants(source_path: Path, speaker: str, dry_run: bool = False) -> dict[str, Path]:
    """为指定说话人生成所有变体."""
    variants = {}

    # 加载原始音频
    print(f"加载: {source_path}")
    audio, sr = load_audio(source_path)
    print(f"  采样率: {sr} Hz, 时长: {len(audio) / sr:.1f}s")

    base_name = source_path.stem
    output_dir = source_path.parent

    for variant_id, config in VARIANTS.items():
        output_name = f"{base_name}_{variant_id}.m4a"
        output_path = output_dir / output_name

        print(f"\n生成 [{config['name']}]: {output_name}")
        print(f"  {config['description']}")

        if dry_run:
            print(f"  [DRY-RUN] 跳过生成: {output_path}")
            variants[variant_id] = output_path
            continue

        # 应用效果
        processed = audio.copy()
        if variant_id == "eq_phone":
            processed = apply_eq_phone_pedalboard(audio, sr)
        elif variant_id == "reverb_hall":
            processed = apply_reverb_hall(audio, sr)
        elif variant_id == "low_bitrate":
            processed = apply_low_bitrate(audio, sr)
        elif variant_id == "noise_hiss":
            processed = apply_noise_hiss(audio, sr)

        # 保存
        save_audio(processed, sr, output_path)
        print(f"  已保存: {output_path}")
        variants[variant_id] = output_path

    return variants


# --------------------------------------------------------------------------- #
#  CLI 入口
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="音频变体生成器 — 为测试音频生成多种效果变体")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行，不实际生成文件",
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        choices=list(SOURCE_FILES.keys()),
        default=list(SOURCE_FILES.keys()),
        help="要处理的说话人 (默认: 全部)",
    )
    args = parser.parse_args()

    # 验证源文件存在
    for speaker in args.speakers:
        source_path = Path(SOURCE_FILES[speaker])
        if not source_path.exists():
            print(f"错误: 源文件不存在: {source_path}")
            sys.exit(1)

    # 生成变体
    all_variants = {}
    for speaker in args.speakers:
        print(f"\n{'=' * 60}")
        print(f"处理说话人: {speaker}")
        print(f"{'=' * 60}")
        source_path = Path(SOURCE_FILES[speaker])
        variants = generate_variants(source_path, speaker, args.dry_run)
        all_variants[speaker] = variants

    # 总结
    print(f"\n{'=' * 60}")
    print("生成完成")
    print(f"{'=' * 60}")
    for speaker, variants in all_variants.items():
        print(f"\n{speaker}:")
        for variant_id, path in variants.items():
            print(f"  {variant_id}: {path}")


if __name__ == "__main__":
    main()
