#!/usr/bin/env python3
"""
将注册音频按静音间隔切分为多个片段。

用法:
    python split_registration.py asset/john/注册.aif output_dir --speaker S01 --scene clean
"""

import argparse
import os
import struct
from pathlib import Path

import numpy as np
import soundfile as sf


def find_silence_gaps(samples: np.ndarray, sample_rate: int,
                      rms_threshold: float = 0.002,
                      min_gap_duration: float = 0.3,
                      window_duration: float = 0.1) -> list[tuple[float, float]]:
    """
    基于 RMS 能量检测静音间隔。

    Returns:
        list of (start_time, end_time) tuples for each gap
    """
    window_size = int(window_duration * sample_rate)
    n_windows = len(samples) // window_size

    # 计算每个窗口的 RMS
    rms_values = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        rms = np.sqrt(np.mean(samples[start:end] ** 2))
        rms_values.append(rms)

    # 检测连续低能量窗口
    gaps = []
    silent_start = None

    for i, rms in enumerate(rms_values):
        if rms < rms_threshold:
            if silent_start is None:
                silent_start = i
        else:
            if silent_start is not None:
                gap_start = silent_start * window_duration
                gap_end = i * window_duration
                if gap_end - gap_start >= min_gap_duration:
                    gaps.append((gap_start, gap_end))
                silent_start = None

    # 处理结尾的静音
    if silent_start is not None:
        gap_start = silent_start * window_duration
        gap_end = n_windows * window_duration
        if gap_end - gap_start >= min_gap_duration:
            gaps.append((gap_start, gap_end))

    return gaps


def split_by_gaps(samples: np.ndarray, sample_rate: int,
                  gaps: list[tuple[float, float]],
                  output_dir: str, speaker_id: str, scene: str,
                  content_type: str = "free") -> list[str]:
    """
    按静音间隔切分音频，并按规范命名保存。

    命名格式: {speakerID}_{scene}_{segmentID}_{type}.wav
    例如: S01_clean_01_free.wav
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 计算片段边界
    segments = []
    prev_end = 0
    for gap_start, gap_end in gaps:
        if gap_start > prev_end:
            segments.append((prev_end, gap_start))
        prev_end = gap_end

    # 添加最后一段
    total_duration = len(samples) / sample_rate
    if prev_end < total_duration:
        segments.append((prev_end, total_duration))

    print(f"检测到 {len(gaps)} 个静音间隔，可切分为 {len(segments)} 个片段")

    saved_files = []
    for idx, (start_time, end_time) in enumerate(segments, 1):
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # 提取片段
        segment = samples[start_sample:end_sample]

        # 跳过太短的片段（< 0.5s）
        duration = end_time - start_time
        if duration < 0.5:
            print(f"  跳过片段 {idx}: {duration:.2f}s (太短)")
            continue

        # 生成文件名
        segment_id = f"{idx:02d}"
        filename = f"{speaker_id}_{scene}_{segment_id}_{content_type}.wav"
        filepath = output_path / filename

        # 保存为 16kHz 单声道 WAV（符合 WeSpeaker 要求）
        if segment.ndim == 2:
            segment = segment.mean(axis=1)

        # 重采样到 16kHz（如果需要）
        if sample_rate != 16000:
            # 简单重采样（对于语音足够）
            ratio = 16000 / sample_rate
            new_length = int(len(segment) * ratio)
            indices = np.linspace(0, len(segment) - 1, new_length)
            segment = np.interp(indices, np.arange(len(segment)), segment)
            out_sr = 16000
        else:
            out_sr = sample_rate

        sf.write(filepath, segment, out_sr, subtype='PCM_16')

        print(f"  [{idx:02d}] {start_time:6.2f}s - {end_time:6.2f}s ({duration:5.2f}s) → {filename}")
        saved_files.append(str(filepath))

    return saved_files


def main():
    parser = argparse.ArgumentParser(description="按静音间隔切分注册音频")
    parser.add_argument("input_audio", help="输入音频文件路径")
    parser.add_argument("output_dir", default="asset/john/registration_segments", help="输出目录")
    parser.add_argument("--speaker", default="S01", help="说话人 ID (默认 S01)")
    parser.add_argument("--scene", default="clean",
                       choices=["clean", "aec_distorted", "cross_talk"],
                       help="场景标签 (默认 clean)")
    parser.add_argument("--type", dest="content_type", default="free",
                       choices=["fixed", "free"],
                       help="内容类型 (默认 free)")
    parser.add_argument("--threshold", type=float, default=0.002,
                       help="RMS 静音阈值 (默认 0.002)")
    parser.add_argument("--min-gap", type=float, default=0.3,
                       help="最小静音间隔时长，单位秒 (默认 0.3)")
    parser.add_argument("--expected-segments", type=int, default=None,
                       help="期望的片段数量（用于验证）")

    args = parser.parse_args()

    # 加载音频
    print(f"加载音频: {args.input_audio}")
    samples, sample_rate = sf.read(args.input_audio)
    duration = len(samples) / sample_rate
    print(f"  采样率: {sample_rate} Hz")
    print(f"  声道数: {samples.ndim if samples.ndim > 1 else 1}")
    print(f"  时长: {duration:.2f}s")
    print()

    # 检测静音间隔
    print("检测静音间隔...")
    gaps = find_silence_gaps(
        samples, sample_rate,
        rms_threshold=args.threshold,
        min_gap_duration=args.min_gap
    )

    if not gaps:
        print("未检测到足够的静音间隔，尝试降低阈值...")
        for t in [0.001, 0.0005, 0.0002]:
            gaps = find_silence_gaps(
                samples, sample_rate,
                rms_threshold=t,
                min_gap_duration=args.min_gap
            )
            if gaps:
                print(f"使用阈值 {t} 检测到 {len(gaps)} 个间隔")
                break

    # 切分并保存
    print(f"\n切分音频到: {args.output_dir}")
    saved_files = split_by_gaps(
        samples, sample_rate, gaps,
        args.output_dir, args.speaker, args.scene, args.content_type
    )

    print(f"\n完成! 共保存 {len(saved_files)} 个文件")

    # 验证片段数量
    if args.expected_segments:
        if len(saved_files) == args.expected_segments:
            print(f"✅ 片段数量匹配预期 ({args.expected_segments})")
        else:
            print(f"⚠️  片段数量不匹配: 实际 {len(saved_files)}, 预期 {args.expected_segments}")


if __name__ == "__main__":
    main()
