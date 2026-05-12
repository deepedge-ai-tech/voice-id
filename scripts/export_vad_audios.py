#!/usr/bin/env python3
"""导出误拒绝案例的 VAD 处理前后音频对比.

用法:
    uv run python scripts/export_vad_audios.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchaudio as ta

from src.wespeaker.wespeaker import _apply_silero_vad, _load_audio, _vad_segments

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 误拒绝案例列表（来自实验报告）
FALSE_NEGATIVES = [
    # John
    ("John", "test_segment_006_3s.wav", 0.5056, 0.0444, 2.47, 1.34, 0.54),
    ("John", "test_segment_010_2s.wav", 0.4576, 0.0924, 1.91, 1.27, 0.66),
    ("John", "test_segment_011_1s.wav", 0.2156, 0.3344, 1.34, 0.38, 0.28),
    ("John", "test_segment_012_1s.wav", 0.1556, 0.3944, 1.12, 0.45, 0.40),
    ("John", "test_segment_013_1s.wav", 0.4672, 0.0828, 1.50, 0.57, 0.38),
    ("John", "test_segment_014_1s.wav", 0.4852, 0.0648, 1.38, 0.71, 0.51),
    ("John", "test_segment_015_1s.wav", 0.3741, 0.1759, 1.51, 0.60, 0.40),
    # Xixi
    ("Xixi", "test_segment_003_4s.wav", 0.5457, 0.0043, 4.85, 3.74, 0.77),
    ("Xixi", "test_segment_011_1s.wav", 0.3607, 0.1893, 1.31, 0.54, 0.41),
    ("Xixi", "test_segment_012_1s.wav", 0.3029, 0.2471, 1.15, 0.54, 0.47),
    ("Xixi", "test_segment_013_1s.wav", 0.5035, 0.0465, 1.32, 0.60, 0.46),
    ("Xixi", "test_segment_014_1s.wav", 0.4239, 0.1261, 1.64, 0.70, 0.43),
]

SAMPLE_RATE = 16000


def find_test_audio(speaker: str, filename: str) -> Path | None:
    """查找测试音频文件路径."""
    # 测试音频在 test_segments 子目录中
    test_path = Path(f"asset/{speaker.lower()}/test_segments/{filename}")
    if test_path.is_file():
        return test_path

    # 尝试其他可能的路径
    base_dirs = [
        Path("asset"),
        Path("asset/john"),
        Path("asset/xixi"),
        Path("."),
    ]

    for base in base_dirs:
        # 递归查找
        for candidate in base.rglob(filename):
            if candidate.is_file():
                return candidate

    return None


def export_vad_comparison(
    audio_path: Path,
    output_dir: Path,
    speaker: str,
    filename: str,
    score: float,
    distance: float,
    orig_duration: float,
    vad_duration: float,
    retain_ratio: float,
) -> None:
    """导出 VAD 处理前后的音频对比."""
    logger.info(f"处理: {speaker}/{filename}")

    # 加载原始音频
    waveform = _load_audio(str(audio_path), SAMPLE_RATE)
    actual_duration = waveform.shape[0] / SAMPLE_RATE

    # 1. 导出原始音频
    original_output = output_dir / f"{speaker}_{filename.stem}_original.wav"
    ta.save(str(original_output), waveform.unsqueeze(0), SAMPLE_RATE)
    logger.info(f"  原始音频: {actual_duration:.2f}s → {original_output.name}")

    # 2. 使用 Silero VAD 处理
    waveform_silero = _apply_silero_vad(waveform, SAMPLE_RATE)
    silero_output = output_dir / f"{speaker}_{filename.stem}_vad_silero.wav"
    ta.save(str(silero_output), waveform_silero.unsqueeze(0), SAMPLE_RATE)
    silero_duration = waveform_silero.shape[0] / SAMPLE_RATE
    silero_retain = silero_duration / actual_duration if actual_duration > 0 else 0
    logger.info(f"  Silero VAD: {silero_duration:.2f}s ({silero_retain:.1%}) → {silero_output.name}")

    # 3. 使用 RMS VAD 处理（用于对比）
    speech_segs = _vad_segments(waveform, rms_threshold=0.005, sample_rate=SAMPLE_RATE)
    waveform_rms = torch.cat(speech_segs) if speech_segs else waveform
    rms_output = output_dir / f"{speaker}_{filename.stem}_vad_rms.wav"
    ta.save(str(rms_output), waveform_rms.unsqueeze(0), SAMPLE_RATE)
    rms_duration = waveform_rms.shape[0] / SAMPLE_RATE
    rms_retain = rms_duration / actual_duration if actual_duration > 0 else 0
    logger.info(f"  RMS VAD:   {rms_duration:.2f}s ({rms_retain:.1%}) → {rms_output.name}")

    # 4. 生成报告
    report_path = output_dir / f"{speaker}_{filename.stem}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"VAD 处理报告: {speaker}/{filename}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"识别得分: {score:.4f}\n")
        f.write(f"距离阈值: {distance:.4f}\n")
        f.write(f"阈值: 0.55\n\n")
        f.write("时长对比:\n")
        f.write(f"  原始音频: {actual_duration:.2f}s\n")
        f.write(f"  Silero VAD: {silero_duration:.2f}s (保留 {silero_retain:.1%})\n")
        f.write(f"  RMS VAD:    {rms_duration:.2f}s (保留 {rms_retain:.1%})\n\n")
        f.write(f"实验报告记录: {orig_duration:.2f}s → {vad_duration:.2f}s (保留 {retain_ratio:.1%})\n")

    logger.info(f"  报告: {report_path.name}\n")


def main() -> None:
    """主函数."""
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiment_log") / f"vad_export_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"输出目录: {output_dir}")
    logger.info(f"共 {len(FALSE_NEGATIVES)} 个误拒绝案例\n")

    processed = 0
    not_found = []

    for speaker, filename, score, distance, orig_dur, vad_dur, retain_ratio in FALSE_NEGATIVES:
        audio_path = find_test_audio(speaker, filename)

        if audio_path is None:
            logger.warning(f"未找到音频: {speaker}/{filename}")
            not_found.append((speaker, filename))
            continue

        try:
            export_vad_comparison(
                audio_path,
                output_dir,
                speaker,
                Path(filename),
                score,
                distance,
                orig_dur,
                vad_dur,
                retain_ratio,
            )
            processed += 1
        except Exception as e:
            logger.error(f"处理失败 {speaker}/{filename}: {e}")

    # 总结
    logger.info("=" * 60)
    logger.info(f"处理完成: {processed}/{len(FALSE_NEGATIVES)}")
    if not_found:
        logger.warning(f"未找到的音频 ({len(not_found)}):")
        for s, f in not_found:
            logger.warning(f"  - {s}/{f}")

    logger.info(f"\n输出目录: {output_dir}")
    logger.info("\n可以使用以下命令播放音频:")
    logger.info(f"  # 播放原始音频")
    logger.info(f"  afplay '{output_dir}/*_original.wav'")
    logger.info(f"  # 播放 VAD 处理后音频")
    logger.info(f"  afplay '{output_dir}/*_vad_silero.wav'")


if __name__ == "__main__":
    main()
