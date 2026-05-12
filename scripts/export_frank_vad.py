#!/usr/bin/env python3
"""导出 Frank 测试片段的 VAD 处理前后音频对比.

用法:
    uv run python scripts/export_frank_vad.py
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

# Frank 所有测试片段
FRANK_SEGMENTS = [
    ("Frank", "test_segment_000_4s.wav", 0.5580),
    ("Frank", "test_segment_001_4s.wav", 0.6193),
    ("Frank", "test_segment_002_4s.wav", 0.4758),  # 误拒绝
    ("Frank", "test_segment_003_4s.wav", 0.4443),  # 误拒绝
    ("Frank", "test_segment_004_3s.wav", 0.6383),
    ("Frank", "test_segment_005_3s.wav", 0.6062),
    ("Frank", "test_segment_007_2s.wav", 0.5251),  # 误拒绝
    ("Frank", "test_segment_008_2s.wav", 0.5933),
    ("Frank", "test_segment_009_2s.wav", 0.5880),
    ("Frank", "test_segment_011_1s.wav", 0.2603),  # 误拒绝
    ("Frank", "test_segment_012_1s.wav", 0.2416),  # 误拒绝
    ("Frank", "test_segment_013_1s.wav", 0.4267),  # 误拒绝
    ("Frank", "test_segment_014_1s.wav", 0.3853),  # 误拒绝
    ("Frank", "test_segment_015_1s.wav", 0.3807),  # 误拒绝
]

SAMPLE_RATE = 16000


def find_test_audio(speaker: str, filename: str) -> Path | None:
    """查找测试音频文件路径."""
    test_path = Path(f"asset/{speaker.lower()}/test_segments/{filename}")
    if test_path.is_file():
        return test_path
    return None


def export_vad_comparison(
    audio_path: Path,
    output_dir: Path,
    speaker: str,
    filename: str,
    score: float,
) -> None:
    """导出 VAD 处理前后的音频对比."""
    logger.info(f"处理: {speaker}/{filename} (得分: {score:.4f})")

    # 加载原始音频
    waveform = _load_audio(str(audio_path), SAMPLE_RATE)
    actual_duration = waveform.shape[0] / SAMPLE_RATE

    # 1. 导出原始音频
    original_output = output_dir / f"{speaker}_{filename.stem}_original.wav"
    ta.save(str(original_output), waveform.unsqueeze(0), SAMPLE_RATE)

    # 2. 使用 Silero VAD 处理
    waveform_silero = _apply_silero_vad(waveform, SAMPLE_RATE)
    silero_output = output_dir / f"{speaker}_{filename.stem}_vad_silero.wav"
    ta.save(str(silero_output), waveform_silero.unsqueeze(0), SAMPLE_RATE)
    silero_duration = waveform_silero.shape[0] / SAMPLE_RATE
    silero_retain = silero_duration / actual_duration if actual_duration > 0 else 0

    # 3. 使用 RMS VAD 处理（用于对比）
    speech_segs = _vad_segments(waveform, rms_threshold=0.005, sample_rate=SAMPLE_RATE)
    waveform_rms = torch.cat(speech_segs) if speech_segs else waveform
    rms_output = output_dir / f"{speaker}_{filename.stem}_vad_rms.wav"
    ta.save(str(rms_output), waveform_rms.unsqueeze(0), SAMPLE_RATE)
    rms_duration = waveform_rms.shape[0] / SAMPLE_RATE
    rms_retain = rms_duration / actual_duration if actual_duration > 0 else 0

    # 标记状态
    status = "❌ 误拒绝" if score < 0.55 else "✅ 通过"

    logger.info(f"  原始: {actual_duration:.2f}s → Silero: {silero_duration:.2f}s ({silero_retain:.1%}), RMS: {rms_duration:.2f}s ({rms_retain:.1%}) {status}")


def main() -> None:
    """主函数."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiment_log") / f"frank_vad_export_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"输出目录: {output_dir}")
    logger.info(f"共 {len(FRANK_SEGMENTS)} 个测试片段\n")

    processed = 0
    not_found = []

    for speaker, filename, score in FRANK_SEGMENTS:
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
            )
            processed += 1
        except Exception as e:
            logger.error(f"处理失败 {speaker}/{filename}: {e}")

    # 总结
    logger.info("=" * 60)
    logger.info(f"处理完成: {processed}/{len(FRANK_SEGMENTS)}")

    logger.info(f"\n输出目录: {output_dir}")
    logger.info("\n播放命令:")
    logger.info(f"  # 原始音频")
    logger.info(f"  afplay '{output_dir}/*_original.wav'")
    logger.info(f"  # VAD 处理后")
    logger.info(f"  afplay '{output_dir}/*_vad_silero.wav'")
    logger.info(f"\n  # 单个对比（误拒绝案例）")
    logger.info(f"  afplay '{output_dir}/Frank_test_segment_002_4s_original.wav'")
    logger.info(f"  afplay '{output_dir}/Frank_test_segment_002_4s_vad_silero.wav'")


if __name__ == "__main__":
    main()
