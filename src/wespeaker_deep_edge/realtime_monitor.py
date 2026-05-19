#!/usr/bin/env python3
"""
Real-time microphone voiceprint monitor.

Continuously captures microphone audio, runs sliding-window voiceprint
recognition against an enrolled voiceprint, and displays real-time status
in the terminal.

Usage:
    uv run python -m wespeaker_deep_edge.realtime_monitor
    uv run python -m wespeaker_deep_edge.realtime_monitor --voiceprint asset/john/voice_best.pkl
    uv run python -m wespeaker_deep_edge.realtime_monitor --window-secs 3.0 --step-secs 0.3
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .wespeaker_deep_dege import WespeakerDeep

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Ring Buffer
# --------------------------------------------------------------------------- #


class RingBuffer:
    """Fixed-size circular buffer for audio samples.

    sounddevice callback runs on a separate audio thread and writes
    into this buffer; the main loop reads from it.
    """

    def __init__(self, capacity_samples: int) -> None:
        self._buffer = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write_pos = 0
        self._total_written = 0

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        if n == 0:
            return

        start = self._write_pos % self._capacity
        remaining = self._capacity - start

        if n <= remaining:
            self._buffer[start : start + n] = data
        else:
            self._buffer[start:] = data[:remaining]
            self._buffer[: n - remaining] = data[remaining:]

        self._write_pos += n
        self._total_written += n

    def read_last(self, n_samples: int) -> np.ndarray:
        """Return the last n_samples written, in chronological order."""
        if self._total_written < n_samples:
            n_samples = self._total_written
        if n_samples == 0:
            return np.array([], dtype=np.float32)

        end = self._write_pos % self._capacity
        start = (end - n_samples) % self._capacity

        if start < end:
            return self._buffer[start:end].copy()
        else:
            return np.concatenate([self._buffer[start:], self._buffer[:end]])

    @property
    def available_samples(self) -> int:
        return min(self._total_written, self._capacity)


# --------------------------------------------------------------------------- #
#  Audio Capture
# --------------------------------------------------------------------------- #


class AudioCapture:
    """Wraps sounddevice.InputStream to capture microphone audio into a RingBuffer.

    Args:
        buffer: RingBuffer instance to write audio into.
        sample_rate: Target sample rate (16000 for wespeaker).
        device: sounddevice device index. None = default input.
    """

    def __init__(
        self,
        buffer: RingBuffer,
        sample_rate: int = 16000,
        device: int | None = None,
    ) -> None:
        self.buffer = buffer
        self.sample_rate = sample_rate
        self._stream = None

        import sounddevice as sd

        self._stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=sample_rate,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            pass
        self.buffer.write(indata[:, 0])

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False


# --------------------------------------------------------------------------- #
#  Real-Time Monitor
# --------------------------------------------------------------------------- #


class RealtimeMonitor:
    """Main monitoring loop: sliding window + embedding + terminal display.

    All voice processing uses WespeakerDeep with its default DeepConfig.

    Args:
        deep: WespeakerDeep instance (uses default DeepConfig).
        voiceprint_path: Path to .pkl voiceprint file.
        window_secs: sliding window duration in seconds.
        step_secs: step interval in seconds.
    """

    def __init__(
        self,
        deep: WespeakerDeep,
        voiceprint_path: str | Path,
        window_secs: float = 2.0,
        step_secs: float = 0.5,
    ) -> None:
        import torch

        self._deep = deep
        self._window_secs = window_secs
        self._step_secs = step_secs
        self._sample_rate = deep.sample_rate

        # Load voiceprint via WespeakerDeep
        vp_np = deep.load(voiceprint_path)
        self._voiceprint = torch.from_numpy(vp_np)

        self._window_samples = int(window_secs * self._sample_rate)
        capacity = int(60.0 * self._sample_rate)
        self._buffer = RingBuffer(capacity)

        self._running = False
        self._window_count = 0
        self._start_time = 0.0

    def _extract_score(self, audio: np.ndarray) -> float:
        """Extract embedding from audio and compute cosine similarity."""
        import torch

        waveform = torch.from_numpy(audio).unsqueeze(0)
        emb = self._deep._model.extract_embedding_from_pcm(waveform, self._sample_rate)
        if emb is None:
            return 0.0
        return float(self._deep._model.cosine_similarity(emb, self._voiceprint))

    def run(self, device_index: int | None = None) -> None:
        """Start the monitoring loop. Blocks until KeyboardInterrupt."""
        import sounddevice as sd

        threshold = self._deep.deep_config.sim_threshold

        print("=" * 60)
        print("  WeSpeaker 实时声纹监控")
        print("=" * 60)
        device_name = (
            sd.query_devices(device_index)["name"]
            if device_index is not None
            else "Default Input"
        )
        print(f"  设备: {device_name}")
        print(f"  窗口: {self._window_secs}s, 步长: {self._step_secs}s")
        print(f"  阈值: {threshold} (DeepConfig)")
        print("=" * 60)
        print("  按 Ctrl+C 停止")
        print("=" * 60)

        with AudioCapture(self._buffer, self._sample_rate, device_index):
            self._running = True
            self._start_time = time.time()
            last_step_time = self._start_time

            try:
                while self._running:
                    now = time.time()
                    if now - last_step_time >= self._step_secs:
                        last_step_time = now

                        available = self._buffer.available_samples
                        if available < self._window_samples:
                            continue

                        audio = self._buffer.read_last(self._window_samples)
                        score = self._extract_score(audio)
                        self._window_count += 1

                        elapsed = now - self._start_time
                        duration = time.strftime("%H:%M:%S", time.gmtime(elapsed))

                        if score >= threshold:
                            line = f"\r[{duration}] 🟢 识别 (confidence: {score:.4f})     "
                        else:
                            line = f"\r[{duration}] 🔴 未识别 (confidence: {score:.4f})     "

                        sys.stdout.write(line)
                        sys.stdout.flush()

            except KeyboardInterrupt:
                self._running = False
                print()
                elapsed = time.time() - self._start_time
                print(f"\n监控停止，运行时长: {elapsed:.1f}s，处理窗口: {self._window_count}")


# --------------------------------------------------------------------------- #
#  CLI Entry
# --------------------------------------------------------------------------- #


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="WeSpeaker 实时声纹监控 — 麦克风实时识别")
    parser.add_argument(
        "--voiceprint",
        type=str,
        default="asset/john/voice_best.pkl",
        help="声纹 .pkl 文件路径 (默认: asset/john/voice_best.pkl)",
    )
    parser.add_argument("--window-secs", type=float, default=2.0, help="滑动窗口时长 (秒)")
    parser.add_argument("--step-secs", type=float, default=0.5, help="步长 (秒)")
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument("--list-devices", action="store_true", help="列出可用音频设备并退出")
    list_group.add_argument("--device-index", type=int, default=None, help="指定麦克风设备索引")

    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    vp_path = Path(args.voiceprint)
    if not vp_path.is_file():
        logger.error("声纹文件不存在: %s", vp_path)
        logger.error("请先注册声纹")
        sys.exit(1)

    deep = WespeakerDeep()

    monitor = RealtimeMonitor(
        deep=deep,
        voiceprint_path=vp_path,
        window_secs=args.window_secs,
        step_secs=args.step_secs,
    )

    monitor.run(device_index=args.device_index)


if __name__ == "__main__":
    main()
