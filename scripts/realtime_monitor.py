#!/usr/bin/env python3
"""
Real-time microphone voiceprint monitor.

Continuously captures microphone audio, runs sliding-window voiceprint
recognition against an enrolled voiceprint, and displays real-time status
in the terminal.

Usage:
    uv run python scripts/realtime_monitor.py
    uv run python scripts/realtime_monitor.py --voiceprint asset/john/voice_best.pkl
    uv run python scripts/realtime_monitor.py --window-secs 3.0 --threshold 0.50
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        """Append audio samples. If buffer is full, oldest samples are overwritten."""
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
        """Return the last n_samples written, in chronological order.
        If fewer samples exist, return all available.
        """
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
        sample_rate: Target sample rate (must be 16000 for wespeaker).
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
            blocksize=1024,  # ~64ms blocks at 16kHz
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        """sounddevice callback — runs on audio thread."""
        if status:
            pass  # status flags like overflow, underflow
        self.buffer.write(indata[:, 0])  # mono

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
#  Real-Time Monitor
# --------------------------------------------------------------------------- #


class RealtimeMonitor:
    """Main monitoring loop: sliding window + embedding + terminal display.

    Args:
        voiceprint: Pre-loaded 256-dim reference embedding (torch.Tensor, L2-normalized).
        model: pyannote audio model (torch.nn.Module).
        device: inference device ("cpu" or "cuda").
        window_secs: sliding window duration in seconds.
        step_secs: step interval in seconds.
        buffer_secs: ring buffer capacity in seconds.
        threshold: cosine similarity threshold for "recognized".
        rms_threshold: RMS energy threshold below which audio is considered silent.
        sample_rate: audio sample rate.
    """

    def __init__(
        self,
        voiceprint: "torch.Tensor",
        model: "torch.nn.Module",
        device: str = "cpu",
        window_secs: float = 2.0,
        step_secs: float = 0.5,
        buffer_secs: float = 60.0,
        threshold: float = 0.55,
        rms_threshold: float = 0.005,
        sample_rate: int = 16000,
    ) -> None:
        import torch

        self.voiceprint = voiceprint
        self.model = model
        self.device = device
        self.window_secs = window_secs
        self.step_secs = step_secs
        self.threshold = threshold
        self.rms_threshold = rms_threshold
        self.sample_rate = sample_rate

        self._window_samples = int(window_secs * sample_rate)
        capacity = int(buffer_secs * sample_rate)
        self._buffer = RingBuffer(capacity)

        self._running = False
        self._window_count = 0
        self._start_time = 0.0
        self._last_state = "init"

    def _compute_rms(self, audio: np.ndarray) -> float:
        """Compute RMS energy of audio segment."""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio**2)))

    def _extract_score(self, audio: np.ndarray) -> float:
        """Extract embedding from audio and compute cosine similarity."""
        import torch
        import torch.nn.functional as F

        from src.wespeaker.wespeaker import _extract_embedding

        waveform = torch.from_numpy(audio)
        emb = F.normalize(_extract_embedding(self.model, waveform), dim=0)
        return float(torch.dot(emb, self.voiceprint).clamp(-1.0, 1.0).item())

    def _format_display(self, score: float | None, rms: float, elapsed: float) -> str:
        """Format the terminal display line."""
        duration = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        if score is None:
            state = "silent"
            line = f"\r[{duration}] ⚪ 静音 (RMS: {rms:.4f})     "
        elif score >= self.threshold:
            state = "speaking"
            line = f"\r[{duration}] 🟢 John 说话中 (confidence: {score:.4f})     "
        else:
            state = "other"
            line = f"\r[{duration}] 🔴 非 John 声音 (confidence: {score:.4f})     "

        self._last_state = state
        return line

    def run(self, device_index: int | None = None) -> None:
        """Start the monitoring loop. Blocks until KeyboardInterrupt."""
        import sounddevice as sd

        print("=" * 60)
        print("  WeSpeaker 实时声纹监控")
        print("=" * 60)
        print(
            f"  设备: {sd.query_devices(device_index)['name'] if device_index is not None else 'Default Input'}"
        )
        print(f"  采样率: {self.sample_rate} Hz")
        print(f"  窗口: {self.window_secs}s, 步长: {self.step_secs}s")
        print(f"  阈值: {self.threshold}")
        print(f"  缓冲区: {int(self.sample_rate * 60)} 样本 (ring buffer)")
        print("=" * 60)
        print("  按 Ctrl+C 停止")
        print("=" * 60)

        with AudioCapture(self._buffer, self.sample_rate, device_index):
            self._running = True
            self._start_time = time.time()
            last_step_time = self._start_time

            try:
                while self._running:
                    now = time.time()
                    if now - last_step_time >= self.step_secs:
                        last_step_time = now

                        available = self._buffer.available_samples
                        if available < self._window_samples:
                            continue

                        audio = self._buffer.read_last(self._window_samples)
                        rms = self._compute_rms(audio)

                        if rms < self.rms_threshold:
                            score = None
                        else:
                            score = self._extract_score(audio)
                            self._window_count += 1

                        elapsed = now - self._start_time
                        display = self._format_display(score, rms, elapsed)
                        sys.stdout.write(display)
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
    import pickle

    import torch
    import torch.nn.functional as F

    from src.wespeaker.wespeaker import WespeakerClient

    parser = argparse.ArgumentParser(description="WeSpeaker 实时声纹监控 — 麦克风实时识别")
    parser.add_argument(
        "--voiceprint",
        type=str,
        default="asset/john/voice_best.pkl",
        help="声纹 .pkl 文件路径 (默认: asset/john/voice_best.pkl)",
    )
    parser.add_argument("--model-path", type=str, default="./models/wespeaker")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--window-secs", type=float, default=2.0)
    parser.add_argument("--step-secs", type=float, default=0.5)
    parser.add_argument("--buffer-secs", type=float, default=60.0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--rms-threshold", type=float, default=0.005)
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument("--list-devices", action="store_true", help="列出可用音频设备并退出")
    list_group.add_argument("--device-index", type=int, default=None, help="指定麦克风设备索引")

    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    # Validate voiceprint file
    vp_path = Path(args.voiceprint)
    if not vp_path.is_file():
        print(f"错误: 声纹文件不存在: {vp_path}")
        print(
            f"请先运行注册: uv run python scripts/best_recognition.py enroll "
            f"--clean asset/john/registration_segments/ "
            f"--noise asset/john/test_noise_segments/嘈杂环境测试.m4a "
            f"--output {vp_path}"
        )
        sys.exit(1)

    # Load voiceprint
    with open(vp_path, "rb") as f:
        vp_data = pickle.load(f)
    voiceprint = F.normalize(torch.from_numpy(np.asarray(vp_data, dtype=np.float32)), dim=0)

    # Initialize model
    client = WespeakerClient(
        model_path=args.model_path, device=args.device, enable_augmentation=False
    )
    client._ensure_model()

    # Start monitoring
    monitor = RealtimeMonitor(
        voiceprint=voiceprint,
        model=client._model,
        device=args.device,
        window_secs=args.window_secs,
        step_secs=args.step_secs,
        buffer_secs=args.buffer_secs,
        threshold=args.threshold,
        rms_threshold=args.rms_threshold,
    )

    monitor.run(device_index=args.device_index)


if __name__ == "__main__":
    main()
