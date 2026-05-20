#!/usr/bin/env python3
"""
VAD-triggered real-time voiceprint recognition.

Captures microphone audio in real-time. Silero VAD detects speech onset;
on detection, captures the first 2 seconds of speech and sends it to
WespeakerDeep for voiceprint recognition. No sliding window.

Usage:
    uv run python scripts/sliding_window_analyzer.py
    uv run python scripts/sliding_window_analyzer.py --voiceprint voice.pkl
    uv run python scripts/sliding_window_analyzer.py --package-pk-index 1
    uv run python scripts/sliding_window_analyzer.py --list-devices
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import logging
import time
from datetime import datetime

import numpy as np
import torch
import torchaudio

from src.wespeaker_deep_edge._utils import _load_silero_vad
from src.wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Ring Buffer
# --------------------------------------------------------------------------- #


class RingBuffer:
    """Fixed-size circular buffer for audio samples."""

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
    """Wraps sounddevice.InputStream to capture microphone audio into a RingBuffer."""

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
#  VAD-Triggered Recognizer
# --------------------------------------------------------------------------- #


class VadTriggeredRecognizer:
    """VAD 触发的实时声纹识别。

    持续采集麦克风音频 → Silero VAD 检测语音起点 → 采集前 2 秒语音
    → WespeakerDeep 识别 → 等待静音后重新待机。

    Args:
        deep: WespeakerDeep 实例（使用默认 DeepConfig）。
        voiceprint_path: 声纹 .pkl 文件路径。None 则使用内置声纹。
        vad_check_secs: VAD 检测窗口长度（秒）。越小响应越快，但可能误触发。
        cooldown_secs: 识别后等待静音的冷却时间（秒）。
    """

    def __init__(
        self,
        deep: WespeakerDeep,
        voiceprint_path: str | Path | None = None,
        vad_check_secs: float = 0.5,
        cooldown_secs: float = 1.5,
    ) -> None:
        self._deep = deep
        self._sample_rate = deep.sample_rate
        self._speech_duration = 2.0
        self._lookback_secs = 0.2  # 回看补偿 VAD 检测延迟
        self._vad_check_samples = int(vad_check_secs * self._sample_rate)
        self._cooldown_secs = cooldown_secs

        # Resolve voiceprint
        self._pk_path: str | None = None
        if voiceprint_path is not None:
            self._pk_path = str(Path(voiceprint_path).resolve())
            vp_np = deep.load(self._pk_path)
        else:
            from src.wespeaker_deep_edge._voiceprints import get_voiceprint_path

            index = (
                deep.deep_config.package_pk_index
                if deep.deep_config.package_pk_index is not None
                else 0
            )
            self._pk_path = get_voiceprint_path(index)
            vp_np = deep.load(self._pk_path)

        self._voiceprint = torch.from_numpy(vp_np)

        # Save directory for captured audio
        self._save_dir = Path("asset/john_real_time/test_segments")
        self._save_dir.mkdir(parents=True, exist_ok=True)

        # Ring buffer: 10 seconds capacity
        self._buffer = RingBuffer(int(10.0 * self._sample_rate))

        # Silero VAD
        self._vad = _load_silero_vad()

        # State machine
        self._prev_speaking = False
        self._state = "idle"  # idle | collecting | cooldown
        self._trigger_time = 0.0
        self._silence_since = 0.0

    def _extract_score(self, audio: np.ndarray) -> float:
        """提取 embedding 并计算原始余弦相似度（范围 [-1, 1]，不做 [0,1] 归一化）。"""
        waveform = torch.from_numpy(audio).unsqueeze(0)
        emb = self._deep._model.extract_embedding_from_pcm(waveform, self._sample_rate)
        if emb is None:
            return 0.0
        return float(torch.dot(emb, self._voiceprint) / (torch.norm(emb) * torch.norm(self._voiceprint)))

    def _is_speaking(self) -> bool:
        """检查 VAD 在最近的音频窗口中是否检测到语音。"""
        available = self._buffer.available_samples
        if available < int(0.1 * self._sample_rate):
            return False

        check_len = min(self._vad_check_samples, available)
        audio = self._buffer.read_last(check_len)
        audio_tensor = torch.from_numpy(audio)

        stamps = self._vad["get_speech_timestamps"](
            audio_tensor,
            self._vad["model"],
            threshold=0.5,
            sampling_rate=self._sample_rate,
            min_speech_duration_ms=50,
            min_silence_duration_ms=50,
            speech_pad_ms=0,
        )
        return len(stamps) > 0

    def run(self, device_index: int | None = None) -> None:
        """启动 VAD 触发式声纹识别。阻塞直到 KeyboardInterrupt。"""
        import sounddevice as sd

        threshold = self._deep.deep_config.sim_threshold
        vad_interval = 0.1  # VAD 检查间隔 (100ms)

        print("=" * 60)
        print("  WeSpeaker VAD 触发声纹识别")
        print("=" * 60)
        device_name = (
            sd.query_devices(device_index)["name"]
            if device_index is not None
            else "Default Input"
        )
        print(f"  设备: {device_name}")
        print(f"  语音采集: {self._speech_duration}s, 阈值: {threshold}")
        print(f"  流程: 待机 → VAD 检测到语音 → 采集 2s → 识别 → 冷却")
        print(f"  声纹文件: {self._pk_path}")
        print(f"  模型文件: {self._deep._model_dir}")
        print("=" * 60)
        print("  按 Ctrl+C 停止")
        print("=" * 60)

        with AudioCapture(self._buffer, self._sample_rate, device_index):
            start_time = time.time()
            last_step = start_time
            last_heartbeat = start_time

            try:
                while True:
                    now = time.time()
                    if now - last_step < vad_interval:
                        time.sleep(0.005)
                        continue
                    last_step = now

                    speaking = self._is_speaking()
                    elapsed = now - start_time
                    ts = time.strftime("%H:%M:%S", time.gmtime(elapsed))

                    if self._state == "idle":
                        # 检测语音起点 (silence → speech 跳变)
                        if speaking and not self._prev_speaking:
                            self._state = "collecting"
                            self._trigger_time = now
                            print(f"\n[{ts}] 🎤 检测到语音，采集中 ...")

                        # 5 秒心跳
                        if now - last_heartbeat >= 5.0:
                            last_heartbeat = now
                            sys.stdout.write(".")
                            sys.stdout.flush()

                    elif self._state == "collecting":
                        duration = now - self._trigger_time
                        remaining = self._speech_duration - duration
                        if remaining > 0:
                            sys.stdout.write(
                                f"\r[{ts}] 🎤 采集中 ({duration:.1f}s / {self._speech_duration:.0f}s) "
                                f"{'█' * int(duration) + '░' * max(0, int(remaining))}"
                            )
                            sys.stdout.flush()
                        else:
                            # 采集完成 → 保存并识别
                            sample_count = int(
                                (self._speech_duration + self._lookback_secs) * self._sample_rate
                            )
                            audio = self._buffer.read_last(sample_count)
                            score = self._extract_score(audio)

                            # 保存音频
                            fname = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{score:.4f}.wav"
                            save_path = self._save_dir / fname
                            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                            torchaudio.save(str(save_path), audio_tensor, self._sample_rate)

                            if score >= threshold:
                                print(
                                    f"\r[{ts}] ✅ 识别成功 (confidence: {score:.4f})  "
                                    f"[{fname}]   "
                                )
                            else:
                                print(
                                    f"\r[{ts}] ❌ 未识别   (confidence: {score:.4f})  "
                                    f"[{fname}]   "
                                )

                            self._state = "cooldown"
                            self._silence_since = now

                    elif self._state == "cooldown":
                        if not speaking:
                            if now - self._silence_since >= self._cooldown_secs:
                                self._state = "idle"
                                print(f"[{ts}] 🔇 静音，重新待机")
                        else:
                            self._silence_since = now  # 还在说话，重置静音计时

                    self._prev_speaking = speaking

            except KeyboardInterrupt:
                print()
                elapsed = time.time() - start_time
                print(f"\n停止，运行时长: {elapsed:.1f}s")


# --------------------------------------------------------------------------- #
#  CLI Entry
# --------------------------------------------------------------------------- #


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WeSpeaker VAD 触发声纹识别 — 麦克风采集 + VAD 触发 + 2s 识别"
    )
    parser.add_argument(
        "--voiceprint",
        type=str,
        default=None,
        help="声纹 .pkl 文件路径 (默认: 使用内置声纹)",
    )
    parser.add_argument(
        "--package-pk-index",
        type=int,
        default=None,
        help="内置声纹索引 (0=John, 1=Frank, ...)",
    )
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument("--list-devices", action="store_true", help="列出可用音频设备并退出")
    list_group.add_argument("--device-index", type=int, default=None, help="指定麦克风设备索引")

    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    deep = WespeakerDeep(package_pk_index=args.package_pk_index)

    recognizer = VadTriggeredRecognizer(
        deep=deep,
        voiceprint_path=args.voiceprint,
    )

    recognizer.run(device_index=args.device_index)


if __name__ == "__main__":
    main()
