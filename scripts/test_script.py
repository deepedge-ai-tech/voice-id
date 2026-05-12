#!/usr/bin/env python3
"""声纹测试录音脚本 — 交互式录制测试音频。

按 Enter 开始录制，再按 Enter 停止录制。录制的音频将保存到指定目录，
可用于声纹验证测试。

用法:
    uv run python scripts/test_script.py
    uv run python scripts/test_script.py --speaker frank
    uv run python scripts/test_script.py --output asset/frank/test_segments
    uv run python scripts/test_script.py --device-index 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd

# --------------------------------------------------------------------------- #
#  录音器
# --------------------------------------------------------------------------- #


class InteractiveRecorder:
    """交互式录音器 — 按 Enter 开始/停止录制。"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.recording = False
        self.audio_data: np.ndarray | None = None
        self.stream: sd.InputStream | None = None

    def _audio_callback(self, indata, frames, time, status):
        """音频流回调函数。"""
        if status:
            print(f"音频流状态: {status}", file=sys.stderr)
        if self.recording and self.audio_data is not None:
            # 追加音频数据
            current_data = indata[:, 0] if self.channels == 1 else indata
            self.audio_data = np.concatenate([self.audio_data, current_data])

    def start_recording(self):
        """开始录制。"""
        if self.recording:
            print("已经在录制中...")
            return

        self.audio_data = np.array([], dtype=np.float32)
        self.recording = True

        if self.stream is None:
            # 创建音频流
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                device=self.device_index,
            )
            self.stream.start()

        print("🔴 录制中... (按 Enter 停止)")

    def stop_recording(self) -> np.ndarray:
        """停止录制并返回音频数据。"""
        if not self.recording:
            print("没有在录制中...")
            return np.array([], dtype=np.float32)

        self.recording = False

        if self.audio_data is None or len(self.audio_data) == 0:
            print("没有录制到音频数据")
            return np.array([], dtype=np.float32)

        duration = len(self.audio_data) / self.sample_rate
        print(f"✅ 录制完成，时长: {duration:.2f} 秒")

        return self.audio_data

    def close(self):
        """关闭音频流。"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def save_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    """保存音频到 WAV 文件。"""
    # 转换到 int16 范围
    audio_int16 = np.clip(audio, -1, 1)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, audio_int16)


def list_devices() -> None:
    """列出可用的音频设备。"""
    print("\n可用音频设备:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev["name"]
        max_inputs = dev["max_inputs"]
        max_outputs = dev["max_output_channels"]
        host_api = sd.query_hostapis(dev["hostapi"])["name"]

        in_mark = ">" if max_inputs > 0 else " "
        out_mark = "<" if max_outputs > 0 else " "

        print(f"  {in_mark} {i} {name}, {host_api} ({max_inputs} in, {max_outputs} out){out_mark}")
    print("-" * 60)


# --------------------------------------------------------------------------- #
#  主程序
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="声纹测试录音脚本")
    parser.add_argument(
        "--speaker",
        default="john",
        help="说话人名称 (default: john)",
    )
    parser.add_argument(
        "--output",
        help="输出文件路径 (default: asset/{speaker}/测试.wav)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        help="音频设备索引 (使用 --list-devices 查看)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="采样率 (default: 16000)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出可用音频设备",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # 确定输出文件
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"asset/{args.speaker}/测试.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("声纹测试录音脚本")
    print("=" * 60)
    print(f"说话人: {args.speaker}")
    print(f"输出文件: {output_path}")
    print(f"采样率: {args.sample_rate} Hz")
    if args.device_index is not None:
        print(f"音频设备: {args.device_index}")
    print("=" * 60)

    # 初始化录音器
    recorder = InteractiveRecorder(
        sample_rate=args.sample_rate,
        channels=1,
        device_index=args.device_index,
    )

    try:
        print("\n操作说明:")
        print("  Enter    - 开始/停止录制")
        print("  q + Enter - 退出程序")
        print()

        count = 0
        while True:
            user_input = input(f"[{count}] 按 Enter 开始录制 (或 q 退出): ")

            if user_input.lower() == "q":
                print("退出程序...")
                break

            # 开始录制
            recorder.start_recording()

            # 等待停止
            input("按 Enter 停止录制...")

            # 停止录制
            audio = recorder.stop_recording()

            if len(audio) > 0:
                # 保存音频（覆盖同名文件）
                save_wav(audio, output_path, args.sample_rate)
                print(f"  已保存: {output_path}")
                count += 1

    except KeyboardInterrupt:
        print("\n\n中断退出")
    finally:
        recorder.close()
        print(f"\n共录制 {count} 次，最终文件: {output_path}")


if __name__ == "__main__":
    main()
