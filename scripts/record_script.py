#!/usr/bin/env python3
"""声纹注册录音脚本 — 交互式录制注册音频。

按 Enter 开始录制，再按 Enter 停止录制。录制的音频将保存到指定目录，
可用于声纹注册。每次录制前会显示 docs/record_script.md 中的参考句子。

用法:
    uv run python scripts/record_script.py
    uv run python scripts/record_script.py --speaker frank
    uv run python scripts/record_script.py --output asset/frank/registration_segments
    uv run python scripts/record_script.py --device-index 1
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile


# --------------------------------------------------------------------------- #
#  脚本读取
# --------------------------------------------------------------------------- #


def read_record_script(path: Path) -> list[str]:
    """读取录音脚本文件，提取所有带引号的句子。

    Args:
        path: 脚本文件路径

    Returns:
        句子列表（去除引号）
    """
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8")
    # 匹配中文引号 "" (U+201C, U+201D) 内的内容
    left_quote = chr(0x201C)  # "
    right_quote = chr(0x201D)  # "
    pattern = left_quote + "(.*?)" + right_quote
    matches = re.findall(pattern, content)
    return matches


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
        max_inputs = dev["max_input_channels"]
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
    parser = argparse.ArgumentParser(description="声纹注册录音脚本")
    parser.add_argument(
        "--speaker",
        default="john",
        help="说话人名称 (default: john)",
    )
    parser.add_argument(
        "--output",
        help="输出目录 (default: asset/{speaker}/registration_segments)",
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
    parser.add_argument(
        "--script",
        default="docs/record_script.md",
        help="录音脚本文件路径 (default: docs/record_script.md)",
    )
    parser.add_argument(
        "--no-script",
        action="store_true",
        help="不使用脚本，自由录制",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"asset/{args.speaker}/registration_segments")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("声纹注册录音脚本")
    print("=" * 60)
    print(f"说话人: {args.speaker}")
    print(f"输出目录: {output_dir}")
    print(f"采样率: {args.sample_rate} Hz")
    if args.device_index is not None:
        print(f"音频设备: {args.device_index}")

    # 读取脚本
    use_script = not args.no_script
    sentences: list[str] = []
    if use_script:
        script_path = Path(args.script)
        if script_path.exists():
            sentences = read_record_script(script_path)
            print(f"脚本: {script_path} ({len(sentences)} 句)")
        else:
            print(f"警告: 脚本文件不存在: {script_path}")
            print("将使用自由录制模式")
            use_script = False
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

        script_index = 0
        count = 0
        while True:
            # 显示当前句子（如果有脚本）
            if use_script and script_index < len(sentences):
                current_sentence = sentences[script_index]
                print(f"\n📝 句子 {script_index + 1}/{len(sentences)}:")
                print(f'   "{current_sentence}"')
                print()

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
                # 保存音频
                output_path = output_dir / f"segment_{count:03d}.wav"
                save_wav(audio, output_path, args.sample_rate)
                print(f"  已保存: {output_path}")
                count += 1

                # 脚本索引递增
                if use_script:
                    script_index += 1
                    # 脚本用完后提示
                    if script_index >= len(sentences):
                        print(f"\n✅ 所有脚本句子已录制完成！")
                        print(f"   共录制 {count} 个片段")
                        user_input = input("\n继续自由录制？ (y/n, 默认 n): ")
                        if user_input.lower() != "y":
                            break
                        use_script = False
                        print("\n切换到自由录制模式...")

    except KeyboardInterrupt:
        print("\n\n中断退出")
    finally:
        recorder.close()
        print(f"\n共录制 {count} 个片段")


if __name__ == "__main__":
    main()
