#!/usr/bin/env python3
"""声纹交叉测试 — 使用 WespeakerDeep API (enroll + load_templates + recognize_multi_pcm)。

测试注册和识别的完整流水线:
  - enroll(): 注册声纹并保存为 .pkl
  - load_templates(): 加载声纹到内存
  - recognize_multi_pcm(): 对待测试音频做多模板匹配

用法:
    uv run python scripts/cross_test.py
    uv run python scripts/cross_test.py --threshold 0.55
    uv run python scripts/cross_test.py --output-dir outputs
    uv run python scripts/cross_test.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchaudio

from wespeaker_deep_edge.wespeaker_deep_dege import DeepConfig, WespeakerDeep

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------- #
#  测试配置
# --------------------------------------------------------------------------- #

ASSET_COMBINE = Path("asset_combine")

SPEAKERS = {
    "John": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john/test_segments",
    },
    "John_USB": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john_usb/test_segments",
    },
    "John_MeetingRoom": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john_metting_room/test_segments",
    },
    "John_D_USB": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john_d_usb/test_segments",
    },
    "John_D_USB_AEC": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john_d_usb_AEC/test_segments",
    },
    "John_RealTime": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john_real_time/test_segments",
    },
    "Michael": {
        "register": str(ASSET_COMBINE / "Michael.wav"),
        "test_dir": "asset/michael/registration_segments",
    },
    "Xixi": {
        "register": str(ASSET_COMBINE / "Xixi.wav"),
        "test_dir": "asset/xixi/test_segments",
    },
    "Frank": {
        "register": str(ASSET_COMBINE / "Frank.wav"),
        "test_dir": "asset/frank/test_segments",
    },
    "Qingqing": {
        "register": str(ASSET_COMBINE / "Qingqing.wav"),
        "test_dir": "asset/qingqing/test_segments",
    },
    "Zhong": {
        "register": str(ASSET_COMBINE / "Zhong.wav"),
        "test_dir": "asset/zhong/test_segments",
    },
    "Zhong_D_USB": {
        "register": str(ASSET_COMBINE / "Zhong.wav"),
        "test_dir": "asset/zhong_d_usb/test_segments",
    },
    "Angle": {
        "register": str(ASSET_COMBINE / "angle.wav"),
        "test_dir": "asset/angle/test_segments",
    },
}

SPEAKER_ORDER = [
    "John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC", "John_RealTime",
    "Michael", "Zhong", "Zhong_D_USB", "Xixi", "Frank", "Qingqing", "Angle",
]

SAME_PERSON_GROUPS: dict[str, set[str]] = {
    "John": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC", "John_RealTime"},
    "John_USB": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC", "John_RealTime"},
    "John_MeetingRoom": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC", "John_RealTime"},
    "John_D_USB": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC", "John_RealTime"},
    "John_D_USB_AEC": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC", "John_RealTime"},
    "John_RealTime": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC", "John_RealTime"},
    "Michael": {"Michael"},
    "Zhong": {"Zhong", "Zhong_D_USB"},
    "Zhong_D_USB": {"Zhong", "Zhong_D_USB"},
    "Xixi": {"Xixi"},
    "Frank": {"Frank"},
    "Qingqing": {"Qingqing"},
    "Angle": {"Angle"},
}


def is_same_person(speaker1: str, speaker2: str) -> bool:
    if speaker1 == speaker2:
        return True
    return speaker2 in SAME_PERSON_GROUPS.get(speaker1, {speaker1})


def load_pcm(path: str | Path) -> tuple[np.ndarray, int]:
    """Load WAV as int16 mono PCM array + sample rate."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    pcm = (waveform * 32768).to(torch.int16).numpy().flatten()
    return pcm, int(sr)


# --------------------------------------------------------------------------- #
#  可视化
# --------------------------------------------------------------------------- #


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    accuracy: float,
    output_path: Path,
) -> None:
    """绘制混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, xticklabels=labels, yticklabels=labels,
        annot=True, fmt="d", cmap="Blues",
        cbar_kws={"label": "识别次数"},
        linewidths=1, linecolor="white", ax=ax,
    )
    ax.set_title(f"声纹交叉测试混淆矩阵 (准确率={accuracy:.1f}%)", fontsize=14)
    ax.set_xlabel("识别结果", fontsize=12)
    ax.set_ylabel("实际说话人", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"混淆矩阵已保存: {output_path}")
    plt.close()


# --------------------------------------------------------------------------- #
#  主流程
# --------------------------------------------------------------------------- #


def cross_test(
    threshold: float,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> None:
    print("=" * 60)
    print("声纹交叉测试 (WespeakerDeep: enroll + load_templates + recognize_multi_pcm)")
    print("=" * 60)

    # 1. Init
    print("\n[1/4] 初始化 WespeakerDeep...")
    deep = WespeakerDeep(config=DeepConfig(sim_threshold=threshold))
    print(f"  模型加载完成, threshold={threshold}")

    # 2. Collect files
    print("\n[2/4] 收集注册和测试文件...")
    active_people: list[str] = []
    enroll_files: dict[str, Path] = {}
    test_files_map: dict[str, list[Path]] = {}

    for person in SPEAKER_ORDER:
        reg = Path(SPEAKERS[person]["register"])
        if not reg.exists():
            print(f"  WARNING: 注册文件不存在 {reg}，跳过 {person}")
            continue
        test_dir = Path(SPEAKERS[person]["test_dir"])
        tests = sorted(test_dir.glob("*.wav"))
        if not tests:
            print(f"  WARNING: 无测试文件 {test_dir}，跳过 {person}")
            continue
        enroll_files[person] = reg
        test_files_map[person] = tests
        active_people.append(person)
        print(f"  {person}: 注册={reg.name}, 测试文件={len(tests)}")

    print(f"\n  活跃说话人 ({len(active_people)}): {active_people}")

    # 3. Enroll (deduplicated by audio file)
    print("\n[3/4] 注册声纹...")
    temp_dir = Path(mkdtemp(prefix="cross_test_"))
    audio_to_pk: dict[str, Path] = {}
    for person in active_people:
        audio = str(enroll_files[person])
        if audio not in audio_to_pk:
            pk = temp_dir / f"enroll_{len(audio_to_pk)}.pkl"
            result = deep.enroll(audio, str(pk))
            if not result["ok"]:
                print(f"  WARNING: {person} 注册失败: {result.get('error')}")
                continue
            audio_to_pk[audio] = pk
            print(f"  enroll {audio}")

    enroll_pks = {p: audio_to_pk[str(enroll_files[p])]
                  for p in active_people if str(enroll_files[p]) in audio_to_pk}
    active_people = [p for p in active_people if p in enroll_pks]

    # 4. Load templates
    template_files = {p: enroll_pks[p] for p in active_people}
    deep.load_templates(files=template_files)
    print(f"  已加载 {len(active_people)} 个模板: {active_people}")

    # 5. Recognize
    print(f"\n[4/4] 识别 ({len(active_people)} 人)...")
    results: list[tuple[str, str, float, str]] = []  # (actual, predicted, confidence, filename)

    for test_person in active_people:
        for test_file in test_files_map[test_person]:
            pcm, sr = load_pcm(test_file)
            result = deep.recognize_multi_pcm(pcm, sr)
            results.append((test_person, result.name, result.confidence, test_file.name))
        n = len(test_files_map[test_person])
        n_ok = sum(1 for r in results if r[0] == test_person)
        print(f"  测试 {test_person}: {n_ok}/{n} 完成")

    # 6. Statistics
    N = len(active_people)
    cm = np.zeros((N, N), dtype=int)  # rows=actual, cols=predicted
    correct = 0
    for actual, predicted, conf, fname in results:
        i = active_people.index(actual)
        j = active_people.index(predicted)
        cm[i, j] += 1
        if actual == predicted:
            correct += 1

    accuracy = correct / len(results) * 100 if results else 0

    print(f"\n准确率: {correct}/{len(results)} = {accuracy:.1f}%")

    print(f"\n  每人准确率:")
    for i, person in enumerate(active_people):
        total = int(cm[i].sum())
        correct_p = int(cm[i, i])
        acc_p = correct_p / total * 100 if total else 0
        print(f"    {person:16s}: {correct_p:3d}/{total:<3d} = {acc_p:5.1f}%")

    # Same-person group accuracy
    print(f"\n  同组准确率:")
    for person in active_people:
        group = SAME_PERSON_GROUPS[person]
        group_results = [r for r in results if r[0] == person]
        group_correct = sum(1 for r in group_results if r[1] in group)
        pct = group_correct / len(group_results) * 100 if group_results else 0
        print(f"    {person:16s}: {group_correct:3d}/{len(group_results):<3d} = {pct:5.1f}%")

    # Error analysis
    errors_fa: list[tuple[str, str, str, float]] = []  # actual, predicted, file, confidence
    errors_fr: list[tuple[str, str, str, float]] = []
    for actual, predicted, conf, fname in results:
        if is_same_person(actual, predicted):
            if conf < threshold:
                errors_fr.append((actual, predicted, fname, conf))
        else:
            if conf >= threshold:
                errors_fa.append((actual, predicted, fname, conf))

    total_comparisons = len(results)
    print(f"\n  误接受 (FA): {len(errors_fa)} ({len(errors_fa)/total_comparisons*100:.1f}%)")
    print(f"  误拒绝 (FR): {len(errors_fr)} ({len(errors_fr)/total_comparisons*100:.1f}%)")

    if verbose and errors_fa:
        print(f"\n  误接受详情 (最多10条):")
        for actual, predicted, fname, conf in errors_fa[:10]:
            print(f"    {actual}/{fname} → {predicted}: 置信度={conf:.3f}")
    if verbose and errors_fr:
        print(f"\n  误拒绝详情 (最多10条):")
        for actual, predicted, fname, conf in errors_fr[:10]:
            print(f"    {actual}/{fname} → {predicted}: 置信度={conf:.3f}")

    # 7. Confusion matrix
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_path = output_dir / f"confusion_matrix_{timestamp}.png"
        plot_confusion_matrix(cm, active_people, accuracy, cm_path)

    # 8. Summary
    print("\n" + "-" * 60)
    print(f"SUMMARY: 准确率={accuracy:.1f}%, FA={len(errors_fa)}, FR={len(errors_fr)}")
    print("-" * 60)

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="声纹交叉测试 — WespeakerDeep API",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="识别阈值 (default: 0.55)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="图表输出目录",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细输出模式",
    )
    args = parser.parse_args()

    # Verify assets
    for name, cfg in SPEAKERS.items():
        if not Path(cfg["register"]).exists():
            print(f"错误: 注册文件不存在: {cfg['register']}")
            sys.exit(1)
        test_dir = Path(cfg["test_dir"])
        if not test_dir.is_dir() or not list(test_dir.glob("*.wav")):
            print(f"错误: 测试目录无 .wav 文件: {test_dir}")
            sys.exit(1)

    output_path = Path(args.output_dir) if args.output_dir else None
    cross_test(args.threshold, output_path, args.verbose)


if __name__ == "__main__":
    main()
