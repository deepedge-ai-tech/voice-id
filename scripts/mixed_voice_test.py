#!/usr/bin/env python3
"""混杂声纹交叉测试 - 测试混合了其他人声音的识别效果。

测试场景:
  - John + Qingqing vs John 声纹
  - Frank + Zhong vs Frank 声纹
  - Michael + Xixi vs Michael 声纹

混杂比例:
  - 旁人声音 100% (等音量)
  - 旁人声音 50% (half音量)
  - 旁人声音 20% (低音量)

用法:
    uv run python scripts/mixed_voice_test.py
    uv run python scripts/mixed_voice_test.py --threshold 0.50
    uv run python scripts/mixed_voice_test.py --output-dir outputs
    uv run python scripts/mixed_voice_test.py --verbose
    uv run python scripts/mixed_voice_test.py --regenerate  # 重新生成混合音频
"""

import logging
import pickle
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.wespeaker import WespeakerBest

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------- #
#  测试配置
# --------------------------------------------------------------------------- #


@dataclass
class VoicePair:
    """声纹对配置."""

    target_name: str
    other_name: str
    target_audio: str
    other_audio: str
    voiceprint_path: str


@dataclass
class MixedTestCase:
    """混合测试用例."""

    pair: VoicePair
    other_ratio: float
    mixed_audio_path: str
    expected_match: bool = True


# 定义测试配置
VOICE_PAIRS = {
    "John": VoicePair(
        target_name="John",
        other_name="Qingqing",
        target_audio="asset/john/安静环境测试测试.m4a",
        other_audio="asset/qingqing/测试.wav",
        voiceprint_path="asset/john/voice_best.pkl",
    ),
    "Frank": VoicePair(
        target_name="Frank",
        other_name="Zhong",
        target_audio="asset/frank/frank 测试.m4a",
        other_audio="asset/zhong/测试.wav",
        voiceprint_path="asset/frank/voice_best.pkl",
    ),
    "Michael": VoicePair(
        target_name="Michael",
        other_name="Xixi",
        target_audio="asset/michael/测试.wav",
        other_audio="asset/xixi/测试.wav",
        voiceprint_path="asset/michael/voice_best.pkl",
    ),
}

# 混杂比例
OTHER_RATIOS = [1.0, 0.5, 0.2]
RATIO_LABELS = {1.0: "100%", 0.5: "50%", 0.2: "20%"}

# --------------------------------------------------------------------------- #
#  混合音频生成
# --------------------------------------------------------------------------- #


def generate_mixed_audios(output_base_dir: str = "asset/mixed") -> dict[str, MixedTestCase]:
    """生成所有混合音频文件.

    Args:
        output_base_dir: 输出目录

    Returns:
        测试用例字典
    """
    from scripts.mix_audio import mix_audios

    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases: dict[str, MixedTestCase] = {}

    for pair_name, pair in VOICE_PAIRS.items():
        for ratio in OTHER_RATIOS:
            ratio_label = RATIO_LABELS[ratio]
            output_name = f"{pair_name.lower()}_mixed_{pair.other_name.lower()}_{ratio_label.replace('%', 'pct')}.wav"
            output_path = output_dir / output_name

            print(f"生成: {output_name}")
            mix_audios(
                target_path=pair.target_audio,
                other_path=pair.other_audio,
                output_path=str(output_path),
                other_ratio=ratio,
            )

            case_key = f"{pair_name}_{ratio_label}"
            test_cases[case_key] = MixedTestCase(
                pair=pair,
                other_ratio=ratio,
                mixed_audio_path=str(output_path),
                expected_match=True,
            )

    return test_cases


# --------------------------------------------------------------------------- #
#  可视化函数
# --------------------------------------------------------------------------- #


def plot_mixed_voice_scores(
    results: list[dict[str, Any]],
    threshold: float,
    output_path: Path | None = None,
) -> None:
    """绘制混杂声音识别得分图.

    Args:
        results: 测试结果列表
        threshold: 识别阈值
        output_path: 输出路径
    """
    # 按说话人和比例组织数据
    speakers = list(VOICE_PAIRS.keys())
    ratios = [RATIO_LABELS[r] for r in OTHER_RATIOS]

    # 创建得分矩阵 (行: 说话人, 列: 比例)
    scores = np.zeros((len(speakers), len(ratios)))

    for r in results:
        speaker = r["speaker"]
        ratio_label = RATIO_LABELS[r["other_ratio"]]
        score = r["score"]
        row = speakers.index(speaker)
        col = ratios.index(ratio_label)
        scores[row, col] = score

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(speakers))
    width = 0.2

    colors = ["#d32f2f", "#f57c00", "#388e3c"]  # 红、橙、绿

    for i, (ratio, color) in enumerate(zip(ratios, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, scores[:, i], width, label=f"旁人 {ratio}", color=color)

        # 添加数值标签
        for bar, score in zip(bars, scores[:, i]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(y=threshold, color="blue", linestyle="--", linewidth=2, label=f"阈值 ({threshold})")

    ax.set_xlabel("说话人", fontsize=12)
    ax.set_ylabel("相似度得分", fontsize=12)
    ax.set_title("混杂声音识别得分 (不同旁人音量比例)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(speakers)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n📊 图表已保存: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_score_decline(
    results: list[dict[str, Any]],
    output_path: Path | None = None,
) -> None:
    """绘制得分随混杂程度变化的趋势图.

    Args:
        results: 测试结果列表
        output_path: 输出路径
    """
    speakers = list(VOICE_PAIRS.keys())
    ratios_pct = [int(r * 100) for r in OTHER_RATIOS]  # [100, 50, 20]

    fig, ax = plt.subplots(figsize=(10, 6))

    for speaker in speakers:
        speaker_results = [r for r in results if r["speaker"] == speaker]
        speaker_results.sort(key=lambda x: x["other_ratio"], reverse=True)
        scores = [r["score"] for r in speaker_results]

        ax.plot(ratios_pct, scores, marker="o", linewidth=2, markersize=8, label=speaker)

    ax.set_xlabel("旁人音量比例 (%)", fontsize=12)
    ax.set_ylabel("相似度得分", fontsize=12)
    ax.set_title("混杂程度对识别得分的影响", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(ratios_pct)
    ax.set_xticklabels([f"{r}%" for r in ratios_pct])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"📊 趋势图已保存: {output_path}")
    else:
        plt.show()

    plt.close()


# --------------------------------------------------------------------------- #
#  测试执行
# --------------------------------------------------------------------------- #


def mixed_voice_cross_test(
    threshold: float,
    output_dir: Path | None = None,
    verbose: bool = False,
    regenerate: bool = False,
) -> None:
    """执行混杂声纹交叉测试.

    Args:
        threshold: 识别阈值
        output_dir: 输出目录
        verbose: 详细输出
        regenerate: 是否重新生成混合音频
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    recognizer = WespeakerBest()
    recognizer.config = recognizer.config.__class__(
        **{**vars(recognizer.config), "sim_threshold": threshold}
    )

    # 加载模型
    print("加载声纹识别模型...")
    recognizer._client._ensure_model()

    # 生成或加载混合音频
    mixed_dir = Path("asset/mixed")
    if regenerate:
        print("\n生成混合音频...")
        test_cases = generate_mixed_audios(str(mixed_dir))
    else:
        print("\n检查混合音频...")
        if not mixed_dir.exists():
            print("混合音频目录不存在，正在生成...")
            test_cases = generate_mixed_audios(str(mixed_dir))
        else:
            # 检查是否所有文件都存在
            test_cases = {}
            for pair_name, pair in VOICE_PAIRS.items():
                for ratio in OTHER_RATIOS:
                    ratio_label = RATIO_LABELS[ratio]
                    output_name = f"{pair_name.lower()}_mixed_{pair.other_name.lower()}_{ratio_label.replace('%', 'pct')}.wav"
                    output_path = mixed_dir / output_name

                    if not output_path.exists():
                        print(f"文件不存在: {output_name}，正在重新生成...")
                        test_cases = generate_mixed_audios(str(mixed_dir))
                        break
                    else:
                        case_key = f"{pair_name}_{ratio_label}"
                        test_cases[case_key] = MixedTestCase(
                            pair=pair,
                            other_ratio=ratio,
                            mixed_audio_path=str(output_path),
                            expected_match=True,
                        )
                if regenerate:
                    break

    # 加载声纹
    print("\n加载声纹...")
    voiceprints: dict[str, torch.Tensor] = {}
    for name, pair in VOICE_PAIRS.items():
        vp_path = Path(pair.voiceprint_path)
        if not vp_path.exists():
            print(f"警告: 声纹文件不存在: {vp_path}")
            print(f"请先运行最佳配置注册生成声纹文件")
            return

        with open(vp_path, "rb") as f:
            voiceprints[name] = torch.from_numpy(pickle.load(f))

    # 执行测试
    print("\n" + "=" * 70)
    print("混杂声纹交叉测试")
    print("=" * 70)
    print(f"阈值: {threshold:.2f}")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    for case_key, case in sorted(test_cases.items()):
        speaker = case.pair.target_name
        other = case.pair.other_name
        ratio_label = RATIO_LABELS[case.other_ratio]

        # 执行识别
        tmp_pk = Path("/tmp/voice_mixed_test.pkl")
        with open(tmp_pk, "wb") as f:
            pickle.dump(voiceprints[speaker].cpu().numpy(), f)

        result = recognizer.recognize(case.mixed_audio_path, str(tmp_pk))
        score = result["confidence"]
        is_match = result["is_recognized"]

        # 记录结果
        results.append(
            {
                "speaker": speaker,
                "other_speaker": other,
                "other_ratio": case.other_ratio,
                "ratio_label": ratio_label,
                "score": float(score),
                "is_match": is_match,
                "passed": is_match == case.expected_match,
            }
        )

        # 打印结果
        status = "✅ PASS" if is_match else "❌ FAIL"
        print(f"{speaker:8} + {other:10} ({ratio_label:>4}): {score:.4f} {status}")

    # 统计结果
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("\n" + "=" * 70)
    print(f"测试总结: {passed}/{total} 通过")
    print("=" * 70)

    # 按比例分组统计
    print("\n各比例统计:")
    for ratio in OTHER_RATIOS:
        ratio_label = RATIO_LABELS[ratio]
        ratio_results = [r for r in results if r["other_ratio"] == ratio]
        ratio_passed = sum(1 for r in ratio_results if r["passed"])
        ratio_avg = np.mean([r["score"] for r in ratio_results])
        print(f"  旁人 {ratio_label:>4}: {ratio_passed}/{len(ratio_results)} 通过, 平均得分 {ratio_avg:.4f}")

    # 按说话人分组统计
    print("\n各说话人统计:")
    for speaker in VOICE_PAIRS.keys():
        speaker_results = [r for r in results if r["speaker"] == speaker]
        speaker_passed = sum(1 for r in speaker_results if r["passed"])
        speaker_avg = np.mean([r["score"] for r in speaker_results])
        print(f"  {speaker}: {speaker_passed}/{len(speaker_results)} 通过, 平均得分 {speaker_avg:.4f}")

    # 生成图表
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now()

        # 柱状图
        bar_path = output_dir / f"mixed_voice_scores_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        plot_mixed_voice_scores(results, threshold, bar_path)

        # 趋势图
        trend_path = output_dir / f"mixed_voice_trend_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        plot_score_decline(results, trend_path)

        # 保存 JSON 结果
        import json

        json_path = output_dir / f"mixed_voice_results_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "threshold": threshold,
                        "timestamp": timestamp.isoformat(),
                    },
                    "results": results,
                    "summary": {
                        "total": total,
                        "passed": passed,
                        "failed": failed,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\n📊 JSON 结果已保存: {json_path}")


# --------------------------------------------------------------------------- #
#  CLI 入口
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="混杂声纹交叉测试")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="识别阈值 (default: 0.55)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="图表输出目录",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出模式",
    )
    parser.add_argument(
        "--regenerate",
        "-r",
        action="store_true",
        help="重新生成混合音频",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir) if args.output_dir else None
    mixed_voice_cross_test(args.threshold, output_path, args.verbose, args.regenerate)


if __name__ == "__main__":
    main()
