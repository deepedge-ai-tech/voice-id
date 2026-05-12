#!/usr/bin/env python3
"""声纹交叉测试 — 6x6 识别矩阵。

测试场景:
  注册: Frank, John, Michael, Zhong, Xixi, Qingqing
  测试: 每人原始 + 4种变体音频

预期:
  - 正确匹配: 同一人的测试音频 vs 自己的声纹 → 通过
  - 正确拒绝: 不同人的测试音频 vs 其他人声纹 → 拒绝

用法:
    uv run python scripts/cross_test.py
    uv run python scripts/cross_test.py --noise asset/john/嘈杂环境测试.m4a
    uv run python scripts/cross_test.py --snrs 20,15,10,5,0
    uv run python scripts/cross_test.py --threshold 0.50
    uv run python scripts/cross_test.py --output-dir outputs  # 生成图表
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker import WespeakerBest

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------- #
#  测试配置
# --------------------------------------------------------------------------- #

SPEAKERS = {
    "Frank": {
        "register_dir": "asset/frank/registration_segments",
        "test_audios": {
            "原始": "asset/frank/frank 测试.m4a",
            "电话音效": "asset/frank/frank 测试_eq_phone.m4a",
            "大厅回音": "asset/frank/frank 测试_reverb_hall.m4a",
            "低码率": "asset/frank/frank 测试_low_bitrate.m4a",
            "底噪": "asset/frank/frank 测试_noise_hiss.m4a",
        },
    },
    "John": {
        "register_dir": "asset/john/registration_segments",
        "test_audios": {
            "原始": "asset/john/安静环境测试测试.m4a",
            "嘈杂环境": "asset/john/嘈杂环境测试.m4a",
            "电话音效": "asset/john/安静环境测试测试_eq_phone.m4a",
            "大厅回音": "asset/john/安静环境测试测试_reverb_hall.m4a",
            "低码率": "asset/john/安静环境测试测试_low_bitrate.m4a",
            "底噪": "asset/john/安静环境测试测试_noise_hiss.m4a",
        },
    },
    "Michael": {
        "register_dir": "asset/michael/registration_segments",
        "test_audios": {
            "原始": "asset/michael/测试.wav",
            "电话音效": "asset/michael/测试_eq_phone.m4a",
            "大厅回音": "asset/michael/测试_reverb_hall.m4a",
            "低码率": "asset/michael/测试_low_bitrate.m4a",
            "底噪": "asset/michael/测试_noise_hiss.m4a",
        },
    },
    "Zhong": {
        "register_dir": "asset/zhong/registration_segments",
        "test_audios": {
            "原始": "asset/zhong/测试.wav",
            "电话音效": "asset/zhong/测试_eq_phone.m4a",
            "大厅回音": "asset/zhong/测试_reverb_hall.m4a",
            "低码率": "asset/zhong/测试_low_bitrate.m4a",
            "底噪": "asset/zhong/测试_noise_hiss.m4a",
        },
    },
    "Xixi": {
        "register_dir": "asset/xixi/registration_segments",
        "test_audios": {
            "原始": "asset/xixi/测试.wav",
            "电话音效": "asset/xixi/测试_eq_phone.m4a",
            "大厅回音": "asset/xixi/测试_reverb_hall.m4a",
            "低码率": "asset/xixi/测试_low_bitrate.m4a",
            "底噪": "asset/xixi/测试_noise_hiss.m4a",
        },
    },
    "Qingqing": {
        "register_dir": "asset/qingqing/registration_segments",
        "test_audios": {
            "原始": "asset/qingqing/测试.wav",
            "电话音效": "asset/qingqing/测试_eq_phone.m4a",
            "大厅回音": "asset/qingqing/测试_reverb_hall.m4a",
            "低码率": "asset/qingqing/测试_low_bitrate.m4a",
            "底噪": "asset/qingqing/测试_noise_hiss.m4a",
        },
    },
}


# --------------------------------------------------------------------------- #
#  可视化函数
# --------------------------------------------------------------------------- #


def plot_heatmap(
    scores: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    threshold: float,
    output_path: Path | None = None,
) -> None:
    """绘制相似度热力图."""
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(scores, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=1.0)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=9)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text_color = "white" if scores[i, j] < threshold else "black"
            text = ax.text(
                j,
                i,
                f"{scores[i, j]:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    ax.set_title(
        f"声纹交叉识别矩阵 (阈值 = {threshold:.2f})\n"
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=14,
        pad=20,
    )
    ax.set_xlabel("注册声纹", fontsize=12)
    ax.set_ylabel("测试音频", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("相似度得分", rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n📊 热力图已保存: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_summary_bar(
    diagonal_scores: dict[str, list[float]],
    threshold: float,
    output_path: Path | None = None,
) -> None:
    """绘制各说话人自识别得分柱状图."""
    speakers = list(diagonal_scores.keys())
    avg_scores = [np.mean(scores) for scores in diagonal_scores.values()]
    min_scores = [np.min(scores) for scores in diagonal_scores.values()]

    x = np.arange(len(speakers))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(x, avg_scores, width, label="平均得分", capsize=5, color="#4CAF50")

    for i, (avg, min_val) in enumerate(zip(avg_scores, min_scores)):
        ax.errorbar(i, avg, yerr=avg - min_val, fmt="none", ecolor="black", capsize=5)

    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=2, label=f"阈值 ({threshold})")

    ax.set_xlabel("说话人", fontsize=12)
    ax.set_ylabel("相似度得分", fontsize=12)
    ax.set_title("各说话人自识别得分统计", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(speakers)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    for bar, score in zip(bars, avg_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"📊 柱状图已保存: {output_path}")
    else:
        plt.show()

    plt.close()


# --------------------------------------------------------------------------- #
#  交叉测试
# --------------------------------------------------------------------------- #


def cross_test(
    noise_path: str,
    snr_levels: list[float],
    threshold: float,
    output_dir: Path | None = None,
) -> None:
    """执行 6x6 交叉测试矩阵，可选导出图表."""
    recognizer = WespeakerBest()
    recognizer.config = recognizer.config.__class__(
        **{**vars(recognizer.config), "sim_threshold": threshold}
    )

    # 1. 加载模型
    recognizer._client._ensure_model()

    # 2. 提取噪声 profile
    print(f"提取噪声 profile: {noise_path}")
    noise_profile = WespeakerBest.extract_noise_profile(noise_path)
    print(f"  噪声长度: {len(noise_profile) / 16000:.1f}s\n")

    # 3. 注册所有说话人
    voiceprints: dict[str, torch.Tensor] = {}
    tmp_pk = Path("/tmp/voice_cross.pkl")
    tmp_pk.parent.mkdir(parents=True, exist_ok=True)

    for name, paths in SPEAKERS.items():
        reg_dir = paths["register_dir"]
        print(f"[注册] {name}: {reg_dir}")
        result = recognizer.enroll(reg_dir, noise_profile, str(tmp_pk), snr_levels)
        voiceprints[name] = result["embedding"]
        print(f"  embedding 维度: {result['embedding_dim']}\n")

    # 4. 交叉识别矩阵
    col_headers = [f"{name} 声纹" for name in SPEAKERS.keys()]
    col_width = 12
    header = f"{'':>14} | " + " | ".join(f"{h:>{col_width}}" for h in col_headers)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  交叉识别矩阵 (阈值 = {:.2f})".format(threshold))
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    all_passed = True

    # 收集用于可视化的数据
    row_labels: list[str] = []
    col_names = list(SPEAKERS.keys())
    scores_matrix: list[list[float]] = []
    diagonal_scores: dict[str, list[float]] = {name: [] for name in SPEAKERS.keys()}

    for test_speaker, speaker_data in SPEAKERS.items():
        for label, audio_path in speaker_data["test_audios"].items():
            row_label = f"{test_speaker}/{label}"
            row_labels.append(row_label)
            row_scores: list[float] = []

            row = f"{row_label:>14} |"
            for ref_name, ref_emb in voiceprints.items():
                with open(tmp_pk, "wb") as f:
                    pickle.dump(ref_emb.cpu().numpy(), f)

                result = recognizer.recognize(audio_path, str(tmp_pk))
                score = result["confidence"]
                is_match = result["is_recognized"]
                mark = "✅" if is_match else "❌"

                row_scores.append(score)

                if test_speaker == ref_name:
                    diagonal_scores[test_speaker].append(score)

                ok = is_match if test_speaker == ref_name else not is_match
                if not ok:
                    all_passed = False

                status = "✅" if ok else "⚠️ "
                row += f" {score:.4f} {mark} {status} |"

            scores_matrix.append(row_scores)
            print(row.rstrip(" |"))

    # 5. 总结
    print()
    if all_passed:
        print("✅ 所有测试通过 — 正确匹配且正确拒绝")
    else:
        print("⚠️  存在测试未通过 — 请检查阈值或注册质量")

    # 6. 生成图表
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        scores_array = np.array(scores_matrix)
        col_labels = [f"{name} 声纹" for name in col_names]

        heatmap_path = output_dir / f"cross_test_heatmap_{timestamp}.png"
        plot_heatmap(scores_array, row_labels, col_labels, threshold, heatmap_path)

        bar_path = output_dir / f"cross_test_summary_{timestamp}.png"
        plot_summary_bar(diagonal_scores, threshold, bar_path)


# --------------------------------------------------------------------------- #
#  CLI 入口
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="声纹交叉测试 — 6x6 识别矩阵")
    parser.add_argument(
        "--noise",
        default="asset/john/嘈杂环境测试.m4a",
        help="噪声音频文件（用于注册时噪声注入）",
    )
    parser.add_argument(
        "--snrs",
        default="20,15,10,5,0",
        help="SNR 级别，逗号分隔 (default: 20,15,10,5,0)",
    )
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
        help="图表输出目录（不指定则不生成图表）",
    )
    args = parser.parse_args()

    snr_levels = [float(x.strip()) for x in args.snrs.split(",")]

    # 验证文件存在
    for speaker_name, speaker_data in SPEAKERS.items():
        reg_dir = Path(speaker_data["register_dir"])
        if not reg_dir.is_dir():
            print(f"错误: 注册目录不存在: {reg_dir}")
            sys.exit(1)
        for label, audio_path in speaker_data["test_audios"].items():
            if not Path(audio_path).is_file():
                print(f"错误: 测试音频不存在: {audio_path}")
                sys.exit(1)
    if not Path(args.noise).is_file():
        print(f"错误: 噪声音频不存在: {args.noise}")
        sys.exit(1)

    output_path = Path(args.output_dir) if args.output_dir else None
    cross_test(args.noise, snr_levels, args.threshold, output_path)


if __name__ == "__main__":
    main()
