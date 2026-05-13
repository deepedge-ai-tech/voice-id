#!/usr/bin/env python3
"""声纹交叉测试 — 7x7 识别矩阵 (John, John_USB, John_MeetingRoom, Xixi, Frank, Qingqing & Zhong)。

测试场景:
  注册: John, John_USB, John_MeetingRoom, Xixi, Frank, Qingqing, Zhong (使用 registration_segments 目录)
  测试: 每人的 test_segments 目录中所有片段，每个片段单独测试
  裁剪: 超过 2 秒的音频只保留前 2 秒

说话人组:
  - John 组: John, John_USB, John_MeetingRoom（同一人，不同录制条件）
  - 其他: Xixi, Frank, Qingqing, Zhong（独立说话人）

预期:
  - 正确匹配: John 组内相互匹配，其他人只匹配自己 → 通过
  - 正确拒绝: 不同人的测试片段 vs 其他人声纹 → 拒绝

用法:
    uv run python scripts/cross_test.py
    uv run python scripts/cross_test.py --noise asset/john/嘈杂环境测试.m4a
    uv run python scripts/cross_test.py --snrs 20,15,10,5,0
    uv run python scripts/cross_test.py --threshold 0.50
    uv run python scripts/cross_test.py --dynamic-threshold  # 启用动态阈值
    uv run python scripts/cross_test.py --dynamic-threshold --threshold-mode smooth
    uv run python scripts/cross_test.py --output-dir outputs  # 生成图表和报告
    uv run python scripts/cross_test.py --verbose  # 详细输出
    uv run python scripts/cross_test.py --debug  # 调试信息
"""

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker import WespeakerBest
from src.wespeaker.wespeaker import _apply_silero_vad, _crop_to_duration
from src.wespeaker.diagnostics import (
    PerformanceMetrics,
    RecognitionDiagnostics,
    RegistrationDiagnostics,
)
from src.wespeaker.reporters import (
    JsonDataExporter,
    MarkdownReportGenerator,
    TerminalReporter,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------- #
#  测试配置
# --------------------------------------------------------------------------- #

SPEAKERS = {
    "John": {
        "register_dir": "asset/john/registration_segments",
        "test_segments_dir": "asset/john/test_segments",
    },
    "John_USB": {
        "register_dir": "asset/john_usb/registration_segments",
        "test_segments_dir": "asset/john_usb/test_segments",
    },
    "John_MeetingRoom": {
        "register_dir": "asset/john_metting_room/registration_segments",
        "test_segments_dir": "asset/john_metting_room/test_segments",
    },
    "Xixi": {
        "register_dir": "asset/xixi/registration_segments",
        "test_segments_dir": "asset/xixi/test_segments",
    },
    "Frank": {
        "register_dir": "asset/frank/registration_segments",
        "test_segments_dir": "asset/frank/test_segments",
    },
    "Qingqing": {
        "register_dir": "asset/qingqing/registration_segments",
        "test_segments_dir": "asset/qingqing/test_segments",
    },
    "Zhong": {
        "register_dir": "asset/zhong/registration_segments",
        "test_segments_dir": "asset/zhong/test_segments",
    },
}

# 同一人组映射：三个 John 变体属于同一人
SAME_PERSON_GROUPS: dict[str, set[str]] = {
    "John": {"John", "John_USB", "John_MeetingRoom"},
    "John_USB": {"John", "John_USB", "John_MeetingRoom"},
    "John_MeetingRoom": {"John", "John_USB", "John_MeetingRoom"},
}


def is_same_person(speaker1: str, speaker2: str) -> bool:
    """判断两个说话人是否属于同一人（考虑同一人的不同录制条件）."""
    if speaker1 == speaker2:
        return True
    # 检查是否在同一人组中
    group1 = SAME_PERSON_GROUPS.get(speaker1, {speaker1})
    return speaker2 in group1


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
    verbose: bool = False,
    debug: bool = False,
    enable_dynamic_threshold: bool = False,
    threshold_mode: str = "segmented",
) -> None:
    """执行 7x7 交叉测试矩阵，集成诊断数据和报告生成.

    Args:
        noise_path: 噪声音频文件路径
        snr_levels: SNR 级别列表
        threshold: 识别阈值（固定阈值模式下的默认值）
        output_dir: 输出目录（可选）
        verbose: 详细输出模式
        debug: 调试模式
        enable_dynamic_threshold: 启用基于 VAD 时长的动态阈值
        threshold_mode: 动态阈值模式 ("segmented" 或 "smooth")
    """
    # 配置日志级别
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)

    # 初始化 reporter 和 metrics
    if output_dir is None:
        output_dir = Path("experiment_log")

    reporter = TerminalReporter(verbose=verbose, debug=debug)
    metrics = PerformanceMetrics()

    threshold_type = "动态" if enable_dynamic_threshold else "固定"
    reporter.print_header(threshold, snr_levels, threshold_type)

    recognizer = WespeakerBest()
    recognizer.config = recognizer.config.__class__(
        **{
            **vars(recognizer.config),
            "sim_threshold": threshold,
            "enable_dynamic_threshold": enable_dynamic_threshold,
            "dynamic_threshold_mode": threshold_mode,
        }
    )

    # 1. 加载模型
    metrics.start("model_load")
    recognizer._client._ensure_model()
    metrics.end("model_load")

    # 2. 提取噪声 profile
    if verbose:
        print(f"\n提取噪声 profile: {noise_path}")
    noise_profile = WespeakerBest.extract_noise_profile(noise_path)
    if verbose:
        print(f"  噪声长度: {len(noise_profile) / 16000:.1f}s")

    # 3. 注册所有说话人（集成诊断）
    voiceprints: dict[str, torch.Tensor] = {}
    tmp_pk = Path("/tmp/voice_cross.pkl")
    tmp_pk.parent.mkdir(parents=True, exist_ok=True)

    registration_data: dict[str, dict] = {}

    for name, paths in SPEAKERS.items():
        reg_dir = paths["register_dir"]
        reporter.print_registration_start(name, reg_dir)

        metrics.start(f"registration_{name}")
        reg_diag = RegistrationDiagnostics(speaker=name)

        # 收集注册片段信息
        reg_path = Path(reg_dir)
        segment_files = sorted(reg_path.glob("*.wav")) + sorted(reg_path.glob("*.m4a"))

        # 执行注册
        result = recognizer.enroll(reg_dir, noise_profile, str(tmp_pk), snr_levels)
        voiceprints[name] = result["embedding"]

        # 添加片段信息到诊断（使用实际的片段 embeddings）
        fragment_embeddings = result.get("fragment_embeddings", [])
        for i, seg_file in enumerate(segment_files):
            import torchaudio

            waveform, sr = torchaudio.load(seg_file)
            duration = waveform.shape[1] / sr
            # 使用对应的片段 embedding（如果可用）
            if i < len(fragment_embeddings):
                reg_diag.add_segment(seg_file.name, duration, sr, fragment_embeddings[i])
            else:
                # 回退到使用均值 embedding（向后兼容）
                reg_diag.add_segment(seg_file.name, duration, sr, result["embedding"])

        # 记录噪声注入效果（模拟）
        for snr in snr_levels:
            reg_diag.record_noise_injection(
                snr_level=snr,
                original_rms=0.1,  # 简化值
                mixed_rms=0.1 * 10 ** (-snr / 20),  # 简化计算
            )

        metrics.end(f"registration_{name}")
        registration_data[name] = reg_diag.to_dict()
        reporter.print_registration_summary(name, registration_data[name])

        if debug:
            reporter.print_debug_embedding(f"{name} embedding", result["embedding"])

    # 4. 交叉识别矩阵（集成诊断）
    col_headers = [f"{name} 声纹" for name in SPEAKERS.keys()]
    col_width = 12
    header = f"{'':>14} | " + " | ".join(f"{h:>{col_width}}" for h in col_headers)
    sep = "-" * len(header)

    if verbose:
        print(f"\n{'=' * len(header)}")
        print("  交叉识别矩阵 (阈值 = {:.2f})".format(threshold))
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)

    all_passed = True
    test_cases: list[dict] = []
    errors: dict = {"false_accepts": [], "false_rejects": []}

    # 收集用于可视化的数据
    row_labels: list[str] = []
    col_names = list(SPEAKERS.keys())
    scores_matrix: list[list[float]] = []
    diagonal_scores: dict[str, list[float]] = {name: [] for name in SPEAKERS.keys()}

    for test_speaker, speaker_data in SPEAKERS.items():
        test_dir = Path(speaker_data["test_segments_dir"])
        test_files = sorted(test_dir.glob("*.wav"))

        for audio_file in test_files:
            label = audio_file.name
            row_label = f"{test_speaker}/{label}"
            row_labels.append(row_label)
            row_scores: list[float] = []

            metrics.start(f"recognize_{row_label}")

            # 创建识别诊断对象
            recog_diag = RecognitionDiagnostics(
                test_speaker=test_speaker,
                test_variant=label,
                threshold=threshold,
            )

            # 加载测试音频获取预处理信息
            import torchaudio

            waveform, sr = torchaudio.load(audio_file)
            original_duration = waveform.shape[1] / sr

            # 转换为单声道
            waveform_mono = waveform.mean(dim=0)

            # 先用 Silero VAD 去除静音
            waveform_vad = _apply_silero_vad(waveform_mono, sr)
            vad_duration = waveform_vad.shape[0] / sr

            # 再裁剪到 2 秒
            waveform_final = _crop_to_duration(waveform_vad, 2.0, sr)
            final_duration = waveform_final.shape[0] / sr

            rms_energy = float(waveform_final.norm())
            # 设置预处理信息（包含VAD前后的时长）
            recog_diag.set_preprocessing_info(
                duration=final_duration,
                sample_rate=sr,
                rms_energy=rms_energy,
                original_duration=original_duration,
                vad_duration=vad_duration,
            )

            # 保存裁剪后的音频到临时文件用于识别
            temp_audio_path = tmp_pk.parent / "temp_test_audio.wav"
            torchaudio.save(str(temp_audio_path), waveform_final.unsqueeze(0), sr)

            if verbose:
                row = f"{row_label:>30} |"

            for ref_name, ref_emb in voiceprints.items():
                with open(tmp_pk, "wb") as f:
                    pickle.dump(ref_emb.cpu().numpy(), f)

                result = recognizer.recognize(str(temp_audio_path), str(tmp_pk))
                score = result["confidence"]
                is_match = result["is_recognized"]

                # 添加比较结果到诊断
                recog_diag.add_comparison(ref_name, float(score), is_match)

                row_scores.append(score)

                # 判断是否为同一人（考虑同一人的不同录制条件）
                same_person = is_same_person(test_speaker, ref_name)

                if same_person:
                    # 对角线方向（同一人或同一人的不同变体）
                    diagonal_scores[test_speaker].append(score)
                    # 设置 confidence 为正确说话人的得分
                    recog_diag.confidence = float(score)
                    # 这是正确的匹配
                    if not is_match:
                        # 误拒绝
                        recog_diag.record_false_negative(float(score))
                        errors["false_rejects"].append(
                            {
                                "test_speaker": test_speaker,
                                "test_variant": label,
                                "ref_speaker": ref_name,
                                "score": float(score),
                                "threshold_distance": threshold - float(score),
                            }
                        )
                        all_passed = False
                else:
                    # 这是不同的说话人，应该拒绝
                    if is_match:
                        # 误接受
                        recog_diag.record_false_positive(ref_name, float(score))
                        errors["false_accepts"].append(
                            {
                                "test_speaker": test_speaker,
                                "test_variant": label,
                                "mistaken_as": ref_name,
                                "score": float(score),
                                "threshold_distance": float(score) - threshold,
                            }
                        )
                        all_passed = False

                if verbose:
                    mark = "✅" if is_match else "❌"
                    ok = is_match if same_person else not is_match
                    status = "✅" if ok else "⚠️ "
                    row += f" {score:.4f} {mark} {status} |"
                else:
                    reporter.print_recognition_progress(row_label, ref_name, float(score), is_match)

            metrics.end(f"recognize_{row_label}")

            # 保存测试用例数据
            test_case_dict = recog_diag.to_dict()
            test_case_dict["row_label"] = row_label
            test_cases.append(test_case_dict)

            scores_matrix.append(row_scores)

            if verbose:
                print(row.rstrip(" |"))

    # 5. 总结
    total_tests = len(scores_matrix)
    passed_tests = total_tests - len(errors["false_accepts"]) - len(errors["false_rejects"])
    reporter.print_test_summary(total_tests, passed_tests, errors)

    # 6. 生成图表和报告
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now()

        scores_array = np.array(scores_matrix)
        col_labels = [f"{name} 声纹" for name in col_names]

        # 生成热力图
        heatmap_path = output_dir / f"cross_test_heatmap_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        plot_heatmap(scores_array, row_labels, col_labels, threshold, heatmap_path)

        # 生成柱状图
        bar_path = output_dir / f"cross_test_summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        plot_summary_bar(diagonal_scores, threshold, bar_path)

        # 生成诊断报告
        report_data = {
            "meta": {
                "threshold": threshold,
                "snr_levels": snr_levels,
                "noise_path": str(noise_path),
            },
            "registration": registration_data,
            "recognition": {
                "test_cases": test_cases,
                "errors": errors,
                "performance": {
                    "total_time": sum(metrics.get_timings().values()),
                    "timings": metrics.get_summary()["operations"],
                },
            },
        }

        # Markdown 报告
        md_gen = MarkdownReportGenerator(output_dir)
        md_path = md_gen.generate(report_data, timestamp)
        print(f"\n📄 诊断报告已保存: {md_path}")

        # JSON 数据导出
        json_exporter = JsonDataExporter(output_dir)
        json_path = json_exporter.export(report_data, timestamp)
        print(f"📊 JSON 数据已导出: {json_path}")


# --------------------------------------------------------------------------- #
#  CLI 入口
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="声纹交叉测试 — 7x7 识别矩阵 (John, John_USB, John_MeetingRoom, Xixi, Frank, Qingqing & Zhong)")
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
        help="图表和报告输出目录（不指定则不生成）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出模式",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="调试模式（打印 embedding 调试信息）",
    )
    parser.add_argument(
        "--dynamic-threshold",
        action="store_true",
        help="启用基于 VAD 时长的动态阈值",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["segmented", "smooth"],
        default="segmented",
        help="动态阈值模式 (default: segmented)",
    )
    args = parser.parse_args()

    snr_levels = [float(x.strip()) for x in args.snrs.split(",")]

    # 验证文件存在
    for speaker_name, speaker_data in SPEAKERS.items():
        reg_dir = Path(speaker_data["register_dir"])
        if not reg_dir.is_dir():
            print(f"错误: 注册目录不存在: {reg_dir}")
            sys.exit(1)
        test_dir = Path(speaker_data["test_segments_dir"])
        if not test_dir.is_dir():
            print(f"错误: 测试片段目录不存在: {test_dir}")
            sys.exit(1)
        # 检查是否有 wav 文件
        if not list(test_dir.glob("*.wav")):
            print(f"错误: 测试片段目录中没有 .wav 文件: {test_dir}")
            sys.exit(1)
    if not Path(args.noise).is_file():
        print(f"错误: 噪声音频不存在: {args.noise}")
        sys.exit(1)

    output_path = Path(args.output_dir) if args.output_dir else None
    cross_test(
        args.noise,
        snr_levels,
        args.threshold,
        output_path,
        args.verbose,
        args.debug,
        args.dynamic_threshold,
        args.threshold_mode,
    )


if __name__ == "__main__":
    main()
