#!/usr/bin/env python3
"""声纹交叉测试 — 合并音频注册版本。

与原 cross_test.py 的区别:
  - 注册时将所有 speaker 的音频合并为一条音频（不永久保存）
  - 然后注册合并后的音频，只注册这一条

测试场景:
  注册: 每人一条合并后的音频（由 registration_segments 目录中所有片段合并）
  测试: 每人的 test_segments 目录中所有片段，每个片段单独测试
  裁剪: 超过 2 秒的音频只保留前 2 秒

说话人组:
  - John 组: John, John_USB, John_MeetingRoom, John_D_USB, John_D_USB_AEC（同一人，不同录制条件/处理）
  - Zhong 组: Zhong, Zhong_D_USB（同一人，不同录制条件）
  - 其他: Xixi, Frank, Qingqing（独立说话人）

用法:
    uv run python scripts/cross_test_merged.py
    uv run python scripts/cross_test_merged.py --noise asset/john/嘈杂环境测试.m4a
    uv run python scripts/cross_test_merged.py --snrs 20,15,10,5,0
    uv run python scripts/cross_test_merged.py --threshold 0.50
    uv run python scripts/cross_test_merged.py --output-dir outputs
"""

import logging
import pickle
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from src.wespeaker import WespeakerBest
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
from src.wespeaker.wespeaker import _apply_silero_vad, _crop_to_duration

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
    "John_D_USB": {
        "register_dir": "asset/john_d_usb/registration_segments",
        "test_segments_dir": "asset/john_d_usb/test_segments",
    },
    "John_D_USB_AEC": {
        "register_dir": "asset/john_d_usb_AEC/registration_segments",
        "test_segments_dir": "asset/john_d_usb_AEC/test_segments",
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
    "Zhong_D_USB": {
        "register_dir": "asset/zhong_d_usb/registration_segments",
        "test_segments_dir": "asset/zhong_d_usb/test_segments",
    },
}

# 同一人组映射
SAME_PERSON_GROUPS: dict[str, set[str]] = {
    "John": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_USB": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_MeetingRoom": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_D_USB": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_D_USB_AEC": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "Zhong": {"Zhong", "Zhong_D_USB"},
    "Zhong_D_USB": {"Zhong", "Zhong_D_USB"},
}


def is_same_person(speaker1: str, speaker2: str) -> bool:
    """判断两个说话人是否属于同一人（考虑同一人的不同录制条件）."""
    if speaker1 == speaker2:
        return True
    group1 = SAME_PERSON_GROUPS.get(speaker1, {speaker1})
    return speaker2 in group1


def merge_audio_files(
    audio_files: list[Path],
    output_path: Path,
    sample_rate: int = 16000,
) -> tuple[int, float]:
    """将多个音频文件合并为一个文件.

    Args:
        audio_files: 音频文件路径列表
        output_path: 输出文件路径
        sample_rate: 目标采样率

    Returns:
        (合并后的采样点数, 合并后的时长秒数)
    """
    all_waveforms = []

    for audio_file in audio_files:
        waveform, sr = torchaudio.load(audio_file)
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # 重采样到目标采样率
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        all_waveforms.append(waveform)

    # 拼接所有音频
    merged = torch.cat(all_waveforms, dim=1)

    # 保存合并后的音频
    torchaudio.save(str(output_path), merged, sample_rate)

    num_samples = merged.shape[1]
    duration = num_samples / sample_rate
    return num_samples, duration


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
    fig_height = max(16, len(row_labels) * 0.35)
    fig, ax = plt.subplots(figsize=(18, fig_height))

    im = ax.imshow(scores, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=1.0)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
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
                fontsize=9,
            )

    ax.set_title(
        f"声纹交叉识别矩阵 (阈值 = {threshold:.2f}) — 合并音频注册\n"
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("注册声纹", fontsize=14)
    ax.set_ylabel("测试音频", fontsize=14)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("相似度得分", rotation=270, labelpad=20, fontsize=13)

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
    ax.set_title("各说话人自识别得分统计 — 合并音频注册", fontsize=14)
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


def cross_test_merged(
    noise_path: str,
    snr_levels: list[float],
    threshold: float,
    output_dir: Path | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """执行交叉测试矩阵 — 使用合并音频注册.

    Args:
        noise_path: 噪声音频文件路径
        snr_levels: SNR 级别列表
        threshold: 识别阈值
        output_dir: 输出目录（可选）
        verbose: 详细输出模式
        debug: 调试模式
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)

    if output_dir is None:
        output_dir = Path("experiment_log")

    reporter = TerminalReporter(verbose=verbose, debug=debug)
    metrics = PerformanceMetrics()

    reporter.print_header(threshold, snr_levels, "合并音频注册")

    recognizer = WespeakerBest()
    recognizer.config = recognizer.config.__class__(
        **{
            **vars(recognizer.config),
            "sim_threshold": threshold,
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

    # 3. 注册所有说话人（合并音频方式）
    voiceprints: dict[str, torch.Tensor] = {}
    tmp_pk = Path("/tmp/voice_cross_merged.pkl")
    tmp_pk.parent.mkdir(parents=True, exist_ok=True)

    # 创建临时目录用于合并音频
    temp_dir = Path(tempfile.mkdtemp(prefix="cross_test_merged_"))

    registration_data: dict[str, dict] = {}

    for name, paths in SPEAKERS.items():
        reg_dir = paths["register_dir"]
        reporter.print_registration_start(name, reg_dir)

        metrics.start(f"registration_{name}")
        reg_diag = RegistrationDiagnostics(speaker=name)

        # 收集注册片段信息
        reg_path = Path(reg_dir)
        segment_files = sorted(reg_path.glob("*.wav")) + sorted(reg_path.glob("*.m4a"))

        if not segment_files:
            print(f"警告: {name} 的注册目录中没有音频文件")
            continue

        # 合并所有音频到一个临时文件
        merged_audio_path = temp_dir / f"{name}_merged.wav"
        num_samples, duration = merge_audio_files(segment_files, merged_audio_path)

        if verbose:
            print(f"  合并 {len(segment_files)} 个片段 → {duration:.2f}s ({num_samples} 采样点)")

        # 创建临时目录，只包含合并后的音频
        merged_dir = temp_dir / f"{name}_merged_dir"
        merged_dir.mkdir(exist_ok=True)
        merged_single_path = merged_dir / "merged.wav"

        # 将合并后的音频复制到临时目录（模拟单个文件）
        import shutil

        shutil.copy(merged_audio_path, merged_single_path)

        # 执行注册（只注册这一条合并后的音频）
        result = recognizer.enroll(str(merged_dir), noise_profile, str(tmp_pk), snr_levels)
        voiceprints[name] = result["embedding"]

        # 添加合并音频信息到诊断
        reg_diag.add_segment(
            "merged_audio",
            duration,
            16000,
            result["embedding"],
        )

        # 记录噪声注入效果
        for snr in snr_levels:
            reg_diag.record_noise_injection(
                snr_level=snr,
                original_rms=0.1,
                mixed_rms=0.1 * 10 ** (-snr / 20),
            )

        metrics.end(f"registration_{name}")
        registration_data[name] = reg_diag.to_dict()
        reporter.print_registration_summary(name, registration_data[name])

        if debug:
            reporter.print_debug_embedding(f"{name} embedding", result["embedding"])

    # 清理临时目录
    import shutil

    shutil.rmtree(temp_dir)

    # 4. 交叉识别矩阵
    SPEAKER_ORDER = [
        "John",
        "John_USB",
        "John_MeetingRoom",
        "John_D_USB",
        "John_D_USB_AEC",
        "Zhong",
        "Zhong_D_USB",
        "Xixi",
        "Frank",
        "Qingqing",
    ]

    ordered_speakers = [name for name in SPEAKER_ORDER if name in SPEAKERS]
    col_headers = [f"{name} 声纹" for name in ordered_speakers]
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

    col_names = list(SPEAKERS.keys())
    diagonal_scores: dict[str, list[float]] = {name: [] for name in SPEAKERS.keys()}
    col_order_map = {name: i for i, name in enumerate(SPEAKER_ORDER) if name in SPEAKERS}

    test_data_list: list[dict] = []

    for test_speaker, speaker_data in SPEAKERS.items():
        test_dir = Path(speaker_data["test_segments_dir"])
        test_files = sorted(test_dir.glob("*.wav"))

        for audio_file in test_files:
            label = audio_file.name
            row_label = f"{test_speaker}/{label}"
            row_scores: list[float] = []

            metrics.start(f"recognize_{row_label}")

            recog_diag = RecognitionDiagnostics(
                test_speaker=test_speaker,
                test_variant=label,
                threshold=threshold,
            )

            waveform, sr = torchaudio.load(audio_file)
            original_duration = waveform.shape[1] / sr
            waveform_mono = waveform.mean(dim=0)
            waveform_vad = _apply_silero_vad(waveform_mono, sr)
            vad_duration = waveform_vad.shape[0] / sr
            waveform_final = _crop_to_duration(waveform_vad, 2.0, sr)
            final_duration = waveform_final.shape[0] / sr

            rms_energy = float(waveform_final.norm())
            recog_diag.set_preprocessing_info(
                duration=final_duration,
                sample_rate=sr,
                rms_energy=rms_energy,
                original_duration=original_duration,
                vad_duration=vad_duration,
            )

            temp_audio_path = tmp_pk.parent / "temp_test_audio.wav"
            torchaudio.save(str(temp_audio_path), waveform_final.unsqueeze(0), sr)

            if verbose:
                row = f"{row_label:>30} |"

            for ref_name in ordered_speakers:
                ref_emb = voiceprints[ref_name]
                with open(tmp_pk, "wb") as f:
                    pickle.dump(ref_emb.cpu().numpy(), f)

                result = recognizer.recognize(str(temp_audio_path), str(tmp_pk))
                score = result["confidence"]
                is_match = result["is_recognized"]

                recog_diag.add_comparison(ref_name, float(score), is_match)
                row_scores.append(score)

                same_person = is_same_person(test_speaker, ref_name)

                if same_person:
                    diagonal_scores[test_speaker].append(score)
                    recog_diag.confidence = float(score)
                    if not is_match:
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
                    if is_match:
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

            test_case_dict = recog_diag.to_dict()
            test_case_dict["row_label"] = row_label
            test_cases.append(test_case_dict)

            test_data_list.append(
                {
                    "test_speaker": test_speaker,
                    "label": label,
                    "vad_duration": vad_duration,
                    "row_scores": row_scores,
                    "test_case_dict": test_case_dict,
                }
            )

            if verbose:
                print(row.rstrip(" |"))

    # 5. 排序
    test_data_list.sort(key=lambda x: (x["test_speaker"], -x["vad_duration"]))

    row_labels: list[str] = []
    scores_matrix: list[list[float]] = []

    for data in test_data_list:
        test_speaker = data["test_speaker"]
        label = data["label"]
        vad_duration = data["vad_duration"]
        row_label = f"{test_speaker}/{label} ({vad_duration:.2f}s)"
        row_labels.append(row_label)
        row_scores = data["row_scores"]
        reordered_scores = [row_scores[i] for i in sorted(col_order_map.values())]
        scores_matrix.append(reordered_scores)

    # 6. 总结
    total_tests = len(scores_matrix)
    passed_tests = total_tests - len(errors["false_accepts"]) - len(errors["false_rejects"])
    reporter.print_test_summary(total_tests, passed_tests, errors)

    # 7. 生成图表和报告
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now()

        scores_array = np.array(scores_matrix)
        ordered_col_names = [name for name in SPEAKER_ORDER if name in SPEAKERS]
        col_labels = [f"{name} 声纹" for name in ordered_col_names]

        heatmap_path = (
            output_dir / f"cross_test_merged_heatmap_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        )
        plot_heatmap(scores_array, row_labels, col_labels, threshold, heatmap_path)

        bar_path = (
            output_dir / f"cross_test_merged_summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        )
        plot_summary_bar(diagonal_scores, threshold, bar_path)

        report_data = {
            "meta": {
                "threshold": threshold,
                "snr_levels": snr_levels,
                "noise_path": str(noise_path),
                "registration_method": "merged_audio",
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

        md_gen = MarkdownReportGenerator(output_dir)
        md_path = md_gen.generate(report_data, timestamp)
        print(f"\n📄 诊断报告已保存: {md_path}")

        json_exporter = JsonDataExporter(output_dir)
        json_path = json_exporter.export(report_data, timestamp)
        print(f"📊 JSON 数据已导出: {json_path}")


# --------------------------------------------------------------------------- #
#  CLI 入口
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="声纹交叉测试 — 合并音频注册版本")
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
        if not list(test_dir.glob("*.wav")):
            print(f"错误: 测试片段目录中没有 .wav 文件: {test_dir}")
            sys.exit(1)
    if not Path(args.noise).is_file():
        print(f"错误: 噪声音频不存在: {args.noise}")
        sys.exit(1)

    output_path = Path(args.output_dir) if args.output_dir else None
    cross_test_merged(
        args.noise,
        snr_levels,
        args.threshold,
        output_path,
        args.verbose,
        args.debug,
    )


if __name__ == "__main__":
    main()
