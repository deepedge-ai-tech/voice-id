#!/usr/bin/env python3
"""Voice-ID 声纹识别自动研究 - 训练/实验主程序。

这是 AI 代理唯一修改的文件。所有可调参数都在文件顶部。

设计理念:
  - prepare.py 包含固定常量和工具函数（不修改）
  - training.py 是代理修改的唯一文件（参数、实验逻辑）
  - 每次实验运行固定时间，输出到 outputs/experiments/

可调参数:
  - sim_threshold: 相似度识别阈值
  - verify_crop_mode: 验证音频裁剪模式 (full_utterance/tail_window/head_window)
  - verify_buffer_keep_secs: 验证 buffer 最大保留时长
  - verify_window_secs: 滑动窗口/裁剪窗口长度
  - enrollment_segment_secs: 注册时分段长度
  - enable_vad: 是否启用 VAD 去静音
  - vad_rms_threshold: VAD 能量阈值
  - noise_injection_snrs: 注册噪声注入 SNR 级别

评估指标:
  - FAR (False Accept Rate): 误接受率，越低越好
  - FRR (False Reject Rate): 误拒绝率，越低越好
  - short_audio_confidence: < 0.6s 音频平均置信度，越高越好
  - overall_accuracy: 总体准确率，越高越好
  - EER (Equal Error Rate): 等错误率，越低越好
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.wespeaker import WespeakerBest
from src.wespeaker.wespeaker import _apply_silero_vad, _crop_to_duration
from src.wespeaker.diagnostics import (
    PerformanceMetrics,
    RecognitionDiagnostics,
    RegistrationDiagnostics,
)
from src.wespeaker.reporters import TerminalReporter

# 导入 prepare.py 中的固定常量和工具函数
from prepare import (
    SPEAKERS,
    is_same_person,
    ExperimentConfig,
    ExperimentMetrics,
    ExperimentResult,
    setup_output_dirs,
    load_experiment_log,
    save_experiment_log,
    load_best_config,
    save_best_config,
    calculate_improvement,
    format_improvement_summary,
    OUTPUT_ROOT,
    EXPERIMENTS_DIR,
    BEST_CONFIG_PATH,
    SPEAKER_ORDER,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================================ #
#  可调参数 - AI 代理修改此区域
# ============================================================================ #
# 代理提示：这是你可以修改的参数区域。所有参数都会影响实验结果。
# 根据上次实验结果调整这些参数，然后运行实验。
# ---------------------------------------------------------------------------- #

# ---- 识别参数 ----
SIM_THRESHOLD = 0.55  # 相似度阈值 [0.20 - 0.75]，降低可减少误拒绝但增加误接受
VERIFY_CROP_MODE = "full_utterance"  # 裁剪模式: full_utterance, tail_window, head_window
VERIFY_BUFFER_KEEP_SECS = 60.0  # buffer 最大保留时长 [2.0 - 60.0]，60.0 表示不截断
VERIFY_WINDOW_SECS = 1.0  # 滑动窗口长度 [0.3 - 3.0]

# ---- 注册参数 ----
ENROLLMENT_SEGMENT_SECS = 1.0  # 注册分段长度 [0.5 - 3.0]

# ---- VAD 参数 ----
ENABLE_VAD = False  # 启用 VAD 去静音 (True/False)，实验表明 False 得分更高
VAD_RMS_THRESHOLD = 0.002  # VAD 能量阈值 [0.001 - 0.02]

# ---- 噪声注入 ----
NOISE_INJECTION_SNRS = [20.0, 15.0, 10.0, 5.0, 0.0]  # SNR 级别列表
NOISE_PATH = "asset/john/嘈杂环境测试.m4a"  # 噪声音频路径

# ---- 滑动窗口参数 ----
SLIDING_WINDOW_SECS = 0.6  # 滑动窗口长度 [0.3 - 3.0]
SLIDING_HOP_SECS = 0.2  # 滑动步长 [0.1 - 1.0]

# ---- 输出控制 ----
VERBOSE = False  # 详细输出模式
DEBUG = False  # 调试模式

# ---------------------------------------------------------------------------- #
#  可调参数区域结束
# ============================================================================ #

# ---- 固定参数（不修改）---- #
ENABLE_SCORE_COMPENSATION = False  # 分数补偿固定关闭
SCORE_COMPENSATION_TARGET_DURATION = 2.0  # 固定值


# ============================================================================ #
#  辅助函数：合并音频
# ============================================================================ #


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
    import torchaudio

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


# ============================================================================ #
#  可视化函数
# ============================================================================ #


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
            ax.text(
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
        logger.info(f"热力图已保存: {output_path}")

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
        logger.info(f"柱状图已保存: {output_path}")

    plt.close()


# --------------------------------------------------------------------------- #
#  交叉测试
# --------------------------------------------------------------------------- #


# ============================================================================ #
#  交叉测试函数
# ============================================================================ #


def cross_test(
    config: ExperimentConfig,
    experiment_id: str,
) -> ExperimentResult:
    """执行交叉测试矩阵，返回实验结果.

    Args:
        config: 实验配置
        experiment_id: 实验 ID

    Returns:
        ExperimentResult: 实验结果
    """
    # 配置日志级别
    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif config.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # 创建实验输出目录
    experiment_dir = EXPERIMENTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    reporter = TerminalReporter(verbose=config.verbose, debug=config.debug)
    metrics = PerformanceMetrics()

    mode = "分数补偿" if config.enable_score_compensation else "固定"
    reporter.print_header(config.sim_threshold, config.noise_injection_snrs, mode)

    # 使用 pyannote.audio 模型
    recognizer = WespeakerBest(model_path="pyannote/wespeaker-voxceleb-resnet34-LM")
    # 只设置 BestConfig 支持的参数
    from src.wespeaker.best import BestConfig

    recognizer.config = BestConfig(
        sim_threshold=config.sim_threshold,
        enable_score_compensation=config.enable_score_compensation,
        score_compensation_target_duration=config.score_compensation_target_duration,
        verify_crop_mode=config.verify_crop_mode,
        verify_buffer_keep_secs=config.verify_buffer_keep_secs,
        verify_window_secs=config.verify_window_secs,
        enrollment_segment_secs=config.enrollment_segment_secs,
        enable_vad=config.enable_vad,
        vad_rms_threshold=config.vad_rms_threshold,
        noise_injection_snrs=tuple(config.noise_injection_snrs),
        sliding_window_secs=SLIDING_WINDOW_SECS,
        sliding_hop_secs=SLIDING_HOP_SECS,
    )

    # 1. 加载模型
    metrics.start("model_load")
    recognizer._client._ensure_model()
    metrics.end("model_load")

    # 2. 提取噪声 profile
    if config.verbose:
        logger.info(f"提取噪声 profile: {config.noise_path}")
    noise_profile = WespeakerBest.extract_noise_profile(config.noise_path)
    if config.verbose:
        logger.info(f"  噪声长度: {len(noise_profile) / 16000:.1f}s")

    # 3. 注册所有说话人（合并音频方式）
    voiceprints: dict[str, torch.Tensor] = {}

    # 创建临时目录用于合并音频和临时文件
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp(prefix="voice_id_merged_"))
    tmp_pk = temp_dir / "temp_voice.pkl"

    registration_data: dict[str, dict] = {}

    for name, paths in SPEAKERS.items():
        reg_dir = paths["register_dir"]
        reporter.print_registration_start(name, reg_dir)

        metrics.start(f"registration_{name}")
        reg_diag = RegistrationDiagnostics(speaker=name)

        reg_path = Path(reg_dir)
        segment_files = sorted(reg_path.glob("*.wav")) + sorted(reg_path.glob("*.m4a"))

        if not segment_files:
            logger.warning(f"{name} 的注册目录中没有音频文件")
            continue

        # 合并所有音频到一个临时文件
        merged_audio_path = temp_dir / f"{name}_merged.wav"
        num_samples, duration = merge_audio_files(segment_files, merged_audio_path)

        if config.verbose:
            logger.info(f"  合并 {len(segment_files)} 个片段 → {duration:.2f}s ({num_samples} 采样点)")

        # 创建临时目录，只包含合并后的音频
        merged_dir = temp_dir / f"{name}_merged_dir"
        merged_dir.mkdir(exist_ok=True)
        merged_single_path = merged_dir / "merged.wav"

        # 将合并后的音频复制到临时目录
        shutil.copy(merged_audio_path, merged_single_path)

        # 执行注册（只注册这一条合并后的音频）
        result = recognizer.enroll(str(merged_dir), noise_profile, str(tmp_pk), config.noise_injection_snrs)
        voiceprints[name] = result["embedding"]

        # 添加合并音频信息到诊断
        reg_diag.add_segment(
            "merged_audio",
            duration,
            16000,
            result["embedding"],
        )

        # 记录噪声注入效果
        for snr in config.noise_injection_snrs:
            reg_diag.record_noise_injection(
                snr_level=snr,
                original_rms=0.1,
                mixed_rms=0.1 * 10 ** (-snr / 20),
            )

        metrics.end(f"registration_{name}")
        registration_data[name] = reg_diag.to_dict()
        reporter.print_registration_summary(name, registration_data[name])

        if config.debug:
            reporter.print_debug_embedding(f"{name} embedding", result["embedding"])

    # 4. 交叉识别矩阵
    ordered_speakers = [name for name in SPEAKER_ORDER if name in SPEAKERS]
    col_headers = [f"{name} 声纹" for name in ordered_speakers]
    col_width = 12
    header = f"{'':>14} | " + " | ".join(f"{h:>{col_width}}" for h in col_headers)
    sep = "-" * len(header)

    if config.verbose:
        print(f"\n{'=' * len(header)}")
        print(f"  交叉识别矩阵 (阈值 = {config.sim_threshold:.2f})")
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
                threshold=config.sim_threshold,
            )

            import torchaudio

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

            temp_audio_path = temp_dir / "temp_test_audio.wav"
            torchaudio.save(str(temp_audio_path), waveform_final.unsqueeze(0), sr)

            if config.verbose:
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
                                "threshold_distance": config.sim_threshold - float(score),
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
                                "threshold_distance": float(score) - config.sim_threshold,
                            }
                        )
                        all_passed = False

                if config.verbose:
                    mark = "✅" if is_match else "❌"
                    ok = is_match if same_person else not is_match
                    status = "✅" if ok else "⚠️ "
                    row += f" {score:.4f} {mark} {status} |"
                else:
                    reporter.print_recognition_progress(row_label, ref_name, float(score), is_match)

            metrics.end(f"recognize_{row_label}")

            test_case_dict = recog_diag.to_dict()
            test_case_dict["row_label"] = row_label
            test_case_dict["test_speaker"] = test_speaker
            test_case_dict["ref_speaker"] = ref_name
            test_case_dict["score"] = float(score)
            test_case_dict["vad_duration"] = vad_duration
            test_case_dict["is_match"] = is_match
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

            if config.verbose:
                print(row.rstrip(" |"))

    # 5. 排序和汇总
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

    total_tests = len(scores_matrix)
    passed_tests = total_tests - len(errors["false_accepts"]) - len(errors["false_rejects"])
    reporter.print_test_summary(total_tests, passed_tests, errors)

    # 6. 计算实验指标
    exp_metrics = ExperimentMetrics.from_results(test_cases, config.sim_threshold)

    # 7. 生成图表和报告
    timestamp = datetime.now()
    scores_array = np.array(scores_matrix)
    ordered_col_names = [name for name in SPEAKER_ORDER if name in SPEAKERS]
    col_labels = [f"{name} 声纹" for name in ordered_col_names]

    # 生成热力图
    heatmap_path = experiment_dir / f"heatmap.png"
    plot_heatmap(scores_array, row_labels, col_labels, config.sim_threshold, heatmap_path)

    # 生成柱状图
    bar_path = experiment_dir / f"summary_bar.png"
    plot_summary_bar(diagonal_scores, config.sim_threshold, bar_path)

    # 保存配置和指标
    config_path = experiment_dir / "config.json"
    config.save(config_path)

    metrics_path = experiment_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(exp_metrics.to_dict(), f, indent=2, ensure_ascii=False)

    # 保存详细结果
    report_data = {
        "meta": {
            "threshold": config.sim_threshold,
            "snr_levels": config.noise_injection_snrs,
            "noise_path": str(config.noise_path),
            "registration_method": "merged_audio",  # 标记为合并音频注册
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

    results_path = experiment_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # 生成 Markdown 摘要
    summary_path = experiment_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# 实验 {experiment_id}\n\n")
        f.write(f"**时间**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 配置\n\n")
        f.write(f"```json\n{json.dumps(config.to_dict(), indent=2)}\n```\n\n")
        f.write("## 指标\n\n")
        f.write(f"- **FAR**: {exp_metrics.far:.4f} ({exp_metrics.false_positives}/{exp_metrics.true_negatives + exp_metrics.false_positives})\n")
        f.write(f"- **FRR**: {exp_metrics.frr:.4f} ({exp_metrics.false_negatives}/{exp_metrics.true_positives + exp_metrics.false_negatives})\n")
        f.write(f"- **短音频置信度**: {exp_metrics.short_audio_confidence:.4f} ({exp_metrics.short_audio_count} 样本)\n")
        f.write(f"- **总体准确率**: {exp_metrics.overall_accuracy:.4f}\n")
        f.write(f"- **EER**: {exp_metrics.eer:.4f}\n\n")
        f.write(f"## 图表\n\n")
        f.write(f"- [热力图](heatmap.png)\n")
        f.write(f"- [柱状图](summary_bar.png)\n")

    logger.info(f"实验结果已保存到: {experiment_dir}")

    # 清理临时目录
    shutil.rmtree(temp_dir)

    return ExperimentResult(
        experiment_id=experiment_id,
        timestamp=timestamp.isoformat(),
        config=config,
        metrics=exp_metrics,
        scores_matrix=scores_matrix,
        row_labels=row_labels,
    )


# ============================================================================ #
#  主函数
# ============================================================================ #


def create_config_from_args() -> ExperimentConfig:
    """从命令行参数创建配置（可选覆盖顶部参数）."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice-ID 声纹识别自动研究实验")
    parser.add_argument("--noise", default=NOISE_PATH, help="噪声音频文件路径")
    parser.add_argument("--snrs", default=",".join(str(s) for s in NOISE_INJECTION_SNRS), help="SNR 级别，逗号分隔")
    parser.add_argument("--threshold", type=float, default=SIM_THRESHOLD, help="识别阈值")
    parser.add_argument("--crop-mode", default=VERIFY_CROP_MODE, choices=["full_utterance", "tail_window", "head_window"], help="裁剪模式")
    parser.add_argument("--buffer-secs", type=float, default=VERIFY_BUFFER_KEEP_SECS, help="buffer 保留时长")
    parser.add_argument("--window-secs", type=float, default=VERIFY_WINDOW_SECS, help="窗口长度")
    parser.add_argument("--segment-secs", type=float, default=ENROLLMENT_SEGMENT_SECS, help="注册分段长度")
    parser.add_argument("--enable-vad", action="store_true", help="启用 VAD（默认关闭）")
    parser.add_argument("--vad-threshold", type=float, default=VAD_RMS_THRESHOLD, help="VAD 能量阈值")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--debug", "-d", action="store_true", help="调试模式")
    parser.add_argument("--from-best", action="store_true", help="从 best_config.json 加载配置")

    args = parser.parse_args()

    # 如果指定了 --from-best，从最佳配置加载
    if args.from_best:
        best_config = load_best_config()
        if best_config:
            return best_config

    # 否则使用命令行参数（或顶部默认值）
    snr_levels = [float(x.strip()) for x in args.snrs.split(",")]

    return ExperimentConfig(
        sim_threshold=args.threshold,
        verify_crop_mode=args.crop_mode,
        verify_buffer_keep_secs=args.buffer_secs,
        verify_window_secs=args.window_secs,
        enrollment_segment_secs=args.segment_secs,
        enable_vad=args.enable_vad,  # 默认 False，需要 --enable-vad 才开启
        vad_rms_threshold=args.vad_threshold,
        enable_score_compensation=ENABLE_SCORE_COMPENSATION,  # 使用固定值
        score_compensation_target_duration=SCORE_COMPENSATION_TARGET_DURATION,  # 使用固定值
        noise_injection_snrs=snr_levels,
        noise_path=args.noise,
        verbose=args.verbose,
        debug=args.debug,
    )


def main() -> None:
    """主函数 - 运行实验并记录结果."""
    print("=" * 70)
    print("Voice-ID 声纹识别自动研究")
    print("=" * 70)

    # 创建输出目录
    setup_output_dirs()

    # 创建配置（可以使用顶部参数或命令行参数）
    config = create_config_from_args()

    # 生成实验 ID
    timestamp = datetime.now()
    experiment_id = f"exp_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    print(f"\n实验 ID: {experiment_id}")
    print(f"配置: {config.to_dict()}")
    print("\n开始实验...\n")

    # 运行实验
    result = cross_test(config, experiment_id)

    # 打印结果摘要
    print("\n" + "=" * 70)
    print("实验结果摘要")
    print("=" * 70)
    print(f"FAR (误接受率):     {result.metrics.far:.4f}")
    print(f"FRR (误拒绝率):     {result.metrics.frr:.4f}")
    print(f"短音频置信度:       {result.metrics.short_audio_confidence:.4f}")
    print(f"总体准确率:         {result.metrics.overall_accuracy:.4f}")
    print(f"EER:                {result.metrics.eer:.4f}")
    print(f"\n详细结果: {EXPERIMENTS_DIR / experiment_id}")
    print("=" * 70)

    # 加载历史实验记录
    experiment_log = load_experiment_log()

    # 计算与上次实验的改善
    improvement = None
    if experiment_log:
        last_result = experiment_log[-1]
        last_metrics = ExperimentMetrics.from_dict(last_result["metrics"])
        improvement = calculate_improvement(last_metrics, result.metrics)
        print("\n与上次实验相比:")
        print(format_improvement_summary(improvement))

    # 判断是否为新的最佳配置
    is_new_best = False
    old_score = 1.0
    if BEST_CONFIG_PATH.exists():
        try:
            with open(BEST_CONFIG_PATH, "r", encoding="utf-8") as f:
                best_data = json.load(f)
            old_score = best_data.get("metrics", {}).get("eer", 1.0)
        except Exception:
            old_score = 1.0
    else:
        is_new_best = True
        print("\n✓ 首次实验，保存为最佳配置")

    new_score = result.metrics.eer
    if not is_new_best and new_score < old_score:
        is_new_best = True
        print(f"\n✓ 新最佳配置！EER 从 {old_score:.4f} 降至 {new_score:.4f}")

    # 保存实验记录
    experiment_log.append(result.to_dict())
    save_experiment_log(experiment_log)
    print(f"\n✓ 实验记录已保存: {OUTPUT_ROOT / 'experiment_log.json'}")

    # 更新最佳配置
    if is_new_best:
        save_best_config(config, result.metrics)
        print(f"✓ 最佳配置已更新: {OUTPUT_ROOT / 'best_config.json'}")

    # 打印下一步建议
    print("\n" + "-" * 70)
    print("下一步建议:")

    if result.metrics.far > 0.05:
        print("  - FAR 较高，考虑提高 sim_threshold")
        print("  - 或启用 VAD 并降低 vad_rms_threshold")

    if result.metrics.frr > 0.10:
        print("  - FRR 较高，考虑降低 sim_threshold")

    if result.metrics.short_audio_confidence < 0.45:
        print("  - 短音频置信度较低，考虑:")
        print("    * 使用 head_window 模式")
        print("    * 降低 verify_window_secs 到 0.3-0.5")
        print("    * 降低 sim_threshold 到 0.30-0.40")
        print("    * 禁用 enable_vad")

    print("-" * 70)
    print(f"\n再次运行: uv run training.py")
    print(f"使用最佳配置: uv run training.py --from-best")
    print(f"查看历史: cat {OUTPUT_ROOT / 'experiment_log.json'}")


if __name__ == "__main__":
    main()
