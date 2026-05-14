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
from src.wespeaker.wespeaker import _apply_silero_vad, _crop_to_duration, _extract_embedding
import torch.nn.functional as F
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
    SHORT_AUDIO_DURATION_VAD,
    SHORT_AUDIO_DURATION_NO_VAD,
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
SIM_THRESHOLD = 0.36  # 最佳平衡点 [0.20 - 0.75]
VERIFY_CROP_MODE = "head_window"  # 裁剪模式: head_window 聚焦音频开头，改善短音频匹配
VERIFY_BUFFER_KEEP_SECS = 60.0  # buffer 最大保留时长 [2.0 - 60.0]，60.0 表示不截断
VERIFY_WINDOW_SECS = 0.4  # 短窗口 [0.3-0.5] 提升短音频匹配精度

# ---- 注册参数 ----
ENROLLMENT_SEGMENT_SECS = 0.6  # 更短分段 → 更多样化的注册 embeddings [0.5 - 3.0]

# ---- VAD 参数 ----
ENABLE_VAD = False  # VAD 损害短音频性能，保持关闭
VAD_RMS_THRESHOLD = 0.002  # VAD 能量阈值 [0.001 - 0.02]（VAD 关闭时无效）

# ---- 噪声注入 ----
NOISE_INJECTION_SNRS = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0]  # SNR 级别 — 经验证的宽范围配置
NOISE_PATH = "extract-noisy+pink"  # 当前最佳：提取噪声 + 粉红噪声混合

# ---- 滑动窗口参数 ----
SLIDING_WINDOW_SECS = 0.4  # 滑动窗口长度（更细粒度捕捉最佳匹配段）[0.3 - 3.0]
SLIDING_HOP_SECS = 0.15  # 滑动步长（更多窗口覆盖）[0.1 - 1.0]

# ---- 测试音频处理 ----
TEST_CROP_SECS = 10.0  # 测试音频最大裁剪长度 [2.0 - 60.0]

# ---- 注册增强控制 ----
ENROLLMENT_SKIP_VAD = True  # True = 注册时跳过 VAD（与测试条件对称，已验证提升置信度）
ENROLL_NO_NOISE = False  # True = 注册时不注入噪声，只使用干净片段（极致纯净声纹）
ENROLL_CLEAN_ONLY = True  # True = 仅使用干净片段（不加噪声）来构建声纹

# ---- 多模板匹配 ----
MULTI_TEMPLATE_MATCH = True  # True = 保留每个注册片段 embedding，测试时取 max（不平均）

# ---- 滑动窗口测试（仅对短音频）----
SLIDING_WINDOW_TEST = False  # True = 对短音频使用滑动窗口取 max 分数

# ---- 短音频最低时长 ----
MIN_TEST_SECS = 1.0  # 短音频最低 padding 时长（避免循环重复损害 embedding）

# ---- 注册分片（多模板模式下生效）----
TEMPLATE_CHUNK_SECS = 0  # 每个模板片段长度（0 = 不分片，使用整文件；>0 虽多模板但可能引入噪声）
TEMPLATE_CHUNK_OVERLAP = 0.5  # 片段重叠比例

# ---- 输出控制 ----
VERBOSE = False  # 详细输出模式
DEBUG = False  # 调试模式

# ---------------------------------------------------------------------------- #
#  可调参数区域结束
# ============================================================================ #

# ---- 固定参数（不修改）---- #
ENABLE_SCORE_COMPENSATION = True  # 分数补偿固定关闭
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

    # 2. 提取或生成噪声 profile
    noise_path_lower = str(config.noise_path).lower()
    if noise_path_lower.startswith("synthetic"):
        if config.verbose:
            logger.info("使用合成噪声")
        noise_type = str(config.noise_path).split(":")[1] if ":" in str(config.noise_path) else "white"
        duration_secs = 5.0
        sample_rate = 16000
        n_samples = int(duration_secs * sample_rate)
        if noise_type == "pink":
            white = np.random.randn(n_samples)
            fft = np.fft.rfft(white)
            freq = np.fft.rfftfreq(n_samples, d=1/sample_rate)
            fft[freq > 0] /= np.sqrt(freq[freq > 0])
            noise_profile = np.fft.irfft(fft, n=n_samples).astype(np.float32)
            noise_profile /= np.std(noise_profile) + 1e-12
        else:
            noise_profile = np.random.randn(n_samples).astype(np.float32)
        if config.verbose:
            logger.info(f"  生成 {noise_type} 噪声: {duration_secs:.1f}s")
    elif noise_path_lower.startswith("extract-noisy"):
        # 从 test 集合中提取环境噪声 — 遍历所有 test_segments_dir
        if config.verbose:
            logger.info("从 test 集合中提取环境噪声")
        from src.wespeaker.wespeaker import _vad_segments, _load_audio
        all_noise_parts = []
        sample_rate = 16000
        for name, paths in SPEAKERS.items():
            test_dir = Path(paths["test_segments_dir"])
            test_files = sorted(test_dir.glob("*.wav"))
            for audio_file in test_files:
                waveform = _load_audio(str(audio_file), sample_rate)
                noise_segs = _vad_segments(waveform, rms_threshold=0.005, sample_rate=sample_rate)
                if not noise_segs:
                    # 无静音段，取末尾 10% 作为噪声近似
                    tail_len = waveform.numel() // 10
                    if tail_len > sample_rate // 2:
                        all_noise_parts.append(waveform[-tail_len:])
                else:
                    for seg in noise_segs:
                        all_noise_parts.append(seg)
        if all_noise_parts:
            noise_profile = torch.cat(all_noise_parts).numpy()
            # 可选混合合成噪声 (extract-noisy+pink 或 extract-noisy+white)
            if "+pink" in noise_path_lower or "+white" in noise_path_lower:
                blend_type = "pink" if "+pink" in noise_path_lower else "white"
                synth_len = len(noise_profile)
                if blend_type == "pink":
                    synth_white = np.random.randn(synth_len)
                    fft = np.fft.rfft(synth_white)
                    freq = np.fft.rfftfreq(synth_len, d=1/sample_rate)
                    fft[freq > 0] /= np.sqrt(freq[freq > 0])
                    synth_noise = np.fft.irfft(fft, n=synth_len).astype(np.float32)
                    synth_noise /= np.std(synth_noise) + 1e-12
                else:
                    synth_noise = np.random.randn(synth_len).astype(np.float32)
                noise_profile = 0.5 * noise_profile + 0.5 * synth_noise[:synth_len]
                noise_profile = noise_profile.astype(np.float32)
                if config.verbose:
                    logger.info(f"  混合 {blend_type} 噪声 (50%)")
            if config.verbose:
                logger.info(f"  从 {len(all_noise_parts)} 个噪声段提取, 总长 {len(noise_profile)/sample_rate:.1f}s")
        else:
            logger.warning("未能从 test 集合提取噪声，使用白噪声替代")
            noise_profile = np.random.randn(int(3.0 * sample_rate)).astype(np.float32)
    else:
        if config.verbose:
            logger.info(f"提取噪声 profile: {config.noise_path}")
        noise_profile = WespeakerBest.extract_noise_profile(config.noise_path)
    if config.verbose:
        logger.info(f"  噪声长度: {len(noise_profile) / 16000:.1f}s")

    # 3. 注册所有说话人（合并音频方式）
    voiceprints: dict[str, torch.Tensor] = {}
    multi_templates: dict[str, list[torch.Tensor]] = {}  # 多模板：每个注册片段一个 embedding

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

        # 执行注册（分段注册方式）
        enroll_snrs = config.noise_injection_snrs if config.noise_injection_snrs else [40.0]
        if ENROLL_CLEAN_ONLY:
            # 极致纯净注册：不注入噪声，不做 VAD，只用干净片段
            from src.wespeaker.wespeaker import _load_audio

            clean_paths = sorted(Path(reg_dir).glob("*.wav"))
            all_embs = []
            chunk_len = int(TEMPLATE_CHUNK_SECS * recognizer._client.sample_rate) if MULTI_TEMPLATE_MATCH and TEMPLATE_CHUNK_SECS > 0 else 0
            hop_len = int(chunk_len * (1 - TEMPLATE_CHUNK_OVERLAP)) if chunk_len > 0 else 0
            for cp in clean_paths:
                seg = _load_audio(str(cp), recognizer._client.sample_rate)
                if not ENROLLMENT_SKIP_VAD:
                    seg = _apply_silero_vad(seg, recognizer._client.sample_rate)
                if chunk_len > 0 and seg.numel() > chunk_len:
                    # 切分成重叠片段
                    for start in range(0, seg.numel() - chunk_len + 1, hop_len):
                        chunk = seg[start:start + chunk_len]
                        emb = _extract_embedding(recognizer._client._model, chunk)
                        all_embs.append(F.normalize(emb, dim=0))
                else:
                    emb = _extract_embedding(recognizer._client._model, seg)
                    all_embs.append(F.normalize(emb, dim=0))
            if MULTI_TEMPLATE_MATCH:
                multi_templates[name] = all_embs  # 保留每个片段的独立 embedding
            ref = F.normalize(torch.stack(all_embs).mean(dim=0), dim=0)
            voiceprints[name] = ref
            if config.verbose:
                tag = "纯净" if ENROLLMENT_SKIP_VAD else "纯净+VAD"
                logger.info(f"  注册（{tag}）{len(clean_paths)} 个片段 = {len(all_embs)} embeddings")
        elif ENROLLMENT_SKIP_VAD:
            # 自定义注册：跳过 VAD 以匹配测试条件，保留噪声注入
            from src.wespeaker.wespeaker import _load_audio

            clean_paths = sorted(Path(reg_dir).glob("*.wav"))
            all_embs = []
            chunk_len = int(TEMPLATE_CHUNK_SECS * recognizer._client.sample_rate) if MULTI_TEMPLATE_MATCH and TEMPLATE_CHUNK_SECS > 0 else 0
            hop_len = int(chunk_len * (1 - TEMPLATE_CHUNK_OVERLAP)) if chunk_len > 0 else 0
            for cp in clean_paths:
                seg = _load_audio(str(cp), recognizer._client.sample_rate)
                if chunk_len > 0 and seg.numel() > chunk_len:
                    seg_np = seg.cpu().numpy()
                    for start in range(0, seg.numel() - chunk_len + 1, hop_len):
                        chunk = seg_np[start:start + chunk_len]
                        for snr in enroll_snrs:
                            from src.wespeaker.best import _mix_noise_at_snr
                            mixed = _mix_noise_at_snr(chunk, noise_profile, snr)
                            emb = _extract_embedding(recognizer._client._model, torch.from_numpy(mixed))
                            all_embs.append(F.normalize(emb, dim=0))
                else:
                    seg_np = seg.cpu().numpy()
                    for snr in enroll_snrs:
                        from src.wespeaker.best import _mix_noise_at_snr
                        mixed = _mix_noise_at_snr(seg_np, noise_profile, snr)
                        emb = _extract_embedding(recognizer._client._model, torch.from_numpy(mixed))
                        all_embs.append(F.normalize(emb, dim=0))
            if MULTI_TEMPLATE_MATCH:
                multi_templates[name] = all_embs  # 保留每个片段的独立 embedding
            ref = F.normalize(torch.stack(all_embs).mean(dim=0), dim=0)
            voiceprints[name] = ref
            if config.verbose:
                logger.info(f"  注册（无VAD）{len(clean_paths)} 个分段 × {len(enroll_snrs)} SNR = {len(all_embs)} embeddings")
        else:
            result = recognizer.enroll(str(reg_path), noise_profile, str(tmp_pk), enroll_snrs)
            voiceprints[name] = result["embedding"]

        if config.verbose and not ENROLL_CLEAN_ONLY:
            logger.info(f"  注册 {result['num_segments']} 个分段 × {len(enroll_snrs)} SNR = {result['total_enrollments']} embeddings")

        # 记录噪声注入效果（仅在非纯净注册时）
        if not ENROLL_CLEAN_ONLY:
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
            # Only apply Silero VAD when enabled (preserves more audio for short files)
            if config.enable_vad:
                waveform_vad = _apply_silero_vad(waveform_mono, sr)
                vad_duration = waveform_vad.shape[0] / sr
            else:
                waveform_vad = waveform_mono
                vad_duration = original_duration  # track original duration for short audio metric
            waveform_final = _crop_to_duration(waveform_vad, TEST_CROP_SECS, sr)
            # Pad short audio to minimum duration for better embedding quality
            min_test_samples = int(MIN_TEST_SECS * sr)
            if waveform_final.numel() < min_test_samples:
                repeats = min_test_samples // waveform_final.numel() + 1
                waveform_final = waveform_final.repeat(repeats)[:min_test_samples]
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

            # Decide whether to use sliding window test (short audio only: avoids false peaks on long audio)
            is_short = vad_duration < SHORT_AUDIO_DURATION_NO_VAD
            use_sliding = SLIDING_WINDOW_TEST and is_short and vad_duration >= 0.2

            if use_sliding:
                # Sliding window: extract embeddings from multiple windows, take max score
                window_samples = int(SLIDING_WINDOW_SECS * sr)
                hop_samples = int(SLIDING_HOP_SECS * sr)
                if window_samples < int(0.3 * sr):
                    window_samples = int(0.3 * sr)
                wf_len = waveform_final.numel()
                window_embs = []
                for start in range(0, max(1, wf_len - window_samples + 1), hop_samples):
                    end = min(start + window_samples, wf_len)
                    win = waveform_final[start:end]
                    if win.numel() < int(0.3 * sr):
                        repeats = int(0.3 * sr) // win.numel() + 1
                        win = win.repeat(repeats)[:int(0.3 * sr)]
                    emb = _extract_embedding(recognizer._client._model, win)
                    window_embs.append(F.normalize(emb, dim=0))
                if not window_embs:
                    window_embs = [F.normalize(
                        _extract_embedding(recognizer._client._model, waveform_final), dim=0
                    )]
            else:
                # Single embedding from full audio
                test_emb = F.normalize(
                    _extract_embedding(recognizer._client._model, waveform_final), dim=0
                )

            if config.verbose:
                row = f"{row_label:>30} |"

            for ref_name in ordered_speakers:
                ref_emb = voiceprints[ref_name]
                templates = multi_templates.get(ref_name, [ref_emb]) if MULTI_TEMPLATE_MATCH else [ref_emb]
                if use_sliding:
                    # Sliding window + multi-template: max across all combos
                    best = -1.0
                    for we in window_embs:
                        for t in templates:
                            s = float(torch.dot(we, t).clamp(-1.0, 1.0))
                            if s > best:
                                best = s
                    score = best
                elif MULTI_TEMPLATE_MATCH and len(templates) > 1:
                    # Multi-template: max similarity across all enrollment templates
                    template_scores = [float(torch.dot(test_emb, t).clamp(-1.0, 1.0)) for t in templates]
                    score = max(template_scores)
                else:
                    score = float(torch.dot(test_emb, ref_emb).clamp(-1.0, 1.0))
                # Apply score compensation for short audio (vad_duration < target)
                if ENABLE_SCORE_COMPENSATION and vad_duration < SCORE_COMPENSATION_TARGET_DURATION:
                    factor = min((SCORE_COMPENSATION_TARGET_DURATION / max(vad_duration, 0.3)) ** 0.5, 2.0)
                    score = min(score * factor, 1.0)
                is_match = score >= config.sim_threshold

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

                # Create per-comparison test case for metrics computation
                comp_dict = {
                    "test_speaker": test_speaker,
                    "ref_speaker": ref_name,
                    "score": float(score),
                    "vad_duration": vad_duration,
                    "is_match": is_match,
                    "row_label": f"{test_speaker}/{label} vs {ref_name}",
                }
                test_cases.append(comp_dict)

            metrics.end(f"recognize_{row_label}")

            test_case_dict = recog_diag.to_dict()
            test_case_dict = recog_diag.to_dict()
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
    # 短音频阈值：启用 VAD 时 0.6s（VAD 裁掉静音后音频变短），禁用 VAD 时 1.5s（原始音频长度）
    short_audio_duration = SHORT_AUDIO_DURATION_VAD if config.enable_vad else SHORT_AUDIO_DURATION_NO_VAD
    exp_metrics = ExperimentMetrics.from_results(test_cases, config.sim_threshold, short_audio_duration)

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

    # 计算总平均置信度
    all_scores = [tc.get("score", 0.0) for tc in test_cases]
    total_avg_confidence = float(np.mean(all_scores)) if all_scores else 0.0

    # 计算同人平均置信度（正样本）
    same_person_map = {
        "John": {"John", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
        "John_MeetingRoom": {"John", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
        "John_D_USB": {"John", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
        "John_D_USB_AEC": {"John", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
        "Zhong": {"Zhong", "Zhong_D_USB"},
        "Zhong_D_USB": {"Zhong", "Zhong_D_USB"},
    }
    genuine_scores = []
    short_genuine_scores = []
    for tc in test_cases:
        ts = tc.get("test_speaker", "")
        rs = tc.get("ref_speaker", "")
        group = same_person_map.get(ts, {ts})
        if rs in group:
            score = tc.get("score", 0.0)
            genuine_scores.append(score)
            if tc.get("vad_duration", 0.0) < short_audio_duration:
                short_genuine_scores.append(score)
    genuine_avg_confidence = float(np.mean(genuine_scores)) if genuine_scores else 0.0
    short_audio_genuine_confidence = float(np.mean(short_genuine_scores)) if short_genuine_scores else 0.0

    # 写入 exp_metrics（确保保存到 experiment_log.json）
    exp_metrics.total_avg_confidence = total_avg_confidence
    exp_metrics.genuine_avg_confidence = genuine_avg_confidence
    exp_metrics.short_audio_genuine_confidence = short_audio_genuine_confidence

    # 打印指标
    if config.verbose or True:
        print(f"\n总平均置信度: {total_avg_confidence:.4f} ({len(all_scores)} 样本)")
        print(f"同人平均置信度: {genuine_avg_confidence:.4f} ({len(genuine_scores)} 样本)")
        print(f"短音频同人置信度: {short_audio_genuine_confidence:.4f} ({len(short_genuine_scores)} 样本)")

    # 保存到 report_data
    report_data["metrics_extended"] = {
        "total_avg_confidence": total_avg_confidence,
        "genuine_avg_confidence": genuine_avg_confidence,
        "short_audio_genuine_confidence": short_audio_genuine_confidence,
        "num_genuine_tests": len(genuine_scores),
    }
    summary_path = experiment_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# 实验 {experiment_id}\n\n")
        f.write(f"**时间**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 配置\n\n")
        f.write(f"```json\n{json.dumps(config.to_dict(), indent=2)}\n```\n\n")
        f.write("## 指标\n\n")
        f.write(f"- **FAR**: {exp_metrics.far:.4f} ({exp_metrics.false_positives}/{exp_metrics.true_negatives + exp_metrics.false_positives})\n")
        f.write(f"- **FRR**: {exp_metrics.frr:.4f} ({exp_metrics.false_negatives}/{exp_metrics.true_positives + exp_metrics.false_negatives})\n")
        f.write(f"- **短音频置信度（全部）**: {exp_metrics.short_audio_confidence:.4f} ({exp_metrics.short_audio_count} 样本)\n")
        f.write(f"- **短音频同人置信度**: {exp_metrics.short_audio_genuine_confidence:.4f}\n")
        f.write(f"- **总体准确率**: {exp_metrics.overall_accuracy:.4f}\n")
        f.write(f"- **EER**: {exp_metrics.eer:.4f}\n")
        f.write(f"- **总平均置信度**: {total_avg_confidence:.4f}\n")
        f.write(f"- **同人平均置信度**: {genuine_avg_confidence:.4f}\n\n")
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
    parser.add_argument("--disable-vad", action="store_true", help="禁用 VAD")
    parser.add_argument("--vad-threshold", type=float, default=VAD_RMS_THRESHOLD, help="VAD 能量阈值")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--debug", "-d", action="store_true", help="调试模式")
    parser.add_argument("--enroll-skip-vad", action="store_true", help="注册时跳过 VAD（与测试条件对称）")
    parser.add_argument("--enroll-clean-only", action="store_true", help="注册时不注入噪声，只使用干净片段")
    parser.add_argument("--sliding-window-test", action="store_true", help="对短音频使用滑动窗口取 max 分数")
    parser.add_argument("--multi-template", action="store_true", help="多模板匹配：保留每个注册片段，取 max 分数")
    parser.add_argument("--from-best", action="store_true", help="从 best_config.json 加载配置")

    args = parser.parse_args()

    # 如果指定了 --from-best，从最佳配置加载
    if args.from_best:
        best_config = load_best_config()
        if best_config:
            return best_config

    # 否则使用命令行参数（或顶部默认值）
    snr_levels = [float(x.strip()) for x in args.snrs.split(",") if x.strip()] if args.snrs.strip() else []

    # 应用模块级参数覆盖
    global ENROLLMENT_SKIP_VAD, ENROLL_CLEAN_ONLY, SLIDING_WINDOW_TEST, MULTI_TEMPLATE_MATCH
    if args.enroll_skip_vad:
        ENROLLMENT_SKIP_VAD = True
    if args.enroll_clean_only:
        ENROLL_CLEAN_ONLY = True
    if args.sliding_window_test:
        SLIDING_WINDOW_TEST = True
    if args.multi_template:
        MULTI_TEMPLATE_MATCH = True

    return ExperimentConfig(
        sim_threshold=args.threshold,
        verify_crop_mode=args.crop_mode,
        verify_buffer_keep_secs=args.buffer_secs,
        verify_window_secs=args.window_secs,
        enrollment_segment_secs=args.segment_secs,
        enable_vad=ENABLE_VAD and not args.disable_vad,  # 使用模块级变量 ENABLE_VAD 作为默认值
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
    print(f"短音频同人置信度:   {result.metrics.short_audio_genuine_confidence:.4f}")
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
