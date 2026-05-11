#!/usr/bin/env python3
"""训练阶段噪声注入实验 — 用真实环境噪声增强注册声纹。

核心思路：
1. 从嘈杂环境测试音频中提取噪声 profile（非语音段）
2. 将噪声以不同 SNR 级别混合到 clean 注册片段
3. 用混合后的片段注册声纹，测试在噪声环境的识别率
"""

import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker.wespeaker import (
    WespeakerClient,
    _extract_embedding,
    _load_audio,
    _vad_segments,
)


# --------------------------------------------------------------------------- #
#  噪声提取
# --------------------------------------------------------------------------- #

def extract_noise_profile(noisy_audio_path: str, sample_rate: int = 16000,
                          rms_threshold: float = 0.005) -> np.ndarray:
    """从噪声音频中提取噪声 profile。

    方法：用 VAD 检测语音段，取非语音段（静音/纯噪声段）拼接。
    """
    waveform = _load_audio(noisy_audio_path, sample_rate)

    speech_segs = _vad_segments(waveform, rms_threshold=rms_threshold,
                                sample_rate=sample_rate)

    if not speech_segs:
        return waveform.cpu().numpy()

    # 用 RMS 能量图找非语音区域
    frame_len = int(0.02 * sample_rate)
    hop_len = int(0.01 * sample_rate)
    rms_values = []
    starts = []
    for start in range(0, len(waveform) - frame_len + 1, hop_len):
        seg = waveform[start:start + frame_len]
        rms = float(torch.sqrt((seg ** 2).mean()))
        rms_values.append(rms)
        starts.append(start)

    if not rms_values:
        return waveform.cpu().numpy()

    sorted_rms = sorted(rms_values)
    noise_floor = sorted_rms[max(0, len(sorted_rms) // 4)]
    adaptive_threshold = max(noise_floor * 2.0, rms_threshold)

    # 找出非语音帧（能量低于阈值的帧）
    noise_regions = []
    in_noise = False
    noise_start = 0
    for i, rms in enumerate(rms_values):
        if rms < adaptive_threshold and not in_noise:
            in_noise = True
            noise_start = starts[i]
        elif (rms >= adaptive_threshold or i == len(rms_values) - 1) and in_noise:
            in_noise = False
            noise_end = starts[i] if i < len(starts) else starts[-1] + frame_len
            if noise_end - noise_start > sample_rate // 10:  # 至少 0.1s
                noise_regions.append((noise_start, noise_end))

    if noise_regions:
        parts = [waveform[s:e] for s, e in noise_regions]
        result = torch.cat(parts)
        print(f"  提取到 {len(noise_regions)} 个非语音段, 总长 {len(result)/sample_rate:.1f}s")
        return result.cpu().numpy()

    # 兜底
    return waveform.cpu().numpy()


def mix_noise_at_snr(clean: np.ndarray, noise: np.ndarray,
                     target_snr_db: float, sample_rate: int = 16000) -> np.ndarray:
    """将噪声混合到 clean 音频中，达到目标 SNR。

    SNR = 10 * log10(P_signal / P_noise)
    """
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-12:
        return clean.copy()

    target_noise_power = clean_power / (10 ** (target_snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    if len(noise) < len(clean):
        noise = np.tile(noise, (len(clean) // len(noise)) + 2)
    noise = noise[:len(clean)]

    return (clean + noise_scale * noise).astype(np.float32)


# --------------------------------------------------------------------------- #
#  注册与测试
# --------------------------------------------------------------------------- #

def enroll_noise_injected(client: WespeakerClient, clean_segments: list[np.ndarray],
                          noise_profile: np.ndarray, snr_levels: list[float]
                          ) -> torch.Tensor:
    """用噪声注入方式注册声纹。"""
    all_embeddings = []

    for clean_seg in clean_segments:
        for snr in snr_levels:
            mixed = mix_noise_at_snr(clean_seg, noise_profile, snr)
            emb = _extract_embedding(
                client._model, torch.from_numpy(mixed)
            )
            all_embeddings.append(F.normalize(emb, dim=0))

    ref = F.normalize(torch.stack(all_embeddings).mean(dim=0), dim=0)
    return ref


def enroll_clean_only(client: WespeakerClient,
                      clean_segments: list[np.ndarray]) -> torch.Tensor:
    """纯 clean 注册（baseline）。"""
    embeddings = []
    for seg in clean_segments:
        emb = _extract_embedding(client._model, torch.from_numpy(seg))
        embeddings.append(F.normalize(emb, dim=0))
    return F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)


def enroll_gaussian_augmented(client: WespeakerClient,
                               clean_segments: list[np.ndarray],
                               seed: int = 42) -> torch.Tensor:
    """高斯 SNR 增强注册（复现之前的实验）。"""
    from src.wespeaker.wespeaker import _NoiseAugmentor
    aug = _NoiseAugmentor(sample_rate=16000, augment_ratio=1.0, seed=seed)
    augmented = aug.augment(clean_segments)

    all_embeddings = []
    for seg in clean_segments:
        emb = _extract_embedding(client._model, torch.from_numpy(seg))
        all_embeddings.append(F.normalize(emb, dim=0))
    for seg in augmented:
        emb = _extract_embedding(client._model, torch.from_numpy(seg))
        all_embeddings.append(F.normalize(emb, dim=0))

    return F.normalize(torch.stack(all_embeddings).mean(dim=0), dim=0)


def sliding_window_test(client: WespeakerClient, audio_path: str,
                        reference: torch.Tensor, window_secs: float = 2.0,
                        step_secs: float = 0.5) -> tuple[list[dict], float]:
    """滑动窗口测试 + full utterance。"""
    waveform = _load_audio(audio_path, client.sample_rate)
    total_secs = len(waveform) / client.sample_rate
    window_samples = int(window_secs * client.sample_rate)
    step_samples = int(step_secs * client.sample_rate)

    results = []
    pos = 0.0
    while pos + window_secs <= total_secs:
        start = int(pos * client.sample_rate)
        end = start + window_samples
        segment = waveform[start:end]
        emb = F.normalize(_extract_embedding(client._model, segment), dim=0)
        score = float(torch.dot(emb, reference).clamp(-1.0, 1.0).item())
        results.append({"start": round(pos, 2), "score": round(score, 4)})
        pos += step_secs

    emb = F.normalize(_extract_embedding(client._model, waveform), dim=0)
    full_score = float(torch.dot(emb, reference).clamp(-1.0, 1.0).item())

    return results, round(full_score, 4)


def print_results(label: str, window_results: list[dict], full_score: float,
                  threshold: float = 0.55) -> dict:
    """打印测试结果并返回统计。"""
    scores = [r["score"] for r in window_results]
    max_score = max(scores)
    max_pos = next(r["start"] for r in window_results if r["score"] == max_score)
    avg_score = float(np.mean(scores))
    passed = sum(1 for s in scores if s >= threshold)
    pass_rate = passed / len(scores) * 100 if scores else 0
    above_70 = sum(1 for s in scores if s >= 0.70)

    print(f"  {label}")
    print(f"    最高分: {max_score:.4f} (@ {max_pos:.1f}s)")
    print(f"    平均分: {avg_score:.4f}")
    print(f"    >= {threshold:.2f}: {passed}/{len(scores)} ({pass_rate:.1f}%)")
    print(f"    >= 0.70: {above_70}/{len(scores)}")
    print(f"    Full utterance: {full_score:.4f} {'✅' if full_score >= threshold else '❌'}")

    return {
        "max_score": max_score, "max_pos": max_pos,
        "mean_score": avg_score, "pass_count": passed,
        "total_windows": len(scores), "pass_rate": pass_rate,
        "above_70": above_70, "full_score": full_score,
    }


# --------------------------------------------------------------------------- #
#  主实验
# --------------------------------------------------------------------------- #

def main():
    client = WespeakerClient(device="cpu", enable_augmentation=False)
    client._ensure_model()
    threshold = client.sim_threshold

    print("=" * 60)
    print("训练阶段噪声注入实验")
    print("=" * 60)
    print(f"模型: pyannote/wespeaker-voxceleb-resnet34-LM")
    print(f"阈值: {threshold}")
    print(f"噪声来源: asset/john/test_noise_segments/嘈杂环境测试.m4a")

    # 加载 clean 注册片段
    clean_paths = sorted(glob.glob("asset/john/registration_segments/*.wav"))
    clean_segments = []
    for p in clean_paths:
        w = _load_audio(p, 16000)
        clean_segments.append(w.cpu().numpy())
    print(f"Clean 注册片段: {len(clean_segments)} 个")

    # 提取噪声 profile
    noisy_path = "asset/john/test_noise_segments/嘈杂环境测试.m4a"
    print("\n提取噪声 profile...")
    noise_profile = extract_noise_profile(noisy_path)
    noise_secs = len(noise_profile) / 16000
    print(f"噪声 profile: {noise_secs:.1f}s ({len(noise_profile)} samples)")

    noisy_test_path = "asset/john/test_noise_segments/嘈杂环境测试.m4a"
    clean_test_path = "asset/john/test_clean_segments/安静环境测试测试.m4a"

    all_results = {}

    # ---- Baseline: 纯 clean 注册 ----
    print(f"\n{'='*60}")
    print("方案 0: Baseline (纯 clean 注册)")
    print(f"{'='*60}")
    ref_baseline = enroll_clean_only(client, clean_segments)
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_baseline)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_baseline)
    all_results["baseline_noisy"] = print_results("Noisy", sw_noisy, full_noisy, threshold)
    all_results["baseline_clean"] = print_results("Clean", sw_clean, full_clean, threshold)

    # ---- 复现: 高斯增强 ----
    print(f"\n{'='*60}")
    print("方案 1: 高斯 SNR 增强 (复现)")
    print(f"{'='*60}")
    ref_gaussian = enroll_gaussian_augmented(client, clean_segments)
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_gaussian)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_gaussian)
    all_results["gaussian_noisy"] = print_results("Noisy", sw_noisy, full_noisy, threshold)
    all_results["gaussian_clean"] = print_results("Clean", sw_clean, full_clean, threshold)

    # ---- 方案 2: 噪声注入 (multi-SNR: 20, 15, 10, 5, 0 dB) ----
    print(f"\n{'='*60}")
    print("方案 2: 噪声注入 (multi-SNR: 20, 15, 10, 5, 0 dB)")
    print(f"{'='*60}")
    snr_levels_multi = [20, 15, 10, 5, 0]
    ref_multi = enroll_noise_injected(client, clean_segments, noise_profile, snr_levels_multi)
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_multi)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_multi)
    all_results["multi_noisy"] = print_results("Noisy", sw_noisy, full_noisy, threshold)
    all_results["multi_clean"] = print_results("Clean", sw_clean, full_clean, threshold)

    # ---- 方案 3: 噪声注入 (仅 SNR=10dB) ----
    print(f"\n{'='*60}")
    print("方案 3: 噪声注入 (SNR=10dB only)")
    print(f"{'='*60}")
    ref_10db = enroll_noise_injected(client, clean_segments, noise_profile, [10])
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_10db)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_10db)
    all_results["10db_noisy"] = print_results("Noisy", sw_noisy, full_noisy, threshold)
    all_results["10db_clean"] = print_results("Clean", sw_clean, full_clean, threshold)

    # ---- 方案 4: 噪声注入 (仅 SNR=5dB) ----
    print(f"\n{'='*60}")
    print("方案 4: 噪声注入 (SNR=5dB only)")
    print(f"{'='*60}")
    ref_5db = enroll_noise_injected(client, clean_segments, noise_profile, [5])
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_5db)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_5db)
    all_results["5db_noisy"] = print_results("Noisy", sw_noisy, full_noisy, threshold)
    all_results["5db_clean"] = print_results("Clean", sw_clean, full_clean, threshold)

    # ---- 方案 5: 噪声注入 (原始 + multi-SNR, 加权) ----
    print(f"\n{'='*60}")
    print("方案 5: 噪声注入 (原始 + multi-SNR, SNR 加权)")
    print(f"{'='*60}")
    all_embeddings_weighted = []
    snr_weights_list = []
    for clean_seg in clean_segments:
        emb = _extract_embedding(client._model, torch.from_numpy(clean_seg))
        all_embeddings_weighted.append(F.normalize(emb, dim=0))
        snr_weights_list.append(30.0)
        for snr in [20, 15, 10, 5, 0]:
            mixed = mix_noise_at_snr(clean_seg, noise_profile, snr)
            emb = _extract_embedding(client._model, torch.from_numpy(mixed))
            all_embeddings_weighted.append(F.normalize(emb, dim=0))
            snr_weights_list.append(snr if snr > 0 else 0.1)

    weights = torch.tensor(snr_weights_list, dtype=torch.float32)
    weights = weights / weights.sum()
    ref_weighted = F.normalize(
        (torch.stack(all_embeddings_weighted) * weights.unsqueeze(1)).sum(dim=0), dim=0
    )
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_weighted)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_weighted)
    all_results["weighted_noisy"] = print_results("Noisy", sw_noisy, full_noisy, threshold)
    all_results["weighted_clean"] = print_results("Clean", sw_clean, full_clean, threshold)

    # ---- 汇总表 ----
    print(f"\n\n{'='*80}")
    print(f"  汇总对比 (Noisy 环境)")
    print(f"{'='*80}")
    print(f"  {'方案':<30} {'最高分':>8} {'平均分':>8} {'>={threshold:.2f}':>12} "
          f"{'>=0.70':>8} {'Full':>8}")
    print(f"  {'-'*80}")

    scheme_names = {
        "baseline_noisy": "Baseline (纯 clean)",
        "gaussian_noisy": "高斯 SNR 增强",
        "multi_noisy": "噪声注入 (multi-SNR)",
        "10db_noisy": "噪声注入 (SNR=10dB)",
        "5db_noisy": "噪声注入 (SNR=5dB)",
        "weighted_noisy": "噪声注入 (加权)",
    }

    for key in ["baseline_noisy", "gaussian_noisy", "multi_noisy",
                "10db_noisy", "5db_noisy", "weighted_noisy"]:
        r = all_results[key]
        print(f"  {scheme_names[key]:<30} {r['max_score']:>8.4f} {r['mean_score']:>8.4f} "
              f"{r['pass_count']:>5}/{r['total_windows']} ({r['pass_rate']:.1f}%) "
              f"{r['above_70']:>4}/{r['total_windows']} {r['full_score']:>8.4f}")

    print(f"{'='*80}")

    # ---- Clean 环境退化检查 ----
    print(f"\n  Clean 环境退化检查")
    print(f"  {'方案':<30} {'最高分':>8} {'Full':>8} {'退化':>8}")
    print(f"  {'-'*60}")
    baseline_clean_max = all_results["baseline_clean"]["max_score"]
    baseline_clean_full = all_results["baseline_clean"]["full_score"]

    for key in ["gaussian_clean", "multi_clean", "10db_clean", "5db_clean", "weighted_clean"]:
        r = all_results[key]
        max_degrade = r["max_score"] - baseline_clean_max
        full_degrade = r["full_score"] - baseline_clean_full
        name = key.replace("_clean", "").replace("baseline", "Baseline").replace("gaussian", "高斯增强").replace("multi", "multi-SNR").replace("10db", "SNR=10dB").replace("5db", "SNR=5dB").replace("weighted", "加权")
        print(f"  {name:<30} {r['max_score']:>8.4f} {r['full_score']:>8.4f} "
              f"{max_degrade:>+8.4f}")


if __name__ == "__main__":
    main()
