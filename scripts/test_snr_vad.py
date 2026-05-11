#!/usr/bin/env python3
"""SNR 加权 + VAD 去静音效果验证。

测试 4 种配置对比：
1. Baseline（无 SNR，无 VAD）
2. 仅 SNR 加权注册
3. 仅 VAD 去静音识别
4. SNR 加权 + VAD 组合
"""

import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker.wespeaker import (
    WespeakerClient,
    _estimate_snr,
    _extract_embedding,
    _load_audio,
    _vad_segments,
)


def enroll_with_config(client: WespeakerClient, segment_dir: str) -> torch.Tensor:
    """注册声纹，使用 client 当前的 enable_snr_weighting 设置。"""
    files = sorted(glob.glob(f"{segment_dir}/S01_*.wav"))
    if not files:
        raise RuntimeError(f"No segments found in {segment_dir}")

    embeddings = []
    segments = []
    for f in files:
        waveform = _load_audio(f, client.sample_rate)
        seg_len = int(client.enrollment_segment_secs * client.sample_rate)
        if waveform.numel() < seg_len:
            continue
        # 使用 1s 片段
        for i in range(len(waveform) // seg_len):
            seg = waveform[i * seg_len : (i + 1) * seg_len]
            segments.append(seg)
            emb = _extract_embedding(client._model, seg)
            embeddings.append(emb)

    if client.enable_snr_weighting:
        snr_values = [_estimate_snr(seg, sample_rate=client.sample_rate) for seg in segments]
        weights = torch.tensor([max(s, 0.0) for s in snr_values], dtype=torch.float32)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            ref = F.normalize((torch.stack(embeddings) * weights.unsqueeze(1)).sum(dim=0), dim=0)
        else:
            ref = F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)
    else:
        ref = F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)

    return ref, len(embeddings)


def sliding_window_test(
    client: WespeakerClient,
    audio_path: str,
    reference: torch.Tensor,
    window_secs: float = 2.0,
    step_secs: float = 0.5,
) -> tuple[list[dict], float]:
    """滑动窗口测试 + full utterance。"""
    waveform = _load_audio(audio_path, client.sample_rate)

    # VAD 处理
    if client.enable_vad and client.verify_crop_mode == "full_utterance":
        speech_segs = _vad_segments(
            waveform, rms_threshold=client.vad_rms_threshold, sample_rate=client.sample_rate
        )
        if speech_segs:
            test_waveform = torch.cat(speech_segs)
        else:
            test_waveform = waveform
    else:
        test_waveform = waveform

    # 限制长度
    max_samples = int(client.verify_buffer_keep_secs * client.sample_rate)
    if test_waveform.numel() > max_samples:
        if client.verify_crop_mode == "head_window":
            test_waveform = test_waveform[:max_samples]
        else:
            test_waveform = test_waveform[-max_samples:]

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

    # Full utterance on processed waveform
    emb = F.normalize(_extract_embedding(client._model, test_waveform), dim=0)
    full_score = float(torch.dot(emb, reference).clamp(-1.0, 1.0).item())

    return results, round(full_score, 4)


def print_results(label: str, window_results: list[dict], full_score: float, threshold: float):
    scores = [r["score"] for r in window_results]
    max_score = max(scores)
    avg_score = np.mean(scores)
    passed = sum(1 for s in scores if s >= threshold)
    pass_rate = passed / len(scores) * 100 if scores else 0

    print(f"\n  {label}")
    print(
        f"  最高分: {max_score:.4f}  平均分: {avg_score:.4f}  "
        f">={threshold:.2f}: {passed}/{len(scores)} ({pass_rate:.1f}%)  "
        f"Full: {full_score:.4f} {'✅' if full_score >= threshold else '❌'}"
    )

    return max_score, avg_score, pass_rate, full_score


def main():
    client = WespeakerClient(device="cpu", enable_augmentation=False)
    client._ensure_model()
    threshold = client.sim_threshold

    print(f"\nSNR 加权 + VAD 效果验证")
    print(f"模型: pyannote/wespeaker-voxceleb-resnet34-LM")
    print(f"阈值: {threshold} | VAD RMS: {client.vad_rms_threshold}")

    clean_test = "asset/john/test_clean_segments/安静环境测试测试.m4a"
    noisy_test = "asset/john/test_noise_segments/嘈杂环境测试.m4a"

    configs = [
        {"name": "Baseline (无 SNR, 无 VAD)", "snr": False, "vad": False},
        {"name": "仅 SNR 加权", "snr": True, "vad": False},
        {"name": "仅 VAD 去静音", "snr": False, "vad": True},
        {"name": "SNR + VAD 组合", "snr": True, "vad": True},
    ]

    summary = []

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"  配置: {cfg['name']}")
        print(f"{'='*70}")

        client.enable_snr_weighting = cfg["snr"]
        client.enable_vad = cfg["vad"]

        # 注册
        ref, n_segs = enroll_with_config(client, "asset/john/registration_segments")

        # SNR 分布展示
        if cfg["snr"]:
            files = sorted(glob.glob("asset/john/registration_segments/S01_*.wav"))
            seg_len = int(client.enrollment_segment_secs * client.sample_rate)
            snr_list = []
            for f in files:
                w = _load_audio(f, 16000)
                for i in range(len(w) // seg_len):
                    seg = w[i * seg_len : (i + 1) * seg_len]
                    snr_list.append(_estimate_snr(seg))
            if snr_list:
                print(
                    f"  注册片段 SNR: min={min(snr_list):.1f}dB, "
                    f"max={max(snr_list):.1f}dB, mean={np.mean(snr_list):.1f}dB"
                )

        # VAD 段数展示
        if cfg["vad"]:
            for label, path in [("Clean", clean_test), ("Noisy", noisy_test)]:
                w = _load_audio(path, 16000)
                segs = _vad_segments(w, rms_threshold=client.vad_rms_threshold)
                total_vad_secs = sum(len(s) / 16000 for s in segs)
                orig_secs = len(w) / 16000
                print(
                    f"  {label} VAD: {len(segs)} 段, {total_vad_secs:.1f}s / {orig_secs:.1f}s "
                    f"(保留 {total_vad_secs/orig_secs*100:.0f}%)"
                )

        # Clean 测试
        clean_results, clean_full = sliding_window_test(client, clean_test, ref)
        c_max, c_avg, c_rate, c_full = print_results("Clean", clean_results, clean_full, threshold)

        # Noisy 测试
        noisy_results, noisy_full = sliding_window_test(client, noisy_test, ref)
        n_max, n_avg, n_rate, n_full = print_results("Noisy", noisy_results, noisy_full, threshold)

        summary.append(
            {
                "name": cfg["name"],
                "clean_max": c_max,
                "clean_full": c_full,
                "clean_rate": c_rate,
                "noisy_max": n_max,
                "noisy_full": n_full,
                "noisy_rate": n_rate,
            }
        )

    # 汇总表
    print(f"\n\n{'='*80}")
    print(f"  汇总对比")
    print(f"{'='*80}")
    print(
        f"  {'配置':<25} {'Clean最高':>10} {'Clean Full':>10} {'Clean通过率':>10}  "
        f"{'Noisy最高':>10} {'Noisy Full':>10} {'Noisy通过率':>10}"
    )
    print(f"  {'-'*80}")
    for s in summary:
        print(
            f"  {s['name']:<25} {s['clean_max']:>10.4f} {s['clean_full']:>10.4f} "
            f"{s['clean_rate']:>9.1f}%  {s['noisy_max']:>10.4f} {s['noisy_full']:>10.4f} "
            f"{s['noisy_rate']:>9.1f}%"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
