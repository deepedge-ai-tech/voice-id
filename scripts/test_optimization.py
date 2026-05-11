"""验证优化后的噪声环境识别率。

使用新的默认参数 (full_utterance, sim_threshold=0.55, verify_buffer_keep_secs=8.0)
对比 clean 和 noisy 环境下的识别表现。
"""

import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker.wespeaker import WespeakerClient, _extract_embedding, _load_audio


def enroll_segments(client: WespeakerClient, segment_dir: str) -> torch.Tensor:
    """注册多个片段，返回归一化的平均 embedding。"""
    files = sorted(glob.glob(f"{segment_dir}/S01_*.wav"))
    if not files:
        raise RuntimeError(f"No segments found in {segment_dir}")

    embeddings = []
    for f in files:
        waveform = _load_audio(f, client.sample_rate)
        seg_len = int(client.enrollment_segment_secs * client.sample_rate)
        if waveform.numel() < seg_len:
            continue
        emb = _extract_embedding(client._model, waveform)
        embeddings.append(emb)

    mean_emb = F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)
    return mean_emb


def sliding_window_test(
    client: WespeakerClient, audio_path: str, reference: torch.Tensor,
    window_secs: float = 2.0, step_secs: float = 0.5,
) -> list[dict]:
    """滑动窗口测试，返回每个窗口的相似度分数。"""
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
        emb = _extract_embedding(client._model, segment)
        emb = F.normalize(emb, dim=0)
        score = float(torch.dot(emb, reference).clamp(-1.0, 1.0).item())
        results.append({"start": round(pos, 2), "score": round(score, 4)})
        pos += step_secs

    # full utterance 结果
    pcm = waveform
    max_samples = int(client.verify_buffer_keep_secs * client.sample_rate)
    if pcm.numel() > max_samples:
        if client.verify_crop_mode == "head_window":
            pcm = pcm[:max_samples]
        else:
            pcm = pcm[-max_samples:]
    emb = _extract_embedding(client._model, pcm)
    emb = F.normalize(emb, dim=0)
    full_score = float(torch.dot(emb, reference).clamp(-1.0, 1.0).item())

    return results, round(full_score, 4)


def format_results(label: str, window_results: list[dict], full_score: float, threshold: float):
    """格式化输出测试结果。"""
    scores = [r["score"] for r in window_results]
    max_score = max(scores)
    avg_score = np.mean(scores)
    passed = sum(1 for s in scores if s >= threshold)
    pass_rate = passed / len(scores) * 100 if scores else 0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  窗口数:          {len(scores)}")
    print(f"  最高分:          {max_score:.4f}")
    print(f"  平均分:          {avg_score:.4f}")
    print(f"  >= {threshold:.2f} 窗口数: {passed}/{len(scores)} ({pass_rate:.1f}%)")
    print(f"  Full utterance:  {full_score:.4f} {'✅' if full_score >= threshold else '❌'}")
    print(f"{'='*60}")

    return max_score, avg_score, pass_rate, full_score


def main():
    client = WespeakerClient(device="cpu", enable_augmentation=False)
    client._ensure_model()

    threshold = client.sim_threshold
    print(f"\n优化验证测试")
    print(f"参数: verify_crop_mode={client.verify_crop_mode}, "
          f"sim_threshold={threshold}, verify_buffer_keep_secs={client.verify_buffer_keep_secs}")

    # 注册：使用 clean 片段
    print(f"\n正在注册 clean 声纹...")
    clean_ref = enroll_segments(client, "asset/john/registration_segments")
    print(f"  使用 {len(glob.glob('asset/john/registration_segments/S01_*.wav'))} 个 clean 片段")

    # 测试 clean 环境
    clean_test = "asset/john/test_clean_segments/安静环境测试测试.m4a"
    noisy_test = "asset/john/test_noise_segments/嘈杂环境测试.m4a"

    print(f"\n--- Clean 环境测试 ---")
    clean_results, clean_full = sliding_window_test(client, clean_test, clean_ref)
    clean_max, clean_avg, clean_rate, clean_full = format_results(
        "Clean 环境", clean_results, clean_full, threshold
    )

    # 滑动窗口详细输出（前 10 个 + 最高分）
    top_idx = max(range(len(clean_results)), key=lambda i: clean_results[i]["score"])
    print(f"\n  前 10 个窗口:")
    for r in clean_results[:10]:
        mark = "✅" if r["score"] >= threshold else "  "
        print(f"    {mark} {r['start']:.2f}s -> {r['score']:.4f}")
    if top_idx >= 10:
        print(f"  最高分窗口: {clean_results[top_idx]['start']:.2f}s -> {clean_results[top_idx]['score']:.4f}")

    print(f"\n--- Noisy 环境测试 ---")
    noisy_results, noisy_full = sliding_window_test(client, noisy_test, clean_ref)
    noisy_max, noisy_avg, noisy_rate, noisy_full = format_results(
        "Noisy 环境", noisy_results, noisy_full, threshold
    )

    top_idx = max(range(len(noisy_results)), key=lambda i: noisy_results[i]["score"])
    print(f"\n  前 10 个窗口:")
    for r in noisy_results[:10]:
        mark = "✅" if r["score"] >= threshold else "  "
        print(f"    {mark} {r['start']:.2f}s -> {r['score']:.4f}")
    if top_idx >= 10:
        print(f"  最高分窗口: {noisy_results[top_idx]['start']:.2f}s -> {noisy_results[top_idx]['score']:.4f}")

    # 对比基线（旧参数下的结果）
    print(f"\n{'='*60}")
    print(f"  与基线对比")
    print(f"{'='*60}")
    print(f"  {'指标':<20} {'旧参数 (0.75/tail)':<22} {'新参数 (0.55/full)':<22}")
    print(f"  {'-'*60}")
    print(f"  {'Clean 最高分':<18} {'0.8338':<22} {f'{clean_max:.4f}':<22}")
    print(f"  {'Clean Full':<18} {'0.9278':<22} {f'{clean_full:.4f}':<22}")
    print(f"  {'Clean 通过率':<17} {'42.4%':<22} {f'{clean_rate:.1f}%':<22}")
    print(f"  {'Noisy 最高分':<18} {'0.5558':<22} {f'{noisy_max:.4f}':<22}")
    print(f"  {'Noisy Full':<18} {'0.5522':<22} {f'{noisy_full:.4f}':<22}")
    print(f"  {'Noisy 通过率':<17} {'0.0%':<22} {f'{noisy_rate:.1f}%':<22}")
    print(f"{'='*60}")

    # 判断
    print(f"\n  优化效果:")
    if noisy_full >= threshold:
        print(f"    ✅ Noisy Full Utterance ({noisy_full:.4f}) 已超过阈值 {threshold}")
    else:
        print(f"    ❌ Noisy Full Utterance ({noisy_full:.4f}) 仍未达阈值 {threshold}")

    if noisy_max >= threshold:
        print(f"    ✅ Noisy 滑动窗口最高分 ({noisy_max:.4f}) 已超过阈值 {threshold}")
    else:
        print(f"    ❌ Noisy 滑动窗口最高分 ({noisy_max:.4f}) 仍未达阈值 {threshold}")


if __name__ == "__main__":
    main()
