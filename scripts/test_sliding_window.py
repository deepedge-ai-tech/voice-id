#!/usr/bin/env python3
"""
使用所有注册片段进行声纹注册，然后用滑动窗口测试识别。
支持测试多个音频文件并对比结果。
"""

import glob
import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker.wespeaker import WespeakerClient, _extract_embedding, _load_audio


def enroll_segments(segment_dir, client):
    """使用目录下所有片段注册声纹。"""
    segment_files = sorted(glob.glob(os.path.join(segment_dir, "S01_clean_*_free.wav")))
    print(f"找到 {len(segment_files)} 个片段")

    embs = []
    for f in segment_files:
        waveform = _load_audio(f, 16000)
        emb = F.normalize(_extract_embedding(client._model, waveform), dim=0)
        embs.append(emb)

    ref = F.normalize(torch.stack(embs).mean(dim=0), dim=0)
    return ref


def sliding_window_test(waveform, ref, client, window_secs=2.0, step_secs=0.5):
    """滑动窗口测试，返回 (位置, 分数) 列表。"""
    window_len = int(window_secs * 16000)
    step_len = int(step_secs * 16000)

    scores = []
    for start in range(0, len(waveform) - window_len + 1, step_len):
        seg = waveform[start : start + window_len]
        emb = F.normalize(_extract_embedding(client._model, seg), dim=0)
        score = float(torch.dot(emb, ref).clamp(-1, 1).item())
        scores.append((start / 16000, score))

    return scores


def format_test_results(name, duration, scores, threshold=0.75):
    """格式化测试结果。"""
    scores_values = [s for _, s in scores]
    max_score = max(scores_values)
    max_pos = next(pos for pos, s in scores if s == max_score)
    mean_score = np.mean(scores_values)

    result = {
        "name": name,
        "duration": duration,
        "max_score": max_score,
        "max_pos": max_pos,
        "mean_score": mean_score,
        "above_0.75": sum(1 for s in scores_values if s > 0.75),
        "above_0.70": sum(1 for s in scores_values if s > 0.70),
        "above_0.65": sum(1 for s in scores_values if s > 0.65),
        "total_windows": len(scores),
        "full_utterance": None,
        "top_10": sorted(scores, key=lambda x: x[1], reverse=True)[:10],
    }
    return result


def save_log(results, output_path):
    """保存测试日志。"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 实验记录：滑动窗口对比测试\n\n")
        f.write(f"**日期**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("**模型**: pyannote/wespeaker-voxceleb-resnet34-LM\n")
        f.write("**注册音频**: asset/john/registration_segments/ (22 个片段)\n")
        f.write("**注册总时长**: ~75.8s 有效语音\n")
        f.write("**滑动窗口**: 2.0s 窗口, 0.5s 步长\n")
        f.write("**阈值**: 0.75\n\n")

        f.write("## 测试音频\n\n")
        f.write("| 文件名 | 时长 | 场景 |\n")
        f.write("|--------|------|------|\n")
        for r in results:
            scene = "clean" if "clean" in r["name"].lower() or "测试" not in r["name"] else "noisy"
            f.write(f"| {r['name']} | {r['duration']:.2f}s | {scene} |\n")
        f.write("\n")

        f.write("## 对比结果\n\n")
        f.write("| 指标")
        for r in results:
            f.write(f" | {r['name']}")
        f.write(" |\n")
        f.write("|------")
        for _ in results:
            f.write("|------")
        f.write("|\n")

        metrics = [
            ("最高分", "max_score", "{:.4f}"),
            ("最高分位置", "max_pos", "{:.2f}s"),
            ("平均分", "mean_score", "{:.4f}"),
            ("> 0.75 窗口数", "above_0.75", "{}"),
            ("> 0.70 窗口数", "above_0.70", "{}"),
            ("> 0.65 窗口数", "above_0.65", "{}"),
            ("总窗口数", "total_windows", "{}"),
            ("Full utterance", "full_utterance", "{:.4f}"),
        ]

        for label, key, fmt in metrics:
            f.write(f"| {label}")
            for r in results:
                val = r[key]
                if val is None:
                    f.write(" | -")
                else:
                    f.write(f" | {fmt.format(val)}")
            f.write(" |\n")

        f.write("\n## 详细结果\n\n")
        for r in results:
            f.write(f"### {r['name']}\n\n")
            f.write(f"- **时长**: {r['duration']:.2f}s\n")
            f.write(f"- **最高分**: {r['max_score']:.4f} (位置: {r['max_pos']:.2f}s)\n")
            f.write(f"- **平均分**: {r['mean_score']:.4f}\n")
            f.write(f"- **窗口通过率**:\n")
            f.write(
                f"  - > 0.75: {r['above_0.75']}/{r['total_windows']} ({100*r['above_0.75']/r['total_windows']:.1f}%)\n"
            )
            f.write(
                f"  - > 0.70: {r['above_0.70']}/{r['total_windows']} ({100*r['above_0.70']/r['total_windows']:.1f}%)\n"
            )
            f.write(
                f"  - > 0.65: {r['above_0.65']}/{r['total_windows']} ({100*r['above_0.65']/r['total_windows']:.1f}%)\n"
            )

            if r["full_utterance"] is not None:
                flag = "PASS" if r["full_utterance"] > 0.75 else "FAIL"
                f.write(f"- **Full utterance**: {r['full_utterance']:.4f} [{flag}]\n")

            f.write(f"\n**Top 5 窗口**:\n\n")
            f.write("| 排名 | 时间范围 | 分数 | 结果 |\n")
            f.write("|------|---------|------|------|\n")
            for i, (pos, score) in enumerate(r["top_10"][:5], 1):
                flag = "PASS" if score > 0.75 else ""
                f.write(f"| {i} | {pos:.2f}s - {pos+2.0:.2f}s | {score:.4f} | {flag} |\n")
            f.write("\n")

        f.write("## 结论\n\n")
        if len(results) >= 2:
            clean = results[0]
            noisy = results[1]
            diff_max = clean["max_score"] - noisy["max_score"]
            diff_mean = clean["mean_score"] - noisy["mean_score"]
            f.write(
                f"1. **噪音影响**：嘈杂环境下最高分下降 {diff_max:.4f} ({clean['max_score']:.4f} → {noisy['max_score']:.4f})\n"
            )
            f.write(
                f"2. **平均分下降**：{diff_mean:.4f} ({clean['mean_score']:.4f} → {noisy['mean_score']:.4f})\n"
            )
            f.write(
                f"3. **通过率变化**：>0.75 从 {clean['above_0.75']}/{clean['total_windows']} 变为 {noisy['above_0.75']}/{noisy['total_windows']}\n"
            )
            f.write(
                f"4. **Full utterance 对比**：{clean['full_utterance']:.4f} → {noisy['full_utterance']:.4f}\n"
            )


def main():
    client = WespeakerClient(device="cpu", enable_augmentation=False)
    client._ensure_model()

    # Step 1: Enroll
    print("=== Step 1: 注册声纹 ===")
    ref = enroll_segments("asset/john/registration_segments", client)
    print(f"注册完成，embedding 维度: {ref.shape[0]}")

    with open("asset/john/voice.pkl", "wb") as fp:
        pickle.dump(ref.cpu().numpy(), fp)
    print("已保存到 asset/john/voice.pkl\n")

    # Step 2: Test multiple audio files
    test_files = [
        ("asset/john/test_clean_segments/安静环境测试测试.m4a", "test_clean"),
        ("asset/john/test_noise_segments/嘈杂环境测试.m4a", "test_noisy"),
    ]

    results = []
    for audio_path, name in test_files:
        print(f"=== 测试: {name} ===")
        waveform = _load_audio(audio_path, 16000)
        duration = len(waveform) / 16000
        print(f"音频时长: {duration:.2f}s")

        scores = sliding_window_test(waveform, ref, client, window_secs=2.0, step_secs=0.5)
        result = format_test_results(name, duration, scores)

        # Full utterance test
        full_emb = F.normalize(_extract_embedding(client._model, waveform), dim=0)
        result["full_utterance"] = float(torch.dot(full_emb, ref).clamp(-1, 1).item())

        results.append(result)
        print(f"最高分: {result['max_score']:.4f}, 平均分: {result['mean_score']:.4f}")
        print(f"Full utterance: {result['full_utterance']:.4f}\n")

    # Step 3: Save log
    log_path = f"experiment_log/{datetime.now().strftime('%Y-%m-%d')}_sliding_window_comparison.md"
    save_log(results, log_path)
    print(f"测试日志已保存到: {log_path}")

    # Print summary
    print("\n=== 对比总结 ===")
    print(f"{'指标':<15} | {'test_clean':<15} | {'test_noisy':<15}")
    print("-" * 50)
    for r in results:
        print(f"最高分           | {r['max_score']:.4f}           | -")
        print(f"平均分           | {r['mean_score']:.4f}           | -")
        print(f"Full utterance   | {r['full_utterance']:.4f}           | -")


if __name__ == "__main__":
    main()
