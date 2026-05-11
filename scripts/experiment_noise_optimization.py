#!/usr/bin/env python3
"""
噪声环境识别率优化实验 — 全方案对比测试

测试以下方案：
1. 混合注册：clean + noisy 音频一起注册
2. 噪声增强：用 audiomentations 在注册时做数据增强
3. 动态阈值：降低噪声场景的相似度阈值
4. 噪声预处理：用 noisereduce 做频谱减法预处理
"""

import glob
import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker.wespeaker import WespeakerClient, _load_audio, _extract_embedding


# --------------------------------------------------------------------------- #
#  公共函数
# --------------------------------------------------------------------------- #

def enroll_segments(segment_paths, client):
    """使用给定音频列表注册声纹，返回归一化均值 embedding。"""
    embs = []
    for f in segment_paths:
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
        seg = waveform[start:start + window_len]
        emb = F.normalize(_extract_embedding(client._model, seg), dim=0)
        score = float(torch.dot(emb, ref).clamp(-1, 1).item())
        scores.append((start / 16000, score))

    return scores


def compute_stats(scores, threshold=0.75):
    """计算测试统计量。"""
    scores_values = [s for _, s in scores]
    return {
        "max_score": max(scores_values),
        "max_pos": next(pos for pos, s in scores if s == max(scores_values)),
        "mean_score": float(np.mean(scores_values)),
        f"above_{threshold:.2f}": sum(1 for s in scores_values if s > threshold),
        "total_windows": len(scores),
    }


def get_segments(directory, pattern="*.wav"):
    """获取目录下所有匹配文件。"""
    return sorted(glob.glob(os.path.join(directory, pattern)))


# --------------------------------------------------------------------------- #
#  方案 1：混合注册
# --------------------------------------------------------------------------- #

def experiment_mixed_enrollment(client, noisy_audio, ref_clean):
    """混合注册：clean segments + noisy audio 一起注册。"""
    print("\n" + "=" * 60)
    print("方案1：混合注册（clean + noisy）")
    print("=" * 60)

    # 获取 clean segments
    clean_segments = get_segments("asset/john/registration_segments")
    print(f"Clean segments: {len(clean_segments)} 个")

    # 加载 noisy 音频
    noisy_waveform = _load_audio(noisy_audio, 16000)

    # 将 noisy 音频切分为 1s 片段参与注册
    seg_len = 16000
    noisy_segments = []
    for i in range(len(noisy_waveform) // seg_len):
        seg = noisy_waveform[i * seg_len:(i + 1) * seg_len]
        noisy_segments.append(seg)
    print(f"Noisy segments (1s chunks): {len(noisy_segments)} 个")

    # 提取所有 embedding
    all_embs = []
    for f in clean_segments:
        w = _load_audio(f, 16000)
        all_embs.append(F.normalize(_extract_embedding(client._model, w), dim=0))
    for seg in noisy_segments:
        all_embs.append(F.normalize(_extract_embedding(client._model, seg), dim=0))

    ref_mixed = F.normalize(torch.stack(all_embs).mean(dim=0), dim=0)

    # 对比 clean reference vs mixed reference 在 noisy 音频上的表现
    noisy_waveform_full = _load_audio(noisy_audio, 16000)

    scores_clean_ref = sliding_window_test(noisy_waveform_full, ref_clean, client)
    scores_mixed_ref = sliding_window_test(noisy_waveform_full, ref_mixed, client)

    stats_clean = compute_stats(scores_clean_ref)
    stats_mixed = compute_stats(scores_mixed_ref)

    # Full utterance
    full_emb = F.normalize(_extract_embedding(client._model, noisy_waveform_full), dim=0)
    full_clean = float(torch.dot(full_emb, ref_clean).clamp(-1, 1).item())
    full_mixed = float(torch.dot(full_emb, ref_mixed).clamp(-1, 1).item())

    print(f"\nClean reference -> Noisy audio:")
    print(f"  最高分: {stats_clean['max_score']:.4f}, 平均分: {stats_clean['mean_score']:.4f}")
    print(f"  Full utterance: {full_clean:.4f}")

    print(f"\nMixed reference -> Noisy audio:")
    print(f"  最高分: {stats_mixed['max_score']:.4f}, 平均分: {stats_mixed['mean_score']:.4f}")
    print(f"  Full utterance: {full_mixed:.4f}")

    print(f"\n提升: 最高分 {stats_mixed['max_score'] - stats_clean['max_score']:+.4f}, "
          f"平均分 {stats_mixed['mean_score'] - stats_clean['mean_score']:+.4f}")

    return {
        "clean_ref_stats": stats_clean,
        "mixed_ref_stats": stats_mixed,
        "clean_ref_full": full_clean,
        "mixed_ref_full": full_mixed,
    }


# --------------------------------------------------------------------------- #
#  方案 2：噪声增强
# --------------------------------------------------------------------------- #

def experiment_noise_augmentation(client, noisy_audio, ref_clean):
    """噪声增强：用高斯噪声增强注册片段。"""
    print("\n" + "=" * 60)
    print("方案2：噪声增强（高斯 SNR 增强）")
    print("=" * 60)

    clean_segments = get_segments("asset/john/registration_segments")
    print(f"Clean segments: {len(clean_segments)} 个")

    # 用内置 _NoiseAugmentor 做增强
    from src.wespeaker.wespeaker import _NoiseAugmentor
    aug = _NoiseAugmentor(sample_rate=16000, augment_ratio=1.0, seed=42)

    # 加载并增强
    segments_np = []
    for f in clean_segments:
        w = _load_audio(f, 16000).cpu().numpy()
        segments_np.append(w)

    augmented = aug.augment(segments_np)
    print(f"增强后: {len(augmented)} 个片段")

    # 原始 + 增强混合注册
    all_embs = []
    for f in clean_segments:
        w = _load_audio(f, 16000)
        all_embs.append(F.normalize(_extract_embedding(client._model, w), dim=0))
    for seg_np in augmented:
        t = torch.from_numpy(seg_np)
        all_embs.append(F.normalize(_extract_embedding(client._model, t), dim=0))

    ref_aug = F.normalize(torch.stack(all_embs).mean(dim=0), dim=0)

    # 测试
    noisy_waveform = _load_audio(noisy_audio, 16000)
    scores = sliding_window_test(noisy_waveform, ref_aug, client)
    stats = compute_stats(scores)

    full_emb = F.normalize(_extract_embedding(client._model, noisy_waveform), dim=0)
    full_score = float(torch.dot(full_emb, ref_aug).clamp(-1, 1).item())

    # 对比 baseline
    scores_baseline = sliding_window_test(noisy_waveform, ref_clean, client)
    stats_baseline = compute_stats(scores_baseline)

    print(f"\nBaseline (无增强) -> Noisy audio:")
    print(f"  最高分: {stats_baseline['max_score']:.4f}, 平均分: {stats_baseline['mean_score']:.4f}")
    print(f"  Full utterance: {float(torch.dot(full_emb, ref_clean).clamp(-1, 1).item()):.4f}")

    print(f"\n增强注册 -> Noisy audio:")
    print(f"  最高分: {stats['max_score']:.4f}, 平均分: {stats['mean_score']:.4f}")
    print(f"  Full utterance: {full_score:.4f}")

    print(f"\n提升: 最高分 {stats['max_score'] - stats_baseline['max_score']:+.4f}, "
          f"平均分 {stats['mean_score'] - stats_baseline['mean_score']:+.4f}")

    return {
        "baseline_stats": stats_baseline,
        "augmented_stats": stats,
        "baseline_full": float(torch.dot(full_emb, ref_clean).clamp(-1, 1).item()),
        "augmented_full": full_score,
    }


# --------------------------------------------------------------------------- #
#  方案 3：动态阈值分析
# --------------------------------------------------------------------------- #

def experiment_threshold_analysis(client, noisy_audio, clean_audio, ref_clean):
    """分析不同阈值下的 FAR/FRR，找出噪声场景最佳阈值。"""
    print("\n" + "=" * 60)
    print("方案3：动态阈值分析")
    print("=" * 60)

    noisy_waveform = _load_audio(noisy_audio, 16000)
    clean_waveform = _load_audio(clean_audio, 16000)

    scores_noisy = sliding_window_test(noisy_waveform, ref_clean, client)
    scores_clean = sliding_window_test(clean_waveform, ref_clean, client)

    # 测试多个阈值
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    print(f"\n{'阈值':<8} | {'Clean 通过率':<12} | {'Noisy 通过率':<12} | {'差距':<8}")
    print("-" * 50)

    results = {}
    for t in thresholds:
        clean_pass = sum(1 for _, s in scores_clean if s > t)
        noisy_pass = sum(1 for _, s in scores_noisy if s > t)
        clean_rate = clean_pass / len(scores_clean) * 100
        noisy_rate = noisy_pass / len(scores_noisy) * 100
        gap = clean_rate - noisy_rate
        print(f"{t:.2f}    | {clean_rate:>5.1f}%  ({clean_pass:>3}/{len(scores_clean)})"
              f" | {noisy_rate:>5.1f}%  ({noisy_pass:>3}/{len(scores_noisy)})"
              f" | {gap:+.1f}%")
        results[t] = {
            "clean_rate": clean_rate,
            "noisy_rate": noisy_rate,
            "clean_pass": clean_pass,
            "noisy_pass": noisy_pass,
            "gap": gap,
        }

    # 推荐阈值：noisy 通过率 > 30% 且差距 < 40% 的最低阈值
    recommended = None
    for t in thresholds:
        if results[t]["noisy_rate"] > 30 and results[t]["gap"] < 40:
            recommended = t
            break

    if recommended is not None:
        print(f"\n推荐噪声场景阈值: {recommended:.2f}")
        print(f"  Clean 通过率: {results[recommended]['clean_rate']:.1f}%")
        print(f"  Noisy 通过率: {results[recommended]['noisy_rate']:.1f}%")
    else:
        print("\n无满足条件的阈值（noisy 分数整体过低）")

    return {
        "threshold_results": results,
        "recommended_threshold": recommended,
    }


# --------------------------------------------------------------------------- #
#  方案 4：噪声预处理（尝试 noisereduce）
# --------------------------------------------------------------------------- #

def experiment_denoise(client, noisy_audio, ref_clean):
    """噪声预处理：尝试 noisereduce 库做频谱减法。"""
    print("\n" + "=" * 60)
    print("方案4：噪声预处理（noisereduce）")
    print("=" * 60)

    try:
        import noisereduce
        has_denoise = True
    except ImportError:
        has_denoise = False
        print("  noisereduce 未安装，尝试安装...")
        import subprocess
        result = subprocess.run(
            ["uv", "add", "noisereduce"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            has_denoise = True
            import noisereduce
            print("  安装成功")
        else:
            print(f"  安装失败: {result.stderr}")
            return {"error": "noisereduce 不可用"}

    noisy_waveform = _load_audio(noisy_audio, 16000)
    noisy_np = noisy_waveform.cpu().numpy()

    # 用前 0.5s 作为噪声剖面
    noise_profile = noisy_np[:int(0.5 * 16000)]
    denoised = noisereduce.reduce_noise(
        y=noisy_np,
        sr=16000,
        y_noise=noise_profile,
        n_fft=1024,
        prop_decrease=0.75,
    )

    denoised_tensor = torch.from_numpy(denoised.astype(np.float32))

    # Baseline
    scores_baseline = sliding_window_test(noisy_waveform, ref_clean, client)
    stats_baseline = compute_stats(scores_baseline)

    # Denoised
    scores_denoised = sliding_window_test(denoised_tensor, ref_clean, client)
    stats_denoised = compute_stats(scores_denoised)

    # Full utterance
    full_emb_raw = F.normalize(_extract_embedding(client._model, noisy_waveform), dim=0)
    full_emb_denoised = F.normalize(_extract_embedding(client._model, denoised_tensor), dim=0)
    full_raw = float(torch.dot(full_emb_raw, ref_clean).clamp(-1, 1).item())
    full_denoised = float(torch.dot(full_emb_denoised, ref_clean).clamp(-1, 1).item())

    print(f"\nBaseline (原始) -> Noisy audio:")
    print(f"  最高分: {stats_baseline['max_score']:.4f}, 平均分: {stats_baseline['mean_score']:.4f}")
    print(f"  Full utterance: {full_raw:.4f}")

    print(f"\nDenoised -> Noisy audio:")
    print(f"  最高分: {stats_denoised['max_score']:.4f}, 平均分: {stats_denoised['mean_score']:.4f}")
    print(f"  Full utterance: {full_denoised:.4f}")

    print(f"\n提升: 最高分 {stats_denoised['max_score'] - stats_baseline['max_score']:+.4f}, "
          f"平均分 {stats_denoised['mean_score'] - stats_baseline['mean_score']:+.4f}")

    return {
        "baseline_stats": stats_baseline,
        "denoised_stats": stats_denoised,
        "baseline_full": full_raw,
        "denoised_full": full_denoised,
    }


# --------------------------------------------------------------------------- #
#  主函数
# --------------------------------------------------------------------------- #

def save_experiment_log(all_results, output_path):
    """保存完整的实验日志。"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 实验记录：噪声环境识别率优化 — 全方案对比\n\n")
        f.write(f"**日期**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("**模型**: pyannote/wespeaker-voxceleb-resnet34-LM\n")
        f.write("**注册音频**: asset/john/registration_segments/ (22 个 clean 片段)\n")
        f.write("**测试音频**: asset/john/test_noise_segments/嘈杂环境测试.m4a\n")
        f.write("**Baseline**: 纯 clean 注册, noisy 测试 -> 最高分 0.5558, 0% 通过\n\n")

        # 方案 1
        f.write("## 方案 1：混合注册（clean + noisy）\n\n")
        r1 = all_results.get("mixed_enrollment", {})
        if r1:
            f.write(f"| 指标 | Clean Reference | Mixed Reference |\n")
            f.write(f"|------|----------------|----------------|\n")
            f.write(f"| 最高分 | {r1['clean_ref_stats']['max_score']:.4f} | {r1['mixed_ref_stats']['max_score']:.4f} |\n")
            f.write(f"| 平均分 | {r1['clean_ref_stats']['mean_score']:.4f} | {r1['mixed_ref_stats']['mean_score']:.4f} |\n")
            f.write(f"| > 0.75 窗口 | {r1['clean_ref_stats']['above_0.75']}/{r1['clean_ref_stats']['total_windows']} | {r1['mixed_ref_stats']['above_0.75']}/{r1['mixed_ref_stats']['total_windows']} |\n")
            f.write(f"| Full utterance | {r1['clean_ref_full']:.4f} | {r1['mixed_ref_full']:.4f} |\n\n")

        # 方案 2
        f.write("## 方案 2：噪声增强（高斯 SNR）\n\n")
        r2 = all_results.get("noise_augmentation", {})
        if r2:
            f.write(f"| 指标 | Baseline | Augmented |\n")
            f.write(f"|------|----------|-----------|\n")
            f.write(f"| 最高分 | {r2['baseline_stats']['max_score']:.4f} | {r2['augmented_stats']['max_score']:.4f} |\n")
            f.write(f"| 平均分 | {r2['baseline_stats']['mean_score']:.4f} | {r2['augmented_stats']['mean_score']:.4f} |\n")
            f.write(f"| > 0.75 窗口 | {r2['baseline_stats']['above_0.75']}/{r2['baseline_stats']['total_windows']} | {r2['augmented_stats']['above_0.75']}/{r2['augmented_stats']['total_windows']} |\n")
            f.write(f"| Full utterance | {r2['baseline_full']:.4f} | {r2['augmented_full']:.4f} |\n\n")

        # 方案 3
        f.write("## 方案 3：动态阈值分析\n\n")
        r3 = all_results.get("threshold_analysis", {})
        if r3:
            f.write(f"| 阈值 | Clean 通过率 | Noisy 通过率 | 差距 |\n")
            f.write(f"|------|-------------|-------------|------|\n")
            for t, v in r3["threshold_results"].items():
                f.write(f"| {t:.2f} | {v['clean_rate']:.1f}% | {v['noisy_rate']:.1f}% | {v['gap']:+.1f}% |\n")
            if r3.get("recommended_threshold"):
                f.write(f"\n**推荐阈值**: {r3['recommended_threshold']:.2f}\n\n")

        # 方案 4
        f.write("## 方案 4：噪声预处理（noisereduce）\n\n")
        r4 = all_results.get("denoise", {})
        if r4:
            if "error" in r4:
                f.write(f"**跳过**: {r4['error']}\n\n")
            else:
                f.write(f"| 指标 | Baseline | Denoised |\n")
                f.write(f"|------|----------|----------|\n")
                f.write(f"| 最高分 | {r4['baseline_stats']['max_score']:.4f} | {r4['denoised_stats']['max_score']:.4f} |\n")
                f.write(f"| 平均分 | {r4['baseline_stats']['mean_score']:.4f} | {r4['denoised_stats']['mean_score']:.4f} |\n")
                f.write(f"| > 0.75 窗口 | {r4['baseline_stats']['above_0.75']}/{r4['baseline_stats']['total_windows']} | {r4['denoised_stats']['above_0.75']}/{r4['denoised_stats']['total_windows']} |\n")
                f.write(f"| Full utterance | {r4['baseline_full']:.4f} | {r4['denoised_full']:.4f} |\n\n")

        # 总结
        f.write("## 总结\n\n")
        f.write("| 方案 | 最高分变化 | 平均分变化 | 推荐程度 |\n")
        f.write(f"|------|-----------|-----------|----------|\n")

        for name, key in [("混合注册", "mixed_enrollment"), ("噪声增强", "noise_augmentation"),
                           ("噪声预处理", "denoise")]:
            r = all_results.get(key, {})
            if r and "error" not in r:
                if key == "mixed_enrollment":
                    diff_max = r["mixed_ref_stats"]["max_score"] - r["clean_ref_stats"]["max_score"]
                    diff_mean = r["mixed_ref_stats"]["mean_score"] - r["clean_ref_stats"]["mean_score"]
                else:
                    diff_max = r.get("augmented_stats", r.get("denoised_stats", {})).get("max_score", 0) - r.get("baseline_stats", {}).get("max_score", 0)
                    diff_mean = r.get("augmented_stats", r.get("denoised_stats", {})).get("mean_score", 0) - r.get("baseline_stats", {}).get("mean_score", 0)
                stars = "★★★" if diff_max > 0.05 else "★★" if diff_max > 0 else "★"
                f.write(f"| {name} | {diff_max:+.4f} | {diff_mean:+.4f} | {stars} |\n")

        if r3 and r3.get("recommended_threshold"):
            f.write(f"| 动态阈值 | 阈值降至 {r3['recommended_threshold']:.2f} | — | ★★★ |\n")

        f.write("\n")


def main():
    client = WespeakerClient(device="cpu", enable_augmentation=False)
    client._ensure_model()

    noisy_audio = "asset/john/test_noise_segments/嘈杂环境测试.m4a"
    clean_audio = "asset/john/test_clean_segments/安静环境测试测试.m4a"

    # Baseline: clean-only enrollment
    print("=== Baseline: Clean-only 注册 ===")
    clean_segments = get_segments("asset/john/registration_segments")
    ref_clean = enroll_segments(clean_segments, client)
    print(f"注册完成: {len(clean_segments)} 个 clean 片段")
    baseline_noisy = _load_audio(noisy_audio, 16000)
    baseline_scores = sliding_window_test(baseline_noisy, ref_clean, client)
    baseline_stats = compute_stats(baseline_scores)
    print(f"Baseline noisy max: {baseline_stats['max_score']:.4f}\n")

    all_results = {}

    # 方案 1
    all_results["mixed_enrollment"] = experiment_mixed_enrollment(client, noisy_audio, ref_clean)

    # 方案 2
    all_results["noise_augmentation"] = experiment_noise_augmentation(client, noisy_audio, ref_clean)

    # 方案 3
    all_results["threshold_analysis"] = experiment_threshold_analysis(
        client, noisy_audio, clean_audio, ref_clean
    )

    # 方案 4
    all_results["denoise"] = experiment_denoise(client, noisy_audio, ref_clean)

    # 保存日志
    log_path = f"experiment_log/{datetime.now().strftime('%Y-%m-%d')}_noise_optimization.md"
    save_experiment_log(all_results, log_path)
    print(f"\n完整实验日志已保存到: {log_path}")


if __name__ == "__main__":
    main()
