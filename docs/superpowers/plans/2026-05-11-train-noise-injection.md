# 训练阶段噪声注入实验 — 完整实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在注册（训练）阶段将真实环境噪声注入 clean 片段，验证是否能提升噪声环境下的识别率。

**Architecture:** 从噪声测试音频中提取噪声 profile（非语音段），以不同 SNR 级别混合到 clean 注册片段中，生成噪声感知的参考声纹。

**Tech Stack:** Python, torchaudio, numpy, pyannote.audio

---

## 背景

### 已失败的方法

| 方法 | 效果 | 失败原因 |
|------|------|----------|
| Gaussian SNR 增强 | +0.0091 | 频谱与真实环境噪声差异太大 |
| 谱减法去噪 | -0.0522 | 破坏了语音自然特征 |

### 唯一有效但不足的方法

| 方法 | 效果 | 局限性 |
|------|------|--------|
| 混合注册 (clean + noisy 1s chunk) | full utterance 0.55→0.81 | 滑动窗口仅 0.68，不够 |

### 本实验的核心假设

用 **真实环境噪声** 替代高斯噪声进行注册增强，能更好地匹配测试时的噪声特征。
前一个实验用 `AddGaussianSNR(min_snr_db=5, max_snr_db=20)` 增强 clean 片段但效果微乎其微，
原因是高斯噪声的频谱平坦，而真实环境噪声有特定的频谱结构（如空调低频、键盘中频等）。

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `scripts/experiment_train_noise_injection.py` | **新建** | 训练阶段噪声注入实验主脚本 |

## 实验设计

### 噪声提取策略

1. **静音段噪声**: 从 `嘈杂环境测试.m4a` 中用 VAD 检测静音段（RMS < threshold 的帧），
   这些段只包含环境噪声不含语音，直接提取为噪声 profile
2. **如果静音段不足 1s**: 用整段音频减去 VAD 检测到的语音段，剩余部分作为噪声估计

### 噪声混合

对每个 clean 注册片段，创建多个副本：
- 原始 clean（不加噪声）
- 混合噪声 @ SNR=20dB（轻微噪声）
- 混合噪声 @ SNR=15dB
- 混合噪声 @ SNR=10dB
- 混合噪声 @ SNR=5dB（中等噪声）
- 混合噪声 @ SNR=0dB（强噪声）

### 对比方案

| 方案 | 描述 | 预期 |
|------|------|------|
| Baseline | 纯 clean 注册 | max ~0.55, full ~0.55 |
| 高斯增强 | audiomentations Gaussian SNR (复现) | max ~0.56, full ~0.60 |
| **噪声注入 (multi-SNR)** | 真实噪声混合到 clean 片段, 所有 SNR 副本平均 | 预期 > 0.70 |
| **噪声注入 (SNR=10dB)** | 仅 10dB 混合 | 测试单一 SNR 的效果 |
| **噪声注入 (SNR=5dB)** | 仅 5dB 混合 | 测试更强噪声的效果 |

---

### Task 1: 实现噪声提取 + 混合函数 + 完整实验

**Files:**
- Create: `scripts/experiment_train_noise_injection.py`

- [ ] **Step 1: 编写完整实验脚本**

创建 `scripts/experiment_train_noise_injection.py`，包含以下功能：

```python
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
    如果非语音段 < 1s，则用整段减去语音段来估计噪声。
    """
    waveform = _load_audio(noisy_audio_path, sample_rate)

    # 检测语音段
    speech_segs = _vad_segments(waveform, rms_threshold=rms_threshold,
                                sample_rate=sample_rate)

    if not speech_segs:
        # 没有检测到语音，整段当作噪声
        return waveform.cpu().numpy()

    # 构建非语音段（从原始波形中扣除语音段）
    speech_regions = []
    for seg in speech_segs:
        # 找到 seg 在 waveform 中的起止位置
        seg_len = len(seg)
        # 简单方法：用 RMS 找匹配
        for start in range(0, len(waveform) - seg_len, sample_rate // 10):
            candidate = waveform[start:start + seg_len]
            if torch.abs(candidate - seg).mean() < 1e-6:
                speech_regions.append((start, start + seg_len))
                break

    # 合并重叠区域
    speech_regions.sort()
    merged = [speech_regions[0]] if speech_regions else []
    for start, end in speech_regions[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # 提取非语音段
    noise_parts = []
    last_end = 0
    for start, end in merged:
        if start > last_end:
            noise_parts.append(waveform[last_end:start])
        last_end = end
    if last_end < len(waveform):
        noise_parts.append(waveform[last_end:])

    if noise_parts:
        return torch.cat(noise_parts).cpu().numpy()

    # 兜底：用整段音频
    return waveform.cpu().numpy()


def mix_noise_at_snr(clean: np.ndarray, noise: np.ndarray,
                     target_snr_db: float, sample_rate: int = 16000) -> np.ndarray:
    """将噪声混合到 clean 音频中，达到目标 SNR。

    SNR = 10 * log10(P_signal / P_noise)
    P_noise_new = P_signal / 10^(SNR/10)
    noise_scale = sqrt(P_noise_new / P_noise)
    """
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-12:
        return clean.copy()

    target_noise_power = clean_power / (10 ** (target_snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    # 确保噪声和 clean 等长
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
    """用噪声注入方式注册声纹。

    对每个 clean 片段，以每个 SNR 级别混合噪声，提取 embedding，
    最后取所有 embedding 的均值并归一化。
    """
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
    # 原始
    for seg in clean_segments:
        emb = _extract_embedding(client._model, torch.from_numpy(seg))
        all_embeddings.append(F.normalize(emb, dim=0))
    # 增强
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

    print(f"训练阶段噪声注入实验")
    print(f"模型: pyannote/wespeaker-voxceleb-resnet34-LM")
    print(f"阈值: {threshold}")
    print(f"噪声来源: asset/john/test_noise_segments/嘈杂环境测试.m4a")

    # 加载 clean 注册片段
    clean_paths = sorted(glob.glob("asset/john/registration_segments/*.wav"))
    clean_segments = []
    for p in clean_paths:
        w = _load_audio(p, 16000)
        clean_segments.append(w.cpu().numpy())
    print(f"\nClean 注册片段: {len(clean_segments)} 个")

    # 提取噪声 profile
    noisy_path = "asset/john/test_noise_segments/嘈杂环境测试.m4a"
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
    all_results["baseline_noisy"] = print_results(f"Noisy", sw_noisy, full_noisy, threshold)
    all_results["baseline_clean"] = print_results(f"Clean", sw_clean, full_clean, threshold)

    # ---- 复现: 高斯增强 ----
    print(f"\n{'='*60}")
    print("方案 1: 高斯 SNR 增强 (复现)")
    print(f"{'='*60}")
    ref_gaussian = enroll_gaussian_augmented(client, clean_segments)
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_gaussian)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_gaussian)
    all_results["gaussian_noisy"] = print_results(f"Noisy", sw_noisy, full_noisy, threshold)
    all_results["gaussian_clean"] = print_results(f"Clean", sw_clean, full_clean, threshold)

    # ---- 方案 2: 噪声注入 (multi-SNR: 20, 15, 10, 5, 0 dB) ----
    print(f"\n{'='*60}")
    print("方案 2: 噪声注入 (multi-SNR: 20, 15, 10, 5, 0 dB)")
    print(f"{'='*60}")
    snr_levels_multi = [20, 15, 10, 5, 0]
    ref_multi = enroll_noise_injected(client, clean_segments, noise_profile, snr_levels_multi)
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_multi)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_multi)
    all_results["multi_noisy"] = print_results(f"Noisy", sw_noisy, full_noisy, threshold)
    all_results["multi_clean"] = print_results(f"Clean", sw_clean, full_clean, threshold)

    # ---- 方案 3: 噪声注入 (仅 SNR=10dB) ----
    print(f"\n{'='*60}")
    print("方案 3: 噪声注入 (SNR=10dB only)")
    print(f"{'='*60}")
    ref_10db = enroll_noise_injected(client, clean_segments, noise_profile, [10])
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_10db)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_10db)
    all_results["10db_noisy"] = print_results(f"Noisy", sw_noisy, full_noisy, threshold)
    all_results["10db_clean"] = print_results(f"Clean", sw_clean, full_clean, threshold)

    # ---- 方案 4: 噪声注入 (仅 SNR=5dB) ----
    print(f"\n{'='*60}")
    print("方案 4: 噪声注入 (SNR=5dB only)")
    print(f"{'='*60}")
    ref_5db = enroll_noise_injected(client, clean_segments, noise_profile, [5])
    sw_noisy, full_noisy = sliding_window_test(client, noisy_test_path, ref_5db)
    sw_clean, full_clean = sliding_window_test(client, clean_test_path, ref_5db)
    all_results["5db_noisy"] = print_results(f"Noisy", sw_noisy, full_noisy, threshold)
    all_results["5db_clean"] = print_results(f"Clean", sw_clean, full_clean, threshold)

    # ---- 方案 5: 噪声注入 (原始 + multi-SNR, 加权) ----
    print(f"\n{'='*60}")
    print("方案 5: 噪声注入 (原始 + multi-SNR, SNR 加权)")
    print(f"{'='*60}")
    # clean 片段 + 各 SNR 混合副本，用 SNR 加权
    all_embeddings_weighted = []
    snr_weights_list = []
    for clean_seg in clean_segments:
        # 原始
        emb = _extract_embedding(client._model, torch.from_numpy(clean_seg))
        all_embeddings_weighted.append(F.normalize(emb, dim=0))
        snr_weights_list.append(30.0)  # clean 给高 SNR 权重
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
    all_results["weighted_noisy"] = print_results(f"Noisy", sw_noisy, full_noisy, threshold)
    all_results["weighted_clean"] = print_results(f"Clean", sw_clean, full_clean, threshold)

    # ---- 汇总表 ----
    print(f"\n\n{'='*80}")
    print(f"  汇总对比 (Noisy 环境)")
    print(f"{'='*80}")
    print(f"  {'方案':<30} {'最高分':>8} {'平均分':>8} {'>= {threshold:.2f}':>10} "
          f"{'>= 0.70':>8} {'Full':>8}")
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
        name = key.replace("_clean", "").replace("baseline", "Baseline")
        print(f"  {name:<30} {r['max_score']:>8.4f} {r['full_score']:>8.4f} "
              f"{max_degrade:>+8.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行实验脚本**

```bash
cd /Users/john/Documents/project/python/wespeaker
uv run python scripts/experiment_train_noise_injection.py
```

预期运行时间 5-15 分钟（每个方案需要多次 embedding 提取）。

- [ ] **Step 3: 记录结果到 experiment_log**

将脚本输出的汇总表写入新的实验记录文件：

```bash
# 复制输出到 experiment_log/2026-05-11_train_noise_injection.md
```

记录格式参照 `experiment_log/2026-05-11_full_experiment_report.md`。

- [ ] **Step 4: 更新项目文档**

如果实验效果良好，更新 `CLAUDE.md` 中的核心 API 表和默认参数。
更新 `docs/diagrams/roadmap.md` 中的实验状态。

---

## Self-Review

### 1. Spec Coverage

用户要求：在训练阶段加入噪声，试试识别效果。

- ✅ Task 1 Step 1: 实现了从噪声音频提取 profile、混合到 clean 片段、多 SNR 级别注册
- ✅ Task 1 Step 1: 包含 6 种对比方案（Baseline、高斯增强复现、multi-SNR 注入、SNR=10dB、SNR=5dB、加权）
- ✅ Task 1 Step 1: 包含 Clean 环境的退化检查
- ✅ Task 1 Step 2-3: 运行并记录结果

### 2. Placeholder Scan

无 TBD/TODO/fill in。所有代码步骤包含完整代码。

### 3. Type Consistency

- 所有函数签名使用 `np.ndarray`, `torch.Tensor`, `str`, `float`, `int`, `list[float]`
- `sliding_window_test` 返回 `tuple[list[dict], float]` 与现有脚本一致
- `print_results` 返回 `dict` 与 `experiment_noise_optimization.py` 中的 `compute_stats` 兼容
- 导入路径 `from src.wespeaker.wespeaker import ...` 与已有脚本一致

---

Plan complete. Saved to `docs/superpowers/plans/2026-05-11-train-noise-injection.md`.

**两种执行方式：**

**1. Subagent-Driven（推荐）** — 每个 task 派一个子代理执行，中间有 review 环节，迭代快
**2. Inline Execution** — 在当前 session 中直接执行

**哪种方式？**
