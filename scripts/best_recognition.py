#!/usr/bin/env python3
"""WeSpeaker 最佳声纹识别配置 — 基于全部实验结果的最优方案。

综合实验结论（见 experiment_log/）：
- 注册增强: 真实噪声 multi-SNR 注入 (20/15/10/5/0 dB) — noisy +0.058, clean -0.038
- 识别阈值: 0.55 — clean 通过率 98.3%
- 裁剪策略: full_utterance + VAD — 完整语音去静音
- Buffer 限制: 8.0s — 防止过长音频

用法:
    # 注册声纹（需要噪声音频目录）
    uv run python scripts/best_recognition.py enroll \
        --clean asset/john/registration_segments/ \
        --noise asset/john/test_noise_segments/嘈杂环境测试.m4a \
        --output asset/john/voice_best.pkl

    # 识别声纹
    uv run python scripts/best_recognition.py recognize \
        --audio asset/john/test_clean_segments/安静环境测试测试.m4a \
        --voiceprint asset/john/voice_best.pkl

    # 滑动窗口测试
    uv run python scripts/best_recognition.py test-sliding \
        --audio asset/john/test_noise_segments/嘈杂环境测试.m4a \
        --voiceprint asset/john/voice_best.pkl
"""

import glob
import os
import pickle
import sys
from pathlib import Path

# 确保能找到 src.wespeaker
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
#  最佳配置常量
# --------------------------------------------------------------------------- #

BEST_CONFIG = {
    "sim_threshold": 0.55,
    "verify_crop_mode": "full_utterance",
    "verify_buffer_keep_secs": 60.0,  # 不截断，使用完整音频
    "verify_window_secs": 1.0,
    "enrollment_segment_secs": 1.0,
    "enable_vad": False,              # 禁用 VAD — 实验表明完整音频得分更高
    "vad_rms_threshold": 0.005,
    "noise_injection_snrs": [20, 15, 10, 5, 0],  # multi-SNR 真实噪声注入
}


# --------------------------------------------------------------------------- #
#  噪声提取与注入
# --------------------------------------------------------------------------- #

def extract_noise_profile(noisy_audio_path: str, sample_rate: int = 16000,
                          rms_threshold: float = 0.005) -> np.ndarray:
    """从噪声音频中提取环境噪声 profile（非语音段）。"""
    waveform = _load_audio(noisy_audio_path, sample_rate)
    speech_segs = _vad_segments(waveform, rms_threshold=rms_threshold,
                                sample_rate=sample_rate)
    if not speech_segs:
        return waveform.cpu().numpy()

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
            if noise_end - noise_start > sample_rate // 10:
                noise_regions.append((noise_start, noise_end))

    if noise_regions:
        parts = [waveform[s:e] for s, e in noise_regions]
        result = torch.cat(parts)
        return result.cpu().numpy()

    return waveform.cpu().numpy()


def mix_noise_at_snr(clean: np.ndarray, noise: np.ndarray,
                     target_snr_db: float) -> np.ndarray:
    """将噪声混合到 clean 音频，达到目标 SNR (dB)。"""
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
#  注册 — 最佳方案: multi-SNR 真实噪声注入
# --------------------------------------------------------------------------- #

def enroll_with_noise_injection(
    client: WespeakerClient,
    clean_dir: str,
    noise_profile: np.ndarray,
    snr_levels: list[float],
) -> torch.Tensor:
    """用 multi-SNR 真实噪声注入方式注册声纹。

    每个 clean 片段在多个 SNR 级别下混合噪声，
    所有 embedding 取均值作为参考声纹。
    """
    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    if not clean_paths:
        clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*")))

    all_embeddings = []
    for path in clean_paths:
        seg = _load_audio(path, client.sample_rate).cpu().numpy()
        for snr in snr_levels:
            mixed = mix_noise_at_snr(seg, noise_profile, snr)
            emb = _extract_embedding(client._model, torch.from_numpy(mixed))
            all_embeddings.append(F.normalize(emb, dim=0))

    return F.normalize(torch.stack(all_embeddings).mean(dim=0), dim=0)


# --------------------------------------------------------------------------- #
#  识别
# --------------------------------------------------------------------------- #

def recognize(client: WespeakerClient, audio_path: str, pk_path: str) -> dict:
    """用最佳配置识别声纹。"""
    if not Path(audio_path).is_file():
        return {"is_recognized": False, "confidence": 0.0, "error": f"文件不存在: {audio_path}"}
    if not Path(pk_path).is_file():
        return {"is_recognized": False, "confidence": 0.0, "error": f"声纹文件不存在: {pk_path}"}

    client._ensure_model()

    with open(pk_path, "rb") as f:
        ref = F.normalize(torch.from_numpy(np.asarray(pickle.load(f), dtype=np.float32)), dim=0)

    waveform = _load_audio(audio_path, client.sample_rate)

    # 限制最大长度
    max_samples = int(client.verify_buffer_keep_secs * client.sample_rate)
    if waveform.numel() > max_samples:
        waveform = waveform[-max_samples:]

    # VAD 去静音
    if client.enable_vad:
        speech_segs = _vad_segments(waveform, rms_threshold=client.vad_rms_threshold,
                                    sample_rate=client.sample_rate)
        if speech_segs:
            pcm = torch.cat(speech_segs)
        else:
            pcm = waveform
    else:
        pcm = waveform

    if pcm.numel() == 0:
        return {"is_recognized": False, "confidence": 0.0, "error": "音频太短"}

    emb = F.normalize(_extract_embedding(client._model, pcm), dim=0)
    score = float(torch.dot(emb, ref).clamp(-1.0, 1.0).item())

    return {
        "is_recognized": score >= client.sim_threshold,
        "confidence": round(score, 4),
        "threshold": client.sim_threshold,
    }


# --------------------------------------------------------------------------- #
#  滑动窗口测试
# --------------------------------------------------------------------------- #

def sliding_window_test(
    client: WespeakerClient,
    audio_path: str,
    reference: torch.Tensor,
    window_secs: float = 2.0,
    step_secs: float = 0.5,
) -> tuple[list[dict], float]:
    """滑动窗口对比 + full utterance 评分。"""
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

    # Full utterance
    full_emb = F.normalize(_extract_embedding(client._model, waveform), dim=0)
    full_score = float(torch.dot(full_emb, reference).clamp(-1.0, 1.0).item())

    return results, round(full_score, 4)


def print_sliding_results(window_results: list[dict], full_score: float,
                          threshold: float, label: str = ""):
    """打印滑动窗口测试结果。"""
    scores = [r["score"] for r in window_results]
    max_score = max(scores)
    max_pos = next(r["start"] for r in window_results if r["score"] == max_score)
    avg_score = float(np.mean(scores))
    passed = sum(1 for s in scores if s >= threshold)
    above_70 = sum(1 for s in scores if s >= 0.70)

    if label:
        print(f"\n{label}")
    print(f"  最高分: {max_score:.4f} (@ {max_pos:.1f}s)")
    print(f"  平均分: {avg_score:.4f}")
    print(f"  >= {threshold:.2f}: {passed}/{len(scores)} ({passed/len(scores)*100:.1f}%)")
    print(f"  >= 0.70: {above_70}/{len(scores)}")
    print(f"  Full utterance: {full_score:.4f} {'✅' if full_score >= threshold else '❌'}")


# --------------------------------------------------------------------------- #
#  CLI 入口
# --------------------------------------------------------------------------- #

def create_client() -> WespeakerClient:
    """用最佳配置创建 client。"""
    return WespeakerClient(
        device="cpu",
        enable_augmentation=False,
        sim_threshold=BEST_CONFIG["sim_threshold"],
        verify_crop_mode=BEST_CONFIG["verify_crop_mode"],
        verify_buffer_keep_secs=BEST_CONFIG["verify_buffer_keep_secs"],
        verify_window_secs=BEST_CONFIG["verify_window_secs"],
        enrollment_segment_secs=BEST_CONFIG["enrollment_segment_secs"],
        enable_vad=BEST_CONFIG["enable_vad"],
        vad_rms_threshold=BEST_CONFIG["vad_rms_threshold"],
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="WeSpeaker 最佳声纹识别配置")
    sub = parser.add_subparsers(dest="cmd")

    # enroll
    p_enroll = sub.add_parser("enroll", help="注册声纹 (multi-SNR 噪声注入)")
    p_enroll.add_argument("--clean", required=True, help="clean 注册片段目录")
    p_enroll.add_argument("--noise", required=True, help="噪声音频文件（用于提取噪声 profile）")
    p_enroll.add_argument("--output", default="voice_best.pkl", help="输出 .pkl 路径")
    p_enroll.add_argument("--snrs", default="20,15,10,5,0", help="SNR 级别，逗号分隔")

    # recognize
    p_recognize = sub.add_parser("recognize", help="识别声纹")
    p_recognize.add_argument("--audio", required=True, help="测试音频")
    p_recognize.add_argument("--voiceprint", required=True, help="声纹 .pkl 文件")

    # test-sliding
    p_test = sub.add_parser("test-sliding", help="滑动窗口测试")
    p_test.add_argument("--audio", required=True, help="测试音频")
    p_test.add_argument("--voiceprint", required=True, help="声纹 .pkl 文件")
    p_test.add_argument("--window", type=float, default=2.0, help="窗口长度 (秒)")
    p_test.add_argument("--step", type=float, default=0.5, help="步长 (秒)")

    args = parser.parse_args()

    if args.cmd == "enroll":
        client = create_client()
        client._ensure_model()

        snr_levels = [float(x.strip()) for x in args.snrs.split(",")]
        print(f"提取噪声 profile: {args.noise}")
        noise_profile = extract_noise_profile(args.noise)
        print(f"噪声 profile: {len(noise_profile)/16000:.1f}s")

        print(f"注册: {args.clean} (SNR levels: {snr_levels})")
        ref = enroll_with_noise_injection(client, args.clean, noise_profile, snr_levels)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(ref.cpu().numpy(), f)
        print(f"声纹已保存: {out.resolve()}")

    elif args.cmd == "recognize":
        client = create_client()
        result = recognize(client, args.audio, args.voiceprint)
        status = "✅ 识别成功" if result["is_recognized"] else "❌ 识别失败"
        print(f"{status}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  阈值:   {result['threshold']:.2f}")
        if "error" in result:
            print(f"  错误:   {result['error']}")

    elif args.cmd == "test-sliding":
        client = create_client()
        client._ensure_model()

        with open(args.voiceprint, "rb") as f:
            ref = F.normalize(torch.from_numpy(
                np.asarray(pickle.load(f), dtype=np.float32)), dim=0)

        window_results, full_score = sliding_window_test(
            client, args.audio, ref,
            window_secs=args.window, step_secs=args.step,
        )
        print_sliding_results(window_results, full_score,
                              threshold=BEST_CONFIG["sim_threshold"],
                              label=f"滑动窗口测试: {args.audio}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
