#!/usr/bin/env python3
"""WeSpeaker 最佳声纹识别配置 — CLI 入口。

实现逻辑已迁移至 src/wespeaker/best.py (WespeakerBest)。
本脚本仅提供命令行接口和滑动窗口诊断工具。

用法:
    uv run python scripts/best_recognition.py enroll \
        --clean asset/john/registration_segments/ \
        --noise asset/john/test_noise_segments/嘈杂环境测试.m4a \
        --output asset/john/voice_best.pkl

    uv run python scripts/best_recognition.py recognize \
        --audio asset/john/test_clean_segments/安静环境测试测试.m4a \
        --voiceprint asset/john/voice_best.pkl

    uv run python scripts/best_recognition.py test-sliding \
        --audio asset/john/test_noise_segments/嘈杂环境测试.m4a \
        --voiceprint asset/john/voice_best.pkl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker import BestConfig, WespeakerBest
from src.wespeaker.wespeaker import _extract_embedding, _load_audio

# --------------------------------------------------------------------------- #
#  滑动窗口诊断工具（保留在脚本中，不属于核心库）
# --------------------------------------------------------------------------- #


def sliding_window_test(
    recognizer: WespeakerBest,
    audio_path: str,
    reference: torch.Tensor,
    window_secs: float = 2.0,
    step_secs: float = 0.5,
) -> tuple[list[dict], float]:
    """滑动窗口对比 + full utterance 评分。"""
    waveform = _load_audio(audio_path, recognizer._client.sample_rate)
    sr = recognizer._client.sample_rate
    total_secs = len(waveform) / sr
    window_samples = int(window_secs * sr)
    step_samples = int(step_secs * sr)

    results = []
    pos = 0.0
    while pos + window_secs <= total_secs:
        start = int(pos * sr)
        end = start + window_samples
        segment = waveform[start:end]
        emb = F.normalize(_extract_embedding(recognizer._client._model, segment), dim=0)
        score = float(torch.dot(emb, reference).clamp(-1.0, 1.0).item())
        results.append({"start": round(pos, 2), "score": round(score, 4)})
        pos += step_secs

    full_emb = F.normalize(_extract_embedding(recognizer._client._model, waveform), dim=0)
    full_score = float(torch.dot(full_emb, reference).clamp(-1.0, 1.0).item())

    return results, round(full_score, 4)


def print_sliding_results(
    window_results: list[dict], full_score: float, threshold: float, label: str = ""
) -> None:
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="WeSpeaker 最佳声纹识别配置")
    sub = parser.add_subparsers(dest="cmd")

    # enroll
    p_enroll = sub.add_parser("enroll", help="注册声纹 (multi-SNR 噪声注入)")
    p_enroll.add_argument("--clean", required=True, help="clean 注册片段目录")
    p_enroll.add_argument("--noise", required=True, help="噪声音频文件")
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
        recognizer = WespeakerBest()
        snr_levels = [float(x.strip()) for x in args.snrs.split(",")]

        print(f"提取噪声 profile: {args.noise}")
        noise_profile = WespeakerBest.extract_noise_profile(args.noise)
        print(f"噪声 profile: {len(noise_profile)/16000:.1f}s")

        print(f"注册: {args.clean} (SNR levels: {snr_levels})")
        result = recognizer.enroll(args.clean, noise_profile, args.output, snr_levels)
        print(f"声纹已保存: {result['pk_path']} (维度: {result['embedding_dim']})")

    elif args.cmd == "recognize":
        recognizer = WespeakerBest()
        result = recognizer.recognize(args.audio, args.voiceprint)
        status = "✅ 识别成功" if result["is_recognized"] else "❌ 识别失败"
        print(f"{status}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  阈值:   {result['threshold']:.2f}")
        if "error" in result:
            print(f"  错误:   {result['error']}")

    elif args.cmd == "test-sliding":
        recognizer = WespeakerBest()
        recognizer._client._ensure_model()

        ref_data = recognizer.load(args.voiceprint)
        ref = F.normalize(torch.from_numpy(ref_data.astype(np.float32)), dim=0)

        window_results, full_score = sliding_window_test(
            recognizer,
            args.audio,
            ref,
            window_secs=args.window,
            step_secs=args.step,
        )
        print_sliding_results(
            window_results,
            full_score,
            threshold=recognizer.config.sim_threshold,
            label=f"滑动窗口测试: {args.audio}",
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
