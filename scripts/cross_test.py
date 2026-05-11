#!/usr/bin/env python3
"""声纹交叉测试 — Frank vs John 2x2 识别矩阵。

测试场景:
  注册: Frank (registration_segments), John (registration_segments)
  测试: Frank (frank 测试.m4a), John (安静环境测试测试.m4a)

预期:
  - Frank 测试 vs Frank 声纹 → 通过 (sim >= threshold)
  - John 测试 vs John 声纹 → 通过
  - Frank 测试 vs John 声纹 → 拒绝 (sim < threshold)
  - John 测试 vs Frank 声纹 → 拒绝

用法:
    uv run python scripts/cross_test.py
    uv run python scripts/cross_test.py --noise asset/john/test_noise_segments/嘈杂环境测试.m4a
    uv run python scripts/cross_test.py --snrs 20,15,10,5,0
    uv run python scripts/cross_test.py --threshold 0.50
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker import WespeakerBest

# --------------------------------------------------------------------------- #
#  测试配置
# --------------------------------------------------------------------------- #

SPEAKERS = {
    "Frank": {
        "register_dir": "asset/frank/registration_segments",
        "test_audios": {
            "frank 测试": "asset/frank/frank 测试.m4a",
        },
    },
    "John": {
        "register_dir": "asset/john/registration_segments",
        "test_audios": {
            "安静环境": "asset/john/安静环境测试测试.m4a",
            "嘈杂环境": "asset/john/嘈杂环境测试.m4a",
        },
    },
}


# --------------------------------------------------------------------------- #
#  交叉测试
# --------------------------------------------------------------------------- #


def cross_test(noise_path: str, snr_levels: list[float], threshold: float) -> None:
    """执行 2x2 交叉测试矩阵。"""
    recognizer = WespeakerBest()
    recognizer.config = recognizer.config.__class__(
        **{**vars(recognizer.config), "sim_threshold": threshold}
    )

    # 1. 加载模型
    recognizer._client._ensure_model()

    # 2. 提取噪声 profile
    print(f"提取噪声 profile: {noise_path}")
    noise_profile = WespeakerBest.extract_noise_profile(noise_path)
    print(f"  噪声长度: {len(noise_profile) / 16000:.1f}s\n")

    # 3. 注册所有说话人
    voiceprints: dict[str, torch.Tensor] = {}
    tmp_pk = Path("/tmp/voice_cross.pkl")
    tmp_pk.parent.mkdir(parents=True, exist_ok=True)

    for name, paths in SPEAKERS.items():
        reg_dir = paths["register_dir"]
        print(f"[注册] {name}: {reg_dir}")
        result = recognizer.enroll(reg_dir, noise_profile, str(tmp_pk), snr_levels)
        voiceprints[name] = result["embedding"]
        print(f"  embedding 维度: {result['embedding_dim']}\n")

    # 4. 交叉识别矩阵
    col_headers = ["Frank 声纹", "John 声纹"]
    col_width = 12
    header = f"{'':>14} | " + " | ".join(f"{h:>{col_width}}" for h in col_headers)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  交叉识别矩阵 (阈值 = {:.2f})".format(threshold))
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    all_passed = True
    for test_speaker, speaker_data in SPEAKERS.items():
        for label, audio_path in speaker_data["test_audios"].items():
            row_label = f"{test_speaker}/{label}"
            row = f"{row_label:>14} |"
            for ref_name, ref_emb in voiceprints.items():
                with open(tmp_pk, "wb") as f:
                    pickle.dump(ref_emb.cpu().numpy(), f)

                result = recognizer.recognize(audio_path, str(tmp_pk))
                score = result["confidence"]
                is_match = result["is_recognized"]
                mark = "✅" if is_match else "❌"

                ok = is_match if test_speaker == ref_name else not is_match
                if not ok:
                    all_passed = False

                status = "✅" if ok else "⚠️ "
                row += f" {score:.4f} {mark} {status} |"

            print(row.rstrip(" |"))

    # 5. 总结
    print()
    if all_passed:
        print("✅ 所有测试通过 — 正确匹配且正确拒绝")
    else:
        print("⚠️  存在测试未通过 — 请检查阈值或注册质量")


# --------------------------------------------------------------------------- #
#  CLI 入口
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="声纹交叉测试 — Frank vs John 2x2 矩阵")
    parser.add_argument(
        "--noise",
        default="asset/john/嘈杂环境测试.m4a",
        help="噪声音频文件（用于注册时噪声注入）",
    )
    parser.add_argument(
        "--snrs",
        default="20,15,10,5,0",
        help="SNR 级别，逗号分隔 (default: 20,15,10,5,0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="识别阈值 (default: 0.55)",
    )
    args = parser.parse_args()

    snr_levels = [float(x.strip()) for x in args.snrs.split(",")]

    # 验证文件存在
    for speaker_name, speaker_data in SPEAKERS.items():
        reg_dir = Path(speaker_data["register_dir"])
        if not reg_dir.is_dir():
            print(f"错误: 注册目录不存在: {reg_dir}")
            sys.exit(1)
        for label, audio_path in speaker_data["test_audios"].items():
            if not Path(audio_path).is_file():
                print(f"错误: 测试音频不存在: {audio_path}")
                sys.exit(1)
    if not Path(args.noise).is_file():
        print(f"错误: 噪声音频不存在: {args.noise}")
        sys.exit(1)

    cross_test(args.noise, snr_levels, args.threshold)


if __name__ == "__main__":
    main()
