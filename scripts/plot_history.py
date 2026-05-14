#!/usr/bin/env python3
"""从 experiment_log.json 绘制实验历史图表到 experiment_history.png."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare import EXPERIMENT_LOG_PATH

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "Noto Sans CJK JP", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "experiment_history.png"


def main() -> None:
    if not EXPERIMENT_LOG_PATH.exists():
        print(f"错误: experiment_log.json 不存在: {EXPERIMENT_LOG_PATH}")
        sys.exit(1)

    with open(EXPERIMENT_LOG_PATH, encoding="utf-8") as f:
        log = json.load(f)

    if not log:
        print("错误: experiment_log.json 为空")
        sys.exit(1)

    best_idx = max(range(len(log)), key=lambda i: log[i]["metrics"].get("short_audio_genuine_confidence", 0))
    best_short_gen = log[best_idx]["metrics"].get("short_audio_genuine_confidence", 0)

    exp_ids = [e["experiment_id"] for e in log]
    short_genuine = [e["metrics"].get("short_audio_genuine_confidence", 0) for e in log]
    short_audio = [e["metrics"].get("short_audio_confidence", 0) for e in log]
    genuine_conf = [e["metrics"].get("genuine_avg_confidence", 0) for e in log]
    total_conf = [e["metrics"].get("total_avg_confidence", 0) for e in log]
    far_vals = [e["metrics"].get("far", 0) for e in log]
    frr_vals = [e["metrics"].get("frr", 0) for e in log]
    eer_vals = [e["metrics"].get("eer", 0) for e in log]
    acc_vals = [e["metrics"].get("overall_accuracy", 0) for e in log]

    x = np.arange(len(exp_ids))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    # 上子图：置信度指标
    ax1.plot(x, total_conf, "o-", label="总平均置信度", alpha=0.5, markersize=3)
    ax1.plot(x, genuine_conf, "s-", label="同人平均置信度", linewidth=2, markersize=4)
    ax1.plot(x, short_audio, "^--", label="短音频置信度(全部)", alpha=0.4, markersize=3)
    ax1.plot(x, short_genuine, "D-", label="短音频同人置信度", linewidth=2, markersize=4, color="purple")
    ax1.axhline(y=0.7, color="green", linestyle=":", alpha=0.5, linewidth=2, label="目标: 同人置信度>0.7")
    ax1.axhline(y=0.55, color="purple", linestyle=":", alpha=0.5, linewidth=2, label="目标: 短音频同人>0.55")
    ax1.axvline(x=best_idx, color="purple", linestyle="--", alpha=0.3)
    ax1.annotate(
        f"最佳: {best_short_gen:.4f}",
        xy=(best_idx, best_short_gen),
        xytext=(best_idx - 5, best_short_gen + 0.05),
        arrowprops=dict(arrowstyle="->", color="purple", alpha=0.5),
        fontsize=10,
        color="purple",
    )
    ax1.set_ylabel("置信度", fontsize=12)
    ax1.set_title("Voice-ID 实验历史 — 置信度指标", fontsize=14)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(alpha=0.3)

    # 下子图：错误率
    ax2.plot(x, far_vals, "o-", label="FAR (误接受率)", alpha=0.7, markersize=3)
    ax2.plot(x, frr_vals, "s-", label="FRR (误拒绝率)", alpha=0.7, markersize=3)
    ax2.plot(x, eer_vals, "^--", label="EER", alpha=0.6, markersize=3)
    ax2.plot(x, acc_vals, "D-", label="总体准确率", alpha=0.7, markersize=3)
    ax2.axhline(y=0.05, color="red", linestyle=":", alpha=0.5, label="FAR<5%")
    ax2.axhline(y=0.10, color="orange", linestyle=":", alpha=0.5, label="FRR<10%")
    ax2.axhline(y=0.08, color="gray", linestyle=":", alpha=0.5, label="EER<8%")
    ax2.set_ylabel("比率", fontsize=12)
    ax2.set_title("Voice-ID 实验历史 — 错误率与准确率", fontsize=14)
    ax2.set_xlabel("实验序号", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
