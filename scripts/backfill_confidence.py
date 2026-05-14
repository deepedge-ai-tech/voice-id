#!/usr/bin/env python3
"""回填历史实验的总置信度和同人平均置信度到 experiment_log.json。

从每个实验的 results.json 中提取 score 数据，计算后写入日志。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare import (
    EXPERIMENTS_DIR,
    EXPERIMENT_LOG_PATH,
    SHORT_AUDIO_DURATION_VAD,
    SHORT_AUDIO_DURATION_NO_VAD,
)


def load_results(experiment_id: str) -> dict | None:
    """加载实验的 results.json."""
    results_path = EXPERIMENTS_DIR / experiment_id / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def compute_confidence(results: dict, enable_vad: bool) -> tuple[float, float, float, float, int]:
    """从 results.json 计算各项置信度指标.

    Returns:
        (total_avg_confidence, genuine_avg_confidence,
         short_audio_confidence, short_audio_genuine_confidence, short_audio_count)
    """
    test_cases = results.get("recognition", {}).get("test_cases", [])
    if not test_cases:
        return 0.0, 0.0, 0.0, 0.0, 0

    import numpy as np

    from prepare import is_same_person

    short_audio_duration = SHORT_AUDIO_DURATION_VAD if enable_vad else SHORT_AUDIO_DURATION_NO_VAD

    all_scores = [tc.get("score", 0.0) for tc in test_cases]
    total_avg = float(np.mean(all_scores)) if all_scores else 0.0

    genuine_scores = []
    short_scores = []
    short_genuine_scores = []
    for tc in test_cases:
        ts = tc.get("test_speaker", "")
        rs = tc.get("ref_speaker", "")
        score = tc.get("score", 0.0)
        if tc.get("vad_duration", 0.0) < short_audio_duration:
            short_scores.append(score)
            if is_same_person(ts, rs):
                short_genuine_scores.append(score)
        if is_same_person(ts, rs):
            genuine_scores.append(score)

    genuine_avg = float(np.mean(genuine_scores)) if genuine_scores else 0.0
    short_avg = float(np.mean(short_scores)) if short_scores else 0.0
    short_genuine_avg = float(np.mean(short_genuine_scores)) if short_genuine_scores else 0.0
    return total_avg, genuine_avg, short_avg, short_genuine_avg, len(short_scores)


def main() -> None:
    if not EXPERIMENT_LOG_PATH.exists():
        print(f"错误: experiment_log.json 不存在: {EXPERIMENT_LOG_PATH}")
        sys.exit(1)

    with open(EXPERIMENT_LOG_PATH, "r", encoding="utf-8") as f:
        log = json.load(f)

    updated = 0
    missing_results = 0

    for entry in log:
        metrics = entry.get("metrics", {})
        config = entry.get("config", {})
        enable_vad = config.get("enable_vad", True)

        exp_id = entry.get("experiment_id", "")
        results = load_results(exp_id)
        if results is None:
            print(f"⚠ 未找到 results.json: {exp_id}")
            missing_results += 1
            continue

        total_avg, genuine_avg, short_avg, short_genuine_avg, short_count = compute_confidence(results, enable_vad)
        metrics["total_avg_confidence"] = round(total_avg, 6)
        metrics["genuine_avg_confidence"] = round(genuine_avg, 6)
        metrics["short_audio_confidence"] = round(short_avg, 6)
        metrics["short_audio_genuine_confidence"] = round(short_genuine_avg, 6)
        metrics["short_audio_count"] = short_count
        updated += 1

        print(f"  {exp_id}: vad={enable_vad}, short_dur={'1.5' if not enable_vad else '0.6'}, short_count={short_count}")
        print(f"    total={total_avg:.4f}, genuine={genuine_avg:.4f}, short={short_avg:.4f}, short_genuine={short_genuine_avg:.4f}")

    if updated > 0:
        with open(EXPERIMENT_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 已更新 {updated} 个实验记录")
    else:
        print(f"\n无需更新（{skipped} 个已有数据，{missing_results} 个缺失 results.json）")


if __name__ == "__main__":
    main()
