"""注册说话人 vs batch_test 交叉对比脚本 — WespeakerDeep 默认参数

注册说话人: john / michael / qingqing / xixi / zhong / frank
测试数据: asset/batch_test/ 全部 WAV (b1~b6)

输出 6×6 热力图 + 完整数据
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"

# ── 路径 ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_DIR / "asset"
BATCH_DIR = ASSET_DIR / "batch_test"
OUTPUT_DIR = PROJECT_DIR / "outputs"
sys.path.insert(0, str(PROJECT_DIR / "src"))

# ── 导入 ──────────────────────────────────────────────────────────────
from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep


def main():
    # ── 注册说话人配置 ────────────────────────────────────────────────
    REG_SPEAKERS = [
        ("john", ASSET_DIR / "john" / "registration_segments"),
        ("michael", ASSET_DIR / "michael" / "registration_segments"),
        ("qingqing", ASSET_DIR / "qingqing" / "registration_segments"),
        ("xixi", ASSET_DIR / "xixi" / "registration_segments"),
        ("zhong", ASSET_DIR / "zhong" / "registration_segments"),
        ("frank", ASSET_DIR / "frank" / "registration_segments"),
    ]

    # 校验目录
    for name, d in REG_SPEAKERS:
        if not d.is_dir():
            print(f"❌ 注册目录不存在: {d}")
            sys.exit(1)

    if not BATCH_DIR.is_dir():
        print(f"❌ 测试目录不存在: {BATCH_DIR}")
        sys.exit(1)

    wavs = sorted(BATCH_DIR.glob("*.wav"))
    if not wavs:
        print(f"❌ 未找到测试 WAV: {BATCH_DIR}")
        sys.exit(1)

    print(f"📁 注册说话人: {len(REG_SPEAKERS)} 个")
    for name, d in REG_SPEAKERS:
        n = len(list(d.glob("*.wav")))
        print(f"   {name}: {n} files")
    print(f"\n📁 batch_test 测试数据: {len(wavs)} 个 WAV\n")

    # ── Step 1: 注册 ──────────────────────────────────────────────────
    deep = WespeakerDeep()
    voiceprints: dict[str, Path] = {}

    print("🔐 注册说话人 ...")
    for name, reg_dir in REG_SPEAKERS:
        pk_path = Path(f"/tmp/vp_{name}.pkl")
        print(f"  {name}: ", end="", flush=True)
        result = deep.enroll(str(reg_dir), pk_path=str(pk_path))
        if not result["ok"]:
            raise RuntimeError(f"注册 {name} 失败: {result}")
        voiceprints[name] = pk_path
        print(f"{result['num_templates']} templates ✅")

    reg_names = [name for name, _ in REG_SPEAKERS]

    # ── Step 2: 识别所有测试文件 ──────────────────────────────────────
    print(f"\n🔄 识别 {len(wavs)} 个测试文件 ...")

    # scores[test_file_idx][reg_name] = confidence
    file_scores: list[dict] = []
    # 按 batch 分组统计
    batch_pattern = re.compile(r"^(b\d+)_")
    batch_groups: dict[str, dict[str, list[float]]] = {}

    for idx, wav in enumerate(wavs):
        # 判断属于哪个 batch 分组
        m = batch_pattern.match(wav.name)
        batch_id = m.group(1) if m else "unknown"

        row = {"file": wav.name, "batch": batch_id, "scores": {}}
        for rn in reg_names:
            result = deep.recognize(str(wav), str(voiceprints[rn]))
            conf = result["confidence"]
            row["scores"][rn] = round(conf, 4)

        file_scores.append(row)

        # 写入分组统计
        bg = batch_groups.setdefault(batch_id, {rn: [] for rn in reg_names})
        for rn in reg_names:
            bg[rn].append(row["scores"][rn])

        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(wavs)}")

    print(f"  {len(wavs)}/{len(wavs)} ✅\n")

    batch_ids = sorted(batch_groups.keys(), key=lambda x: int(x[1:]))

    # ── Step 3: 打印汇总表 ────────────────────────────────────────────
    threshold = deep.deep_config.sim_threshold
    print(f"⚙️  阈值: sim_threshold = {threshold}")
    header = f"{'batch↓/注册→':>12s}" + "".join(f"{rn:>10s}" for rn in reg_names)
    print(f"\n{header}")
    for bid in batch_ids:
        bg = batch_groups[bid]
        means = [np.mean(bg[rn]) for rn in reg_names]
        # 本 batch 最大值的注册人
        best = reg_names[int(np.argmax(means))]
        row = f"{bid:>12s}" + "".join(f"{m:>10.4f}" for m in means)
        row += f"  🏆 {best}"
        print(row)

    # ── Step 4: 指标计算 ──────────────────────────────────────────────
    # 对于每个 batch 分组，判断其"应该"命中的注册人
    # 找出每个 batch 命中率最高的注册人
    batch_best: dict[str, str] = {}
    for bid in batch_ids:
        bg = batch_groups[bid]
        means = [np.mean(bg[rn]) for rn in reg_names]
        max_mean = max(means)
        # 只选超过阈值的
        if max_mean >= threshold:
            batch_best[bid] = reg_names[int(np.argmax(means))]
        else:
            batch_best[bid] = "none"

    print(f"\n  Best Match (mean, min~max):")
    for bid in batch_ids:
        best = batch_best[bid]
        bg = batch_groups[bid]
        if best != "none":
            vals = bg[best]
            mean_val = np.mean(vals)
            min_val = np.min(vals)
            max_val = np.max(vals)
            print(f"  {bid} -> {best:>8s}  mean={mean_val:.4f}  (min={min_val:.4f} ~ max={max_val:.4f})")
        else:
            print(f"  {bid} -> {'none':>8s}  (all below threshold)")

    # ── Step 5: 保存数据 ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 构建 6×6 矩阵
    mean_matrix = np.zeros((len(batch_ids), len(reg_names)))
    std_matrix = np.zeros((len(batch_ids), len(reg_names)))
    min_matrix = np.zeros((len(batch_ids), len(reg_names)))
    max_matrix = np.zeros((len(batch_ids), len(reg_names)))
    for i, bid in enumerate(batch_ids):
        for j, rn in enumerate(reg_names):
            vals = batch_groups[bid][rn]
            mean_matrix[i, j] = np.mean(vals) if vals else 0
            std_matrix[i, j] = np.std(vals) if vals else 0
            min_matrix[i, j] = np.min(vals) if vals else 0
            max_matrix[i, j] = np.max(vals) if vals else 0

    data = {
        "threshold": threshold,
        "registered_speakers": reg_names,
        "batch_groups": batch_ids,
        "file_count": len(wavs),
        "mean_matrix": {bid: {rn: round(mean_matrix[i, j], 4) for j, rn in enumerate(reg_names)} for i, bid in enumerate(batch_ids)},
        "min_matrix": {bid: {rn: round(min_matrix[i, j], 4) for j, rn in enumerate(reg_names)} for i, bid in enumerate(batch_ids)},
        "max_matrix": {bid: {rn: round(max_matrix[i, j], 4) for j, rn in enumerate(reg_names)} for i, bid in enumerate(batch_ids)},
        "batch_best_match": {bid: batch_best[bid] for bid in batch_ids},
        "file_details": file_scores,
    }
    json_path = OUTPUT_DIR / "batch_cross_results.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 数据保存: {json_path}")

    # ── Step 6: 热力图 ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.colormaps["RdYlGn"]

    im = ax.imshow(mean_matrix, vmin=0, vmax=1, cmap=cmap, aspect="equal")

    # 标注格子
    for i in range(len(batch_ids)):
        for j in range(len(reg_names)):
            val = mean_matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    # 标注最佳匹配行
    for i, bid in enumerate(batch_ids):
        best_rn = batch_best[bid]
        if best_rn != "none":
            j = reg_names.index(best_rn)
            ax.text(j, i, f"★{mean_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="blue")

    ax.set_xticks(range(len(reg_names)))
    ax.set_yticks(range(len(batch_ids)))
    ax.set_xticklabels(reg_names, fontsize=11)
    ax.set_yticklabels(batch_ids, fontsize=11)
    ax.set_xlabel("Registered Speaker", fontsize=13)
    ax.set_ylabel("batch_test Group", fontsize=13)
    ax.set_title(
        f"Registered Speakers vs batch_test (threshold={threshold})\n"
        f"★ = best match per group",
        fontsize=14,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Confidence", fontsize=11)

    fig.tight_layout()
    png_path = OUTPUT_DIR / "batch_cross_heatmap.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  热力图: {png_path}")

    # ── 清理 ──────────────────────────────────────────────────────────
    for pk in voiceprints.values():
        pk.unlink(missing_ok=True)
    print("🧹 临时 voiceprint 已清理")
    print("✅ 完成")


if __name__ == "__main__":
    main()
