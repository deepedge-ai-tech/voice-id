#!/usr/bin/env python3
"""声纹交叉测试 — 纯 wespeaker 官方包 (standalone)。

测试场景:
  注册: John, John_USB, John_MeetingRoom, John_D_USB, John_D_USB_AEC,
        Xixi, Frank, Qingqing, Zhong, Zhong_D_USB, Angle
  测试: 每人 test_segments 目录

用法:
    uv run python scripts/cross_test.py
    uv run python scripts/cross_test.py --threshold 0.50
    uv run python scripts/cross_test.py --output-dir outputs
    uv run python scripts/cross_test.py --verbose
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wespeaker_deep_edge  # 将 vendored _wespeaker/ 加入 sys.path
import wespeaker

from wespeaker_deep_edge.asnorm import CohortCache

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------- #
#  测试配置
# --------------------------------------------------------------------------- #

ASSET_COMBINE = Path("asset_combine")

SPEAKERS = {
    "John": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john/test_segments",
    },
    "John_USB": {
        "register": str(ASSET_COMBINE / "John.wav"),  # asset_combine has no variant files → use base
        "test_dir": "asset/john_usb/test_segments",
    },
    "John_MeetingRoom": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john_metting_room/test_segments",
    },
    "John_D_USB": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john_d_usb/test_segments",
    },
    "John_D_USB_AEC": {
        "register": str(ASSET_COMBINE / "John.wav"),
        "test_dir": "asset/john_d_usb_AEC/test_segments",
    },
    "Michael": {
        "register": str(ASSET_COMBINE / "Michael.wav"),
        "test_dir": "asset/michael/registration_segments",
    },
    "Xixi": {
        "register": str(ASSET_COMBINE / "Xixi.wav"),
        "test_dir": "asset/xixi/test_segments",
    },
    "Frank": {
        "register": str(ASSET_COMBINE / "Frank.wav"),
        "test_dir": "asset/frank/test_segments",
    },
    "Qingqing": {
        "register": str(ASSET_COMBINE / "Qingqing.wav"),
        "test_dir": "asset/qingqing/test_segments",
    },
    "Zhong": {
        "register": str(ASSET_COMBINE / "Zhong.wav"),
        "test_dir": "asset/zhong/test_segments",
    },
    "Zhong_D_USB": {
        "register": str(ASSET_COMBINE / "Zhong.wav"),
        "test_dir": "asset/zhong_d_usb/test_segments",
    },
    "Angle": {
        "register": str(ASSET_COMBINE / "angle.wav"),
        "test_dir": "asset/angle/test_segments",
    },
}

# Order for display: John group → Zhong group → others
SPEAKER_ORDER = [
    "John",
    "John_USB",
    "John_MeetingRoom",
    "John_D_USB",
    "John_D_USB_AEC",
    "Michael",
    "Zhong",
    "Zhong_D_USB",
    "Xixi",
    "Frank",
    "Qingqing",
    "Angle",
]

# Same-person groups
SAME_PERSON_GROUPS: dict[str, set[str]] = {
    "John": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_USB": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_MeetingRoom": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_D_USB": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "John_D_USB_AEC": {"John", "John_USB", "John_MeetingRoom", "John_D_USB", "John_D_USB_AEC"},
    "Michael": {"Michael"},
    "Zhong": {"Zhong", "Zhong_D_USB"},
    "Zhong_D_USB": {"Zhong", "Zhong_D_USB"},
    "Xixi": {"Xixi"},
    "Frank": {"Frank"},
    "Qingqing": {"Qingqing"},
    "Angle": {"Angle"},
}


def is_same_person(speaker1: str, speaker2: str) -> bool:
    if speaker1 == speaker2:
        return True
    group = SAME_PERSON_GROUPS.get(speaker1, {speaker1})
    return speaker2 in group


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12))


# --------------------------------------------------------------------------- #
#  可视化
# --------------------------------------------------------------------------- #


def plot_heatmap(
    scores: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    threshold: float,
    output_path: Path | None = None,
) -> None:
    fig_height = max(16, len(row_labels) * 0.35)
    fig, ax = plt.subplots(figsize=(18, fig_height))

    vmax = max(scores.max(), 1.0)
    vmin = min(scores.min(), -0.2)
    im = ax.imshow(scores, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticklabels(row_labels, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text_color = "white" if scores[i, j] < threshold else "black"
            ax.text(
                j, i, f"{scores[i, j]:.3f}",
                ha="center", va="center", color=text_color, fontsize=9,
            )

    ax.set_title(
        f"声纹交叉识别矩阵 (阈值 = {threshold:.2f})\n"
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=16, pad=20,
    )
    ax.set_xlabel("注册声纹", fontsize=14)
    ax.set_ylabel("测试音频", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="相似度得分")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n热力图已保存: {output_path}")
    plt.close()


def plot_summary_bar(
    diagonal_scores: dict[str, list[float]],
    threshold: float,
    output_path: Path | None = None,
) -> None:
    speakers = list(diagonal_scores.keys())
    avg_scores = [np.mean(scores) if scores else 0 for scores in diagonal_scores.values()]
    min_scores = [np.min(scores) if scores else 0 for scores in diagonal_scores.values()]

    x = np.arange(len(speakers))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, avg_scores, width=0.6, label="平均得分", color="#4CAF50")

    for i, (avg, mn) in enumerate(zip(avg_scores, min_scores)):
        ax.errorbar(i, avg, yerr=avg - mn, fmt="none", ecolor="black", capsize=5)

    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=2, label=f"阈值 ({threshold})")
    ax.set_xlabel("说话人", fontsize=12)
    ax.set_ylabel("相似度得分", fontsize=12)
    ax.set_title("各说话人自识别得分统计", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(speakers)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    for bar, score in zip(bars, avg_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02, f"{score:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"柱状图已保存: {output_path}")
    plt.close()


# --------------------------------------------------------------------------- #
#  注册 & 识别
# --------------------------------------------------------------------------- #


def enroll(
    model,
    enroll_files: dict[str, Path],
    active_people: list[str],
) -> dict[str, np.ndarray]:
    """Extract enrollment embeddings, deduplicated by file path.

    Returns:
        Dict mapping speaker name to embedding vector.
    """
    enroll_embs: dict[str, np.ndarray] = {}
    cache: dict[str, np.ndarray] = {}

    for person in active_people:
        path = str(enroll_files[person])
        if path not in cache:
            cache[path] = model.extract_embedding(path)
        emb = cache[path]
        if emb is None:
            print(f"  WARNING: {person} 注册 VAD 过滤全部音频，跳过")
            continue
        enroll_embs[person] = emb
        print(f"  注册 {person}: 完成")

    return enroll_embs


def recognize(
    model,
    test_files_map: dict[str, list[Path]],
    enroll_embs: dict[str, np.ndarray],
    active_people: list[str],
    cohort: CohortCache | None = None,
    norm_type: str = "asnorm",
    top_k: int = 300,
) -> dict[str, list[tuple[str, list[float]]]]:
    """Extract test embeddings and compute per-file similarity against all enrollments.

    Args:
        model: Loaded wespeaker model.
        test_files_map: Dict mapping test speaker to list of WAV paths.
        enroll_embs: Dict mapping enrollment speaker name to embedding.
        active_people: Ordered list of active speakers.
        cohort: Optional CohortCache for AS-Norm normalization.

    Returns:
        Dict mapping test speaker to list of (filename, [scores]) tuples,
        where scores[j] = similarity against enroll_embs[active_people[j]].
    """
    # 1. Extract test embeddings
    test_embs: dict[str, list[np.ndarray]] = {}
    for person in active_people:
        embs = []
        for f in test_files_map[person]:
            e = model.extract_embedding(str(f))
            if e is not None:
                embs.append(e)
        if not embs:
            print(f"  WARNING: {person} 所有测试文件被 VAD 过滤")
        test_embs[person] = embs
        print(f"  测试 {person}: {len(embs)}/{len(test_files_map[person])} 有效")

    # 2. Filter to people with valid test embeddings
    valid = [p for p in active_people if p in test_embs and test_embs[p]]
    if len(valid) < len(active_people):
        dropped = set(active_people) - set(valid)
        print(f"  以下说话人无有效测试 embedding，跳过: {dropped}")

    # 3. Compute per-file similarity
    per_file_results: dict[str, list[tuple[str, list[float]]]] = {}
    for tp in valid:
        entries = []
        for fname, temb in zip(test_files_map[tp], test_embs[tp]):
            raw_scores = [cosine_similarity(temb, enroll_embs[ep]) for ep in valid]

            if cohort is not None and cohort._enroll_mu is not None:
                temb_np = (
                    temb.cpu().numpy().astype(np.float32)
                    if hasattr(temb, "cpu")
                    else np.asarray(temb, dtype=np.float32)
                )
                norm_scores, _, _ = cohort.apply(temb_np, top_k=top_k, norm_type=norm_type)
                entries.append((fname.name, norm_scores.tolist()))
            else:
                entries.append((fname.name, raw_scores))

        per_file_results[tp] = entries

    return per_file_results


# --------------------------------------------------------------------------- #
#  主流程
# --------------------------------------------------------------------------- #


def cross_test(
    threshold: float,
    output_dir: Path | None = None,
    verbose: bool = False,
    asnorm: bool = False,
    norm_type: str = "asnorm",
    top_k: int = 300,
) -> None:
    print("=" * 60)
    print("声纹交叉测试 (wespeaker vblinkf + VAD)")
    if asnorm:
        if threshold == 0.55:
            threshold = 6.0
        print(f"  AS-Norm 归一化已启用 ({norm_type.upper()}, top_k={top_k}, 阈值={threshold})")
    print("=" * 60)

    # 1. Load model
    print("\n[1/4] 加载模型 vblinkf (VoxBlink2 SAM-ResNet34)...")
    model = wespeaker.load_model("vblinkf")
    model.set_vad(True)
    print("  模型加载完成，VAD 已启用")

    # 2. Collect files
    print("\n[2/4] 收集注册和测试文件...")
    active_people: list[str] = []
    enroll_files: dict[str, Path] = {}
    test_files_map: dict[str, list[Path]] = {}

    for person in SPEAKER_ORDER:
        reg = Path(SPEAKERS[person]["register"])
        if not reg.exists():
            print(f"  WARNING: 注册文件不存在 {reg}，跳过 {person}")
            continue
        test_dir = Path(SPEAKERS[person]["test_dir"])
        tests = sorted(test_dir.glob("*.wav"))
        if not tests:
            print(f"  WARNING: 无测试文件 {test_dir}，跳过 {person}")
            continue
        enroll_files[person] = reg
        test_files_map[person] = tests
        active_people.append(person)
        print(f"  {person}: 注册={reg.name}, 测试文件={len(tests)}")

    print(f"\n  活跃说话人 ({len(active_people)}): {active_people}")

    # 3. Enroll
    print("\n[3/4] 注册...")
    enroll_embs = enroll(model, enroll_files, active_people)
    active_people = [p for p in active_people if p in enroll_embs]

    # 3b. AS-Norm cohort setup (optional)
    cohort: CohortCache | None = None
    if asnorm:
        cohort_path = "asset/cohort/cohort_embeddings.npy"
        if Path(cohort_path).is_file():
            cohort = CohortCache.load(cohort_path)
            enroll_matrix = np.stack([
                e.cpu().numpy().astype(np.float32) if hasattr(e, "cpu")
                else np.asarray(e, dtype=np.float32)
                for e in (enroll_embs[p] for p in active_people)
            ])
            cohort.precompute_enroll_stats(enroll_matrix, enroll_names=active_people, top_k=top_k)
            print(f"  AS-Norm cohort 已加载: {cohort_path} ({cohort.size} speakers)")
        else:
            print(f"  WARNING: cohort 文件不存在: {cohort_path}，AS-Norm 跳过")

    # 4. Recognize (extract test embeddings + compute similarity)
    print(f"\n[4/4] 识别 ({len(active_people)} 人)...")
    per_file_results = recognize(model, test_files_map, enroll_embs, active_people, cohort=cohort, norm_type=norm_type, top_k=top_k)
    active_people = list(per_file_results.keys())

    N = len(active_people)
    col_labels = active_people

    # Aggregate matrix
    sim_matrix = np.zeros((N, N), dtype=np.float32)
    for i, tp in enumerate(active_people):
        for j, ep in enumerate(active_people):
            vals = [e[1][j] for e in per_file_results[tp]]
            sim_matrix[i][j] = np.mean(vals)

    # 5. Print results
    col_labels = active_people
    header = f"{'测试说话人':14s} {'测试文件':30s}" + "".join(f"{c:>10s}" for c in col_labels)
    print("\n" + "=" * len(header))
    print("每文件详细结果")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for tp in active_people:
        for fname, scores in per_file_results[tp]:
            scores_str = "".join(f"{s:>10.3f}" for s in scores)
            print(f"{tp:14s} {fname:30s} {scores_str}")
        print()

    # 6. Stats by same-person groups
    print("=" * 60)
    print("分组统计")
    print("=" * 60)

    same_scores_all = []
    diff_scores_all = []
    for i, tp in enumerate(active_people):
        for j, ep in enumerate(active_people):
            vals = [e[1][j] for e in per_file_results[tp]]
            if is_same_person(tp, ep):
                same_scores_all.extend(vals)
            else:
                diff_scores_all.extend(vals)

    same_mean = np.mean(same_scores_all) * 100
    diff_mean = np.mean(diff_scores_all) * 100
    gap = same_mean - diff_mean

    print(f"\n  同人平均:     {same_mean:.1f}%")
    print(f"  异人平均:     {diff_mean:.1f}%")
    print(f"  差距:         {gap:.1f}%")

    # Per-person stats
    print(f"\n  每人统计:")
    for i, p in enumerate(active_people):
        self_scores = [e[1][i] for e in per_file_results[p]]
        # Find reference indices that are same person (including self)
        same_indices = [j for j, ep in enumerate(active_people) if is_same_person(p, ep) and j != i]
        other_indices = [j for j, ep in enumerate(active_people) if not is_same_person(p, ep)]

        same_other = []
        for e in per_file_results[p]:
            for si in same_indices:
                same_other.append(e[1][si])
        diff = []
        for e in per_file_results[p]:
            for oi in other_indices:
                diff.append(e[1][oi])

        self_mean = np.mean(self_scores) * 100 if self_scores else 0
        same_other_mean = np.mean(same_other) * 100 if same_other else 0
        diff_mean_p = np.mean(diff) * 100 if diff else 0
        n = len(per_file_results[p])
        print(f"    {p:16s}: 自身={self_mean:5.1f}%  同组其他={same_other_mean:5.1f}%  异人={diff_mean_p:5.1f}%  n={n}")

    # 7. Error analysis
    errors = {"false_accepts": [], "false_rejects": []}
    for tp in active_people:
        for fname, scores in per_file_results[tp]:
            for j, ep in enumerate(active_people):
                score = scores[j]
                if is_same_person(tp, ep):
                    if score < threshold:
                        errors["false_rejects"].append({
                            "test": tp, "file": fname, "ref": ep,
                            "score": score, "gap": threshold - score,
                        })
                else:
                    if score >= threshold:
                        errors["false_accepts"].append({
                            "test": tp, "file": fname, "mistaken_as": ep,
                            "score": score, "gap": score - threshold,
                        })

    total = sum(len(per_file_results[tp]) * N for tp in active_people)
    n_fa = len(errors["false_accepts"])
    n_fr = len(errors["false_rejects"])
    print(f"\n  总比对: {total}, 误接受: {n_fa} ({n_fa/total*100:.1f}%), "
          f"误拒绝: {n_fr} ({n_fr/total*100:.1f}%)")

    if verbose and errors["false_accepts"]:
        print(f"\n  误接受详情:")
        for e in errors["false_accepts"][:10]:
            print(f"    {e['test']}/{e['file']} → {e['mistaken_as']}: 得分={e['score']:.3f} 超出阈值={e['gap']:.3f}")

    if verbose and errors["false_rejects"]:
        print(f"\n  误拒绝详情:")
        for e in errors["false_rejects"][:10]:
            print(f"    {e['test']}/{e['file']} → {e['ref']}: 得分={e['score']:.3f} 低于阈值={e['gap']:.3f}")

    # 8. Optimal threshold search (EER) for AS-Norm
    if asnorm:
        best_eer = 100.0
        best_th = threshold
        for th in np.arange(0.5, 20.0, 0.25):
            fa = sum(1 for s in diff_scores_all if s >= th)
            fr = sum(1 for s in same_scores_all if s < th)
            total_diff = len(diff_scores_all) or 1
            total_same = len(same_scores_all) or 1
            fa_rate = fa / total_diff * 100
            fr_rate = fr / total_same * 100
            eer = (fa_rate + fr_rate) / 2
            if eer < best_eer:
                best_eer = eer
                best_th = th
        print(f"\n  最优阈值: {best_th:.2f} (EER={best_eer:.1f}%, FA={sum(1 for s in diff_scores_all if s >= best_th)/max(len(diff_scores_all),1)*100:.1f}%, FR={sum(1 for s in same_scores_all if s < best_th)/max(len(same_scores_all),1)*100:.1f}%)")

    # 9. Save charts
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build per-file matrix
        row_labels: list[str] = []
        per_file_arr: list[list[float]] = []
        for tp in active_people:
            for fname, scores in per_file_results[tp]:
                row_labels.append(f"{tp}/{fname}")
                per_file_arr.append(scores)
        scores_array = np.array(per_file_arr, dtype=np.float32)

        # Heatmap
        heatmap_path = output_dir / f"cross_test_heatmap_{timestamp}.png"
        plot_heatmap(scores_array, row_labels, col_labels, threshold, heatmap_path)

        # Aggregate heatmap
        agg_path = output_dir / f"cross_test_aggregate_{timestamp}.png"
        fig, ax = plt.subplots(figsize=(10, 8))
        agg_vmax = max(sim_matrix.max(), 1.0)
        agg_vmin = min(sim_matrix.min(), -0.2)
        sns.heatmap(
            sim_matrix,
            xticklabels=col_labels, yticklabels=col_labels,
            annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=agg_vmin, vmax=agg_vmax,
            cbar_kws={"label": "Score"},
            linewidths=1, linecolor="white", ax=ax,
        )
        ax.set_title(f"WeSpeaker 交叉测试聚合 (vblinkf + VAD, 阈值={threshold})", fontsize=14)
        ax.set_xlabel("注册声纹", fontsize=12)
        ax.set_ylabel("测试说话人 (平均)", fontsize=12)
        plt.tight_layout()
        fig.savefig(agg_path, dpi=150, bbox_inches="tight")
        print(f"聚合热力图已保存: {agg_path}")
        plt.close()

        # Bar chart
        diag_scores = {tp: [e[1][i] for e in per_file_results[tp]]
                       for i, tp in enumerate(active_people)}
        bar_path = output_dir / f"cross_test_summary_{timestamp}.png"
        plot_summary_bar(diag_scores, threshold, bar_path)

    # 9. Summary line
    print("\n" + "-" * 60)
    norm_label = f"asnorm={'on' if asnorm else 'off'}"
    if asnorm:
        norm_label += f"/{norm_type}(k={top_k})"
    print(f"SUMMARY: 同人={same_mean:.1f}%, 异人={diff_mean:.1f}%, "
          f"差距={gap:.1f}%, FA={n_fa}, FR={n_fr}, "
          f"{norm_label}")
    print("-" * 60)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="声纹交叉测试 — 纯 wespeaker 官方包 (vblinkf + VAD)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="识别阈值 (default: 0.55)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="图表输出目录",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细输出模式",
    )
    parser.add_argument(
        "--asnorm", action="store_true",
        help="启用 AS-Norm 分数归一化",
    )
    parser.add_argument(
        "--norm-type", type=str, default="asnorm",
        choices=["asnorm", "snorm", "tnorm"],
        help="归一化类型: asnorm (both), snorm (test-side), tnorm (enroll-side)",
    )
    parser.add_argument(
        "--top-k", type=int, default=300,
        help="Cohort top-k 统计数量 (default: 300)",
    )
    args = parser.parse_args()

    # Verify asset files exist
    for name, cfg in SPEAKERS.items():
        if not Path(cfg["register"]).exists():
            print(f"错误: 注册文件不存在: {cfg['register']}")
            sys.exit(1)
        test_dir = Path(cfg["test_dir"])
        if not test_dir.is_dir() or not list(test_dir.glob("*.wav")):
            print(f"错误: 测试目录无 .wav 文件: {test_dir}")
            sys.exit(1)

    output_path = Path(args.output_dir) if args.output_dir else None
    cross_test(args.threshold, output_path, args.verbose, args.asnorm,
               norm_type=args.norm_type, top_k=args.top_k)


if __name__ == "__main__":
    main()
