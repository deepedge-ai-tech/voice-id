"""从 batch_cross_results.json 生成 summary 图（含 min~max 范围）"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"

PROJECT_DIR = Path(__file__).resolve().parent.parent
json_path = PROJECT_DIR / "outputs" / "batch_cross_results.json"
if not json_path.exists():
    print(f"Error: data file not found. Run batch_cross_test.py first.")
    sys.exit(1)

data = json.loads(json_path.read_text(encoding="utf-8"))

reg_names = data["registered_speakers"]
batch_ids = data["batch_groups"]
threshold = data["threshold"]
mean_matrix = data["mean_matrix"]
min_matrix = data.get("min_matrix", None)
max_matrix = data.get("max_matrix", None)

n_batch = len(batch_ids)
n_reg = len(reg_names)

values = np.zeros((n_batch, n_reg))
mins = np.zeros((n_batch, n_reg))
maxs = np.zeros((n_batch, n_reg))
best_indices = []
for i, bid in enumerate(batch_ids):
    row = []
    for j, rn in enumerate(reg_names):
        row.append(mean_matrix[bid][rn])
    values[i] = row
    if min_matrix:
        mins[i] = [min_matrix[bid][rn] for rn in reg_names]
    if max_matrix:
        maxs[i] = [max_matrix[bid][rn] for rn in reg_names]
    best_indices.append(int(np.argmax(row)))

colors = plt.colormaps["Set2"](np.linspace(0, 1, n_reg))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2.2, 1]})

# ── Left: Grouped bar chart with min~max range ─────────────────────────
x = np.arange(n_batch)
bar_width = 0.12

for j in range(n_reg):
    offset = (j - n_reg / 2 + 0.5) * bar_width
    bars = ax1.bar(
        x + offset, values[:, j], bar_width,
        label=reg_names[j], color=colors[j], edgecolor="white", linewidth=0.5,
    )

    # Add min~max range as error bars
    if min_matrix and max_matrix:
        lower_err = values[:, j] - mins[:, j]
        upper_err = maxs[:, j] - values[:, j]
        ax1.errorbar(
            x + offset, values[:, j],
            yerr=[lower_err, upper_err],
            fmt="none", ecolor="gray", capsize=2, capthick=1, elinewidth=1, alpha=0.6,
        )

ax1.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label=f"threshold={threshold}")
ax1.set_xlabel("batch_test Group", fontsize=13)
ax1.set_ylabel("Confidence", fontsize=13)
ax1.set_title("Registered Speaker Match per batch_test Group", fontsize=14, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(batch_ids, fontsize=12)
ax1.set_ylim(0, 1.0)
ax1.legend(loc="upper left", fontsize=9, ncol=2)
ax1.grid(axis="y", alpha=0.3)

# Star marker on best match
for i in range(n_batch):
    j = best_indices[i]
    offset = (j - n_reg / 2 + 0.5) * bar_width
    val = values[i, j]
    color_star = "red" if val >= threshold else "orange"
    ax1.text(x[i] + offset, maxs[i, j] + 0.03, "*", ha="center", va="bottom",
             fontsize=18, color=color_star, fontweight="bold")

# ── Right: Summary text with min~max ──────────────────────────────────
ax2.axis("off")
lines = []
lines.append("== Summary ==")
lines.append("=" * 32)
lines.append("")
lines.append(f"Threshold: {threshold}")
lines.append(f"Total test files: {data['file_count']}")
lines.append("")
lines.append(f"{'Group':<6s} {'Best Match':<10s} {'mean':>6s}  {'min~max':<16s}")
lines.append("-" * 40)

for i, bid in enumerate(batch_ids):
    j = best_indices[i]
    best_name = reg_names[j]
    mean_val = values[i, j]
    min_val = mins[i, j]
    max_val = maxs[i, j]
    flag = " *" if mean_val >= threshold else "  "
    lines.append(f"{bid:<6s} {best_name:<10s} {mean_val:>6.3f}  {min_val:.3f}~{max_val:.3f}{flag}")

lines.append("")
lines.append("-" * 40)
above = sum(1 for i in range(n_batch) if values[i, best_indices[i]] >= threshold)
lines.append(f"Above threshold: {above}/{n_batch}")
lines.append("")
lines.append("Top-2 per group:")
for i, bid in enumerate(batch_ids):
    j = best_indices[i]
    top2 = sorted(
        [(reg_names[k], values[i, k]) for k in range(n_reg) if k != j],
        key=lambda x: -x[1],
    )
    lines.append(f"  {bid}: {top2[0][0]} ({top2[0][1]:.3f})")

ax2.text(
    0, 0.95, "\n".join(lines), fontsize=11, fontfamily="monospace",
    verticalalignment="top", linespacing=1.5,
    bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.8),
)

fig.tight_layout()
png_path = PROJECT_DIR / "outputs" / "batch_cross_summary.png"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Summary chart saved: {png_path}")
