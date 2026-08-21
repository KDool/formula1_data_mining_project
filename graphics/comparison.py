"""
SVM vs Random Forest -- nested CV per-fold comparison chart.
Real per-fold macro-F1 scores, both models aligned on the same 12
features, same dataset, same cold-start handling.
"""

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Real data -- 5 outer folds, nested race-level TimeSeriesSplit CV
# ------------------------------------------------------------------
svm_folds = np.array([0.6189, 0.6763, 0.6797, 0.7272, 0.6736])
rf_folds  = np.array([0.7027, 0.6708, 0.7171, 0.6820, 0.7043])

svm_holdout = 0.6818
rf_holdout  = 0.6665

# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#4A5568",
    "axes.linewidth": 0.8,
})

TEAL = "#0F9D8F"
NAVY = "#12294B"
GRAY = "#8A94A6"
RED  = "#C0392B"

fig, ax = plt.subplots(figsize=(7, 5))

positions = [1, 2]
data = [svm_folds, rf_folds]
labels = ["SVM", "Random Forest"]
colors = [TEAL, NAVY]

# Boxplots, no outlier markers (we plot the real points ourselves)
bp = ax.boxplot(
    data, positions=positions, widths=0.42, patch_artist=True,
    showfliers=False, showmeans=False,
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2, color="#4A5568"),
    capprops=dict(linewidth=1.2, color="#4A5568"),
    medianprops=dict(linewidth=1.6, color="white"),
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor(color)

# Individual fold points, jittered slightly so they don't overlap
rng = np.random.default_rng(0)
for pos, folds, color in zip(positions, data, colors):
    jitter = rng.uniform(-0.10, 0.10, size=len(folds))
    ax.scatter(
        np.full(len(folds), pos) + jitter, folds,
        s=55, color="white", edgecolor=color, linewidth=1.6,
        zorder=5,
    )

# Mean markers (diamond) + labels
for pos, folds, color in zip(positions, data, colors):
    mean_val = folds.mean()
    ax.scatter([pos], [mean_val], marker="D", s=70, color=color,
               edgecolor="white", linewidth=1.2, zorder=6)
    ax.annotate(f"mean = {mean_val:.3f}", xy=(pos, mean_val),
                xytext=(pos + 0.28, mean_val), va="center", fontsize=9.5,
                color=color, fontweight="bold")

# Holdout markers, shown as a separate reference (star), since these are
# single-point true-holdout results, not part of the nested-CV distribution
for pos, holdout_val, color in zip(positions, [svm_holdout, rf_holdout], colors):
    ax.scatter([pos], [holdout_val], marker="*", s=260, color="#F5B700",
               edgecolor=color, linewidth=1.2, zorder=7)

ax.scatter([], [], marker="*", s=200, color="#F5B700", edgecolor="black",
           linewidth=0.8, label="2025 holdout (single true test)")
ax.scatter([], [], marker="D", s=70, color=GRAY, edgecolor="white",
           label="Nested-CV mean (5 folds)")
ax.scatter([], [], marker="o", s=55, color="white", edgecolor=GRAY,
           linewidth=1.6, label="Individual outer fold")

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
ax.set_xlim(0.5, 2.7)
ax.set_ylabel("Macro-F1", fontsize=11)
ax.set_ylim(0.55, 0.78)
ax.set_title("SVM vs Random Forest \u2014 aligned comparison\n"
             "(same 12 features, same dataset, same cold-start handling)",
             fontsize=12, fontweight="bold", pad=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="-", linewidth=0.5, color="#E2E8F0", zorder=0)
ax.set_axisbelow(True)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3,
          frameon=False, fontsize=9)

# Wilcoxon annotation
ax.text(1.5, 0.765, "Wilcoxon signed-rank: p = 0.625 (n=5, not significant)",
        ha="center", fontsize=9.5, color="#4A5568", style="italic")

plt.tight_layout()
plt.savefig("model_comparison_aligned.png", dpi=250, bbox_inches="tight")
plt.show()
print("Saved: model_comparison_aligned.png")