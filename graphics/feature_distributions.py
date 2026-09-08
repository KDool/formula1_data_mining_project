"""
Feature distribution plots, split by type, restricted to the 25 features
actually used by the final Random Forest model
(json-parameters/random-forest/random_forest_best_params.json).

Numerical features -> histograms (with KDE, median line, sentinel-value
count where the feature uses -1 as a "no history / cold start" flag).

Categorical/flag features -> count plots (constructor_change_flag,
sprint_flag, is_home_race, is_home_constructor_race).
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "dataset" / "outputs" / "prediction.csv"
RF_PARAMS_PATH = REPO_ROOT / "json-parameters" / "random-forest" / "random_forest_best_params.json"
ALIGNED_PARAMS_PATH = REPO_ROOT / "json-parameters" / "random-forest" / "random_forest_params_aligned.json"
OUT_DIR = pathlib.Path(__file__).resolve().parent

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.0)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

SENTINEL_VALUE = -1

CATEGORICAL_COLS = [
    "constructor_change_flag",
    "sprint_flag",
    "is_home_race",
    "is_home_constructor_race",
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    rf_params = json.load(open(RF_PARAMS_PATH))
    rf_features = rf_params["features_used"]

    numerical_cols = [c for c in rf_features if c not in CATEGORICAL_COLS]
    categorical_cols = [c for c in rf_features if c in CATEGORICAL_COLS]

    print(f"Dataset: {DATA_PATH} -- shape {df.shape}")
    print(f"RF features used: {len(rf_features)} (from {RF_PARAMS_PATH.name})")
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    return df, numerical_cols, categorical_cols


def load_aligned_data():
    """The 12-feature set aligned for the SVM vs Random Forest comparison
    (json-parameters/random-forest/random_forest_params_aligned.json). All
    12 are numerical -- no categorical features in this set."""
    df = pd.read_csv(DATA_PATH)
    aligned_params = json.load(open(ALIGNED_PARAMS_PATH))
    aligned_features = aligned_params["data"]["feature_columns"]
    print(f"Aligned (SVM/RF comparison) features: {len(aligned_features)} (from {ALIGNED_PARAMS_PATH.name})")
    return df, aligned_features


def plot_numerical_distributions(df, numerical_cols, out_path, n_cols=5, title=None):
    n_rows = int(np.ceil(len(numerical_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, numerical_cols):
        values = df[col].dropna()
        sentinel_count = int((values == SENTINEL_VALUE).sum())
        sentinel_pct = sentinel_count / len(values) * 100 if len(values) else 0.0

        sns.histplot(values, kde=True, ax=ax, color="teal", bins=40,
                     edgecolor="black", linewidth=0.3)
        median_value = values.median()
        ax.axvline(median_value, color="crimson", linestyle="--",
                    linewidth=1.3, label=f"median={median_value:.2f}")

        ax.set_title(col, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("count", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(loc="upper right", fontsize=11, frameon=True)

        if sentinel_count > 0:
            ax.text(0.02, 0.95, f"sentinel(-1): {sentinel_pct:.1f}%",
                     transform=ax.transAxes, ha="left", va="top", fontsize=11,
                     bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))

    for ax in axes[len(numerical_cols):]:
        fig.delaxes(ax)

    if title is None:
        title = f"Numerical Feature Distributions ({len(numerical_cols)} features)"
    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_categorical_distributions(df, categorical_cols, out_path):
    n_cols = 2
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, categorical_cols):
        counts = df[col].value_counts().sort_index()
        is_constant = df[col].nunique() == 1

        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                    color="steelblue", edgecolor="black")
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v}\n({v / len(df) * 100:.1f}%)",
                    ha="center", va="bottom", fontsize=8)

        title = col
        if is_constant:
            title += "  [CONSTANT -- no variance]"
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color="crimson" if is_constant else "black")
        ax.set_xlabel("")
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=8, axis="x", rotation=45 if df[col].nunique() > 6 else 0)

    for ax in axes[len(categorical_cols):]:
        fig.delaxes(ax)

    fig.suptitle(f"Categorical Feature Distributions ",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_correlation_heatmap(df, numerical_cols, out_path):
    corr = df[numerical_cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 6}, square=True,
                linewidths=.4, cbar_kws={"shrink": .7}, ax=ax)
    ax.set_title(f"Spearman Correlation Heatmap ({len(numerical_cols)} numerical RF features)",
                 fontsize=15, fontweight="bold", pad=14)
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    df, numerical_cols, categorical_cols = load_data()
    plot_numerical_distributions(
        df, numerical_cols, OUT_DIR / "feature_distributions_rf_numerical.png"
    )
    excluded_constant_cols = ["is_home_race", "is_home_constructor_race"]
    categorical_cols_to_plot = [c for c in categorical_cols if c not in excluded_constant_cols]
    plot_categorical_distributions(
        df, categorical_cols_to_plot, OUT_DIR / "feature_distributions_rf_categorical.png"
    )
    plot_correlation_heatmap(
        df, numerical_cols, OUT_DIR / "feature_correlation_rf_heatmap.png"
    )

    df_aligned, aligned_features = load_aligned_data()
    plot_numerical_distributions(
        df_aligned, aligned_features,
        OUT_DIR / "feature_distributions_aligned12_numerical.png", n_cols=6,
        title="Numerical Distribution"
    )


if __name__ == "__main__":
    main()