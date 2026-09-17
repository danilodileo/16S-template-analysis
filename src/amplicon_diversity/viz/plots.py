"""Shared plotting functions with a consistent style, used across all example workflows."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = "Set2"


def _get_ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    return ax


def plot_alpha_diversity(values: pd.Series, groups: pd.Series, ax=None, title: str | None = None):
    """Boxplot + jittered points of an alpha-diversity metric, split by group."""
    ax = _get_ax(ax)
    groups = groups.reindex(values.index)
    df = pd.DataFrame({"value": values, "group": groups})
    order = sorted(df["group"].unique())
    sns.boxplot(
        data=df, x="group", y="value", order=order, hue="group", palette=_PALETTE,
        legend=False, ax=ax, showfliers=False,
    )
    sns.stripplot(data=df, x="group", y="value", order=order, color="black", alpha=0.5, size=4, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(values.name or "diversity")
    ax.set_title(title or f"Alpha diversity ({values.name})")
    return ax


def plot_ordination(
    coords: pd.DataFrame,
    groups: pd.Series | None = None,
    explained_variance: np.ndarray | None = None,
    ax=None,
    title: str = "Ordination",
):
    """Scatter plot of a 2-D ordination (PCoA/NMDS), optionally colored by group."""
    ax = _get_ax(ax)
    x_col, y_col = coords.columns[0], coords.columns[1]
    if groups is not None:
        groups = groups.reindex(coords.index)
        df = coords.copy()
        df["group"] = groups
        sns.scatterplot(data=df, x=x_col, y=y_col, hue="group", palette=_PALETTE, s=80, ax=ax)
        ax.legend(title="", frameon=False)
    else:
        ax.scatter(coords[x_col], coords[y_col], s=80, color=sns.color_palette(_PALETTE)[0])

    if explained_variance is not None:
        ax.set_xlabel(f"{x_col} ({explained_variance[0] * 100:.1f}%)")
        ax.set_ylabel(f"{y_col} ({explained_variance[1] * 100:.1f}%)")
    else:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    return ax


def plot_taxa_barplot(
    table: pd.DataFrame, groups: pd.Series | None = None, ax=None, title: str = "Taxonomic composition"
):
    """Stacked relative-abundance barplot, one bar per sample (optionally sorted by group)."""
    ax = _get_ax(ax)
    rel = table.div(table.sum(axis=1), axis=0)
    order = rel.index
    if groups is not None:
        groups = groups.reindex(rel.index)
        order = groups.sort_values().index
    rel = rel.loc[order]

    colors = sns.color_palette("tab20", n_colors=rel.shape[1])
    bottom = np.zeros(rel.shape[0])
    for color, taxon in zip(colors, rel.columns):
        ax.bar(
            range(rel.shape[0]), rel[taxon].values, bottom=bottom, color=color, label=taxon, width=0.9
        )
        bottom += rel[taxon].values

    ax.set_xlim(-0.5, rel.shape[0] - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Relative abundance")
    ax.set_xticks([])
    ax.set_xlabel("Samples" + (" (grouped)" if groups is not None else ""))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small", frameon=False)
    ax.set_title(title)
    return ax


def plot_diff_abundance(
    result: pd.DataFrame,
    ax=None,
    title: str = "Differential abundance (ANCOM)",
):
    """ANCOM plot: W-statistic (how many pairwise log-ratio tests a feature was
    significant in, out of n_features - 1) vs. a log2 fold-change of group
    medians, from a ``stats.differential_abundance`` result. Features ANCOM
    itself declared differentially abundant (``reject_null``) are highlighted
    — no separate significance threshold to choose, unlike a p/q-value volcano.
    """
    ax = _get_ax(ax)
    median_cols = [c for c in result.columns if c.startswith("median_")]
    if len(median_cols) != 2:
        raise ValueError("expected exactly two 'median_<group>' columns in the ANCOM result")
    col1, col2 = median_cols
    eps = 1e-6
    log2fc = np.log2((result[col1] + eps) / (result[col2] + eps))
    significant = result["reject_null"].astype(bool)

    ax.scatter(
        log2fc[~significant], result.loc[~significant, "W"],
        color="lightgrey", s=25, label="not significant",
    )
    ax.scatter(
        log2fc[significant], result.loc[significant, "W"],
        color=sns.color_palette(_PALETTE)[1], s=35, label="significant (ANCOM)",
    )
    for feat in result.index[significant][:15]:
        ax.annotate(feat, (log2fc[feat], result.loc[feat, "W"]), fontsize=7, alpha=0.8)

    ax.set_xlabel(f"log2 fold-change (median {col1[len('median_'):]} / {col2[len('median_'):]})")
    ax.set_ylabel("W statistic")
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def plot_abundance_heatmap(
    table: pd.DataFrame, z_score: bool = True, ax=None, title: str = "Abundance heatmap"
):
    """Heatmap of a samples x features table, optionally z-scored per feature (column)."""
    ax = _get_ax(ax)
    data = table.copy()
    if z_score:
        data = (data - data.mean(axis=0)) / data.std(axis=0).replace(0, 1)
    sns.heatmap(
        data,
        cmap="vlag" if z_score else "viridis",
        center=0 if z_score else None,
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(title)
    return ax


def plot_roc_curve(y_true, y_proba, ax=None, label: str = "model", title: str = "ROC curve"):
    """ROC curve for a binary classifier's predicted probabilities, with the diagonal chance line."""
    ax = _get_ax(ax)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    from sklearn.metrics import auc

    roc_auc = auc(fpr, tpr)
    ax.plot(
        fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})", color=sns.color_palette(_PALETTE)[0], linewidth=2
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    return ax


def plot_feature_importance(
    importance: pd.Series, top_n: int = 15, ax=None, title: str = "Feature importance"
):
    """Horizontal barplot of the top-N most important features from a fitted model."""
    ax = _get_ax(ax)
    top = importance.sort_values(ascending=True).tail(top_n)
    ax.barh(top.index, top.values, color=sns.color_palette(_PALETTE)[2])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    return ax
