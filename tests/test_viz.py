import matplotlib.axes

from amplicon_diversity.diversity import alpha_diversity, beta_diversity, pcoa
from amplicon_diversity.ml import cross_validate_classifier, feature_importance
from amplicon_diversity.stats import differential_abundance
from amplicon_diversity.taxonomy import top_taxa
from amplicon_diversity.viz import (
    plot_abundance_heatmap,
    plot_alpha_diversity,
    plot_diff_abundance,
    plot_feature_importance,
    plot_ordination,
    plot_roc_curve,
    plot_taxa_barplot,
)


def test_plot_alpha_diversity_returns_axes(amplicon_data):
    counts, groups = amplicon_data
    shannon_values = alpha_diversity(counts, metric="shannon")
    ax = plot_alpha_diversity(shannon_values, groups)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_ordination_returns_axes(amplicon_data):
    counts, groups = amplicon_data
    dist = beta_diversity(counts)
    coords, explained = pcoa(dist)
    ax = plot_ordination(coords, groups, explained_variance=explained)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_taxa_barplot_returns_axes(amplicon_data):
    counts, groups = amplicon_data
    top = top_taxa(counts, n=5)
    ax = plot_taxa_barplot(top, groups)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_diff_abundance_returns_axes(amplicon_data):
    counts, groups = amplicon_data
    result = differential_abundance(counts, groups)
    ax = plot_diff_abundance(result)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_abundance_heatmap_returns_axes(amplicon_data):
    counts, _ = amplicon_data
    top = top_taxa(counts, n=10)
    ax = plot_abundance_heatmap(top)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_roc_and_feature_importance_return_axes(amplicon_data):
    counts, groups = amplicon_data
    result = cross_validate_classifier(counts, groups, n_splits=3, seed=0)
    y_binary = (groups == sorted(groups.unique())[1]).astype(int)
    ax1 = plot_roc_curve(y_binary, result["probabilities"].iloc[:, 1])
    assert isinstance(ax1, matplotlib.axes.Axes)

    importance = feature_importance(result["fitted_model"], list(counts.columns))
    ax2 = plot_feature_importance(importance)
    assert isinstance(ax2, matplotlib.axes.Axes)
