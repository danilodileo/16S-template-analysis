"""Beta diversity: pairwise between-sample dissimilarity metrics.

The single-pair convenience functions (``bray_curtis``, ``jaccard``) wrap
``scipy.spatial.distance`` — the same primitives scikit-bio itself dispatches
to for these metrics. The table-level ``beta_diversity`` wraps
``skbio.diversity.beta_diversity``, the same computation ``vegan::vegdist()``
performs in R, and is also what makes UniFrac available: unlike Bray-Curtis/
Jaccard, UniFrac weights dissimilarity by how evolutionarily distinct the
differing taxa are, which requires a phylogenetic tree relating the features
(``tree=``) rather than the abundance table alone.
"""
from __future__ import annotations

import pandas as pd
import skbio
from scipy.spatial.distance import braycurtis as _braycurtis
from scipy.spatial.distance import jaccard as _jaccard


def bray_curtis(x, y) -> float:
    """Bray-Curtis dissimilarity: sum(|x_i - y_i|) / sum(x_i + y_i)."""
    return float(_braycurtis(x, y))


def jaccard(x, y) -> float:
    """Jaccard dissimilarity on presence/absence: 1 - |intersection| / |union|."""
    return float(_jaccard(x, y))


_METRIC_NAMES = {
    "bray_curtis": "braycurtis",
    "jaccard": "jaccard",
    "weighted_unifrac": "weighted_unifrac",
    "unweighted_unifrac": "unweighted_unifrac",
}
_UNIFRAC_METRICS = {"weighted_unifrac", "unweighted_unifrac"}


def beta_diversity(
    table: pd.DataFrame,
    metric: str = "bray_curtis",
    tree: skbio.TreeNode | None = None,
) -> pd.DataFrame:
    """Compute a full samples x samples dissimilarity matrix.

    Parameters
    ----------
    table : DataFrame
        Samples as rows, features as columns, raw (or relative) abundances.
    metric : str
        "bray_curtis", "jaccard", "weighted_unifrac", or "unweighted_unifrac".
        The UniFrac metrics require ``tree``.
    tree : skbio.TreeNode, optional
        Phylogenetic tree relating the features (table columns) to tip names;
        required for "weighted_unifrac"/"unweighted_unifrac".

    Returns
    -------
    DataFrame, shape (n_samples, n_samples), symmetric, zero diagonal.
    """
    if metric not in _METRIC_NAMES:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(_METRIC_NAMES)}")

    kwargs = {}
    if metric in _UNIFRAC_METRICS:
        if tree is None:
            raise ValueError(f"metric {metric!r} requires a phylogenetic tree (tree=...)")
        kwargs = {"tree": tree, "taxa": table.columns.tolist()}

    dm = skbio.diversity.beta_diversity(
        _METRIC_NAMES[metric], table.values, ids=table.index.tolist(), **kwargs
    )
    return dm.to_data_frame()
