"""Beta diversity: pairwise between-sample dissimilarity metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform


def bray_curtis(x: np.ndarray, y: np.ndarray) -> float:
    """Bray-Curtis dissimilarity: sum(|x_i - y_i|) / sum(x_i + y_i)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = np.sum(x + y)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(x - y)) / denom)


def jaccard(x: np.ndarray, y: np.ndarray) -> float:
    """Jaccard dissimilarity on presence/absence: 1 - |intersection| / |union|."""
    x_present = np.asarray(x) > 0
    y_present = np.asarray(y) > 0
    union = np.sum(x_present | y_present)
    if union == 0:
        return 0.0
    intersection = np.sum(x_present & y_present)
    return float(1.0 - intersection / union)


_METRICS = {"bray_curtis": bray_curtis, "jaccard": jaccard}


def beta_diversity(table: pd.DataFrame, metric: str = "bray_curtis") -> pd.DataFrame:
    """Compute a full samples x samples dissimilarity matrix.

    Parameters
    ----------
    table : DataFrame
        Samples as rows, features as columns, raw (or relative) abundances.
    metric : str
        "bray_curtis" or "jaccard".

    Returns
    -------
    DataFrame, shape (n_samples, n_samples), symmetric, zero diagonal.
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(_METRICS)}")
    func = _METRICS[metric]
    values = table.values
    n = values.shape[0]
    condensed = np.zeros(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            condensed[k] = func(values[i], values[j])
            k += 1
    matrix = squareform(condensed)
    return pd.DataFrame(matrix, index=table.index, columns=table.index)
