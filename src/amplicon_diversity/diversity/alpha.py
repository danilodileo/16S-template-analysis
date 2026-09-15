"""Alpha diversity: within-sample community diversity metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_array(counts: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(counts, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D array of counts for a single sample")
    return arr


def shannon(counts: np.ndarray | pd.Series, base: float = np.e) -> float:
    """Shannon entropy index H' = -sum(p_i * log(p_i))."""
    arr = _as_array(counts)
    total = arr.sum()
    if total == 0:
        return 0.0
    p = arr[arr > 0] / total
    return float(-np.sum(p * np.log(p)) / np.log(base))


def simpson(counts: np.ndarray | pd.Series) -> float:
    """Simpson's diversity index 1 - sum(p_i^2) (probability two random draws differ)."""
    arr = _as_array(counts)
    total = arr.sum()
    if total == 0:
        return 0.0
    p = arr / total
    return float(1.0 - np.sum(p**2))


def pielou_evenness(counts: np.ndarray | pd.Series) -> float:
    """Pielou's evenness J = H' / log(S), the observed richness."""
    arr = _as_array(counts)
    s = observed_features(arr)
    if s <= 1:
        return 0.0
    return float(shannon(arr) / np.log(s))


def observed_features(counts: np.ndarray | pd.Series) -> int:
    """Number of features with a nonzero count (observed richness)."""
    arr = _as_array(counts)
    return int(np.sum(arr > 0))


def chao1(counts: np.ndarray | pd.Series) -> float:
    """Chao1 richness estimator, correcting observed richness for unseen rare taxa.

    Chao1 = S_obs + f1(f1-1) / (2*(f2+1)), where f1/f2 are the number of
    singleton/doubleton features.
    """
    arr = _as_array(counts)
    s_obs = observed_features(arr)
    f1 = np.sum(arr == 1)
    f2 = np.sum(arr == 2)
    if f2 == 0:
        correction = f1 * (f1 - 1) / 2.0
    else:
        correction = (f1**2) / (2.0 * f2)
    return float(s_obs + correction)


_METRICS = {
    "shannon": shannon,
    "simpson": simpson,
    "pielou_evenness": pielou_evenness,
    "observed_features": observed_features,
    "chao1": chao1,
}


def alpha_diversity(table: pd.DataFrame, metric: str = "shannon") -> pd.Series:
    """Compute an alpha diversity metric for every sample in a samples x features table.

    Parameters
    ----------
    table : DataFrame
        Samples as rows, features (OTUs/ASVs/genes) as columns, raw counts.
    metric : str
        One of "shannon", "simpson", "pielou_evenness", "observed_features", "chao1".
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(_METRICS)}")
    func = _METRICS[metric]
    values = table.apply(lambda row: func(row.values), axis=1)
    values.name = metric
    return values
