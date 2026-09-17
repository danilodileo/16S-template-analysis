"""Alpha diversity: within-sample community diversity metrics.

Thin wrappers around scikit-bio's ``skbio.diversity.alpha`` implementations —
the same statistics ``vegan::diversity()`` / ``phyloseq::estimate_richness()``
compute in R. scikit-bio's defaults already match the conventions used
throughout this package (natural-log Shannon, ``1 - sum(p_i^2)`` Simpson,
bias-corrected Chao1), so these wrappers exist to give a stable, table-aware
API rather than to change the math. The one deliberate deviation: scikit-bio
returns ``NaN`` for an empty (all-zero) sample, which this package treats as
0 diversity instead — useful when a real table has samples that dropped to
zero reads after QC filtering, since a ``NaN`` propagating into a groupby or
boxplot is more disruptive than a defined zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import skbio.diversity.alpha as skbio_alpha


def _as_array(counts: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(counts, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D array of counts for a single sample")
    return arr


def _zero_safe(metric):
    def wrapped(counts: np.ndarray | pd.Series) -> float:
        arr = _as_array(counts)
        if arr.sum() == 0:
            return 0.0
        return float(metric(arr))

    return wrapped


shannon = _zero_safe(skbio_alpha.shannon)
"""Shannon entropy index H' = -sum(p_i * log(p_i)), natural log."""

simpson = _zero_safe(skbio_alpha.simpson)
"""Simpson's diversity index 1 - sum(p_i^2) (probability two random draws differ)."""

pielou_evenness = _zero_safe(skbio_alpha.pielou_e)
"""Pielou's evenness J = H' / log(S), the observed richness."""


def observed_features(counts: np.ndarray | pd.Series) -> int:
    """Number of features with a nonzero count (observed richness)."""
    return int(skbio_alpha.observed_features(_as_array(counts)))


def chao1(counts: np.ndarray | pd.Series) -> float:
    """Bias-corrected Chao1 richness estimator, correcting observed richness
    for unseen rare taxa: S_obs + f1(f1-1) / (2*(f2+1)), where f1/f2 are the
    number of singleton/doubleton features.
    """
    return _zero_safe(lambda arr: skbio_alpha.chao1(arr, bias_corrected=True))(counts)


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
