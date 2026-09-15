"""Group-comparison statistics shared by the functional and diversity modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)
    out = np.empty(n)
    out[order] = bh
    return out


def differential_abundance(
    table: pd.DataFrame,
    groups: pd.Series,
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Two-group differential abundance testing, feature by feature.

    Runs a Mann-Whitney U test per feature (columns of ``table``, on
    relative abundance) between the two groups in ``groups``, with a
    Benjamini-Hochberg FDR correction and a log2 fold-change of the group
    means (after adding ``pseudocount`` to avoid log(0)).

    Parameters
    ----------
    table : DataFrame
        Samples x features count or relative-abundance table.
    groups : Series
        Index-aligned with ``table``; must have exactly 2 unique values.

    Returns
    -------
    DataFrame indexed by feature, columns: group1, group2, mean_group1,
    mean_group2, log2fc (group1 vs group2), pvalue, qvalue. Sorted by qvalue.
    """
    groups = groups.reindex(table.index)
    levels = sorted(groups.unique())
    if len(levels) != 2:
        raise ValueError(f"expected exactly 2 groups, got {levels}")
    g1, g2 = levels

    rel = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    mask1 = (groups == g1).values
    mask2 = (groups == g2).values

    pvalues = np.ones(table.shape[1])
    mean1 = rel.values[mask1].mean(axis=0)
    mean2 = rel.values[mask2].mean(axis=0)

    for j in range(table.shape[1]):
        x = rel.values[mask1, j]
        y = rel.values[mask2, j]
        combined = np.concatenate([x, y])
        if np.allclose(combined, combined[0]):
            pvalues[j] = 1.0
            continue
        _, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
        pvalues[j] = p

    qvalues = _benjamini_hochberg(pvalues)
    log2fc = np.log2((mean1 + pseudocount) / (mean2 + pseudocount))

    result = pd.DataFrame(
        {
            "mean_group1": mean1,
            "mean_group2": mean2,
            "log2fc": log2fc,
            "pvalue": pvalues,
            "qvalue": qvalues,
        },
        index=table.columns,
    )
    result.attrs["group1"] = g1
    result.attrs["group2"] = g2
    return result.sort_values("qvalue")


def permanova(
    distance_matrix: pd.DataFrame,
    groups: pd.Series,
    n_permutations: int = 999,
    seed: int | None = 0,
) -> dict:
    """PERMANOVA (Anderson 2001): test whether group centroids differ in beta-diversity space.

    Computes the pseudo-F statistic from a distance matrix and assesses
    significance by permuting group labels.

    Returns
    -------
    dict with keys "test_statistic" (pseudo-F), "p_value", "n_permutations".
    """
    groups = groups.reindex(distance_matrix.index)
    d2 = distance_matrix.values.astype(float) ** 2
    n = d2.shape[0]

    def pseudo_f(labels: np.ndarray) -> float:
        ss_total = d2.sum() / (2 * n)
        ss_within = 0.0
        for level in np.unique(labels):
            idx = np.where(labels == level)[0]
            n_g = len(idx)
            if n_g <= 1:
                continue
            sub = d2[np.ix_(idx, idx)]
            ss_within += sub.sum() / (2 * n_g)
        ss_among = ss_total - ss_within
        n_groups = len(np.unique(labels))
        df_among = n_groups - 1
        df_within = n - n_groups
        if df_among <= 0 or df_within <= 0 or ss_within == 0:
            return 0.0
        return (ss_among / df_among) / (ss_within / df_within)

    labels = groups.values
    observed = pseudo_f(labels)

    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        permuted = rng.permutation(labels)
        perm_stats[i] = pseudo_f(permuted)

    p_value = (np.sum(perm_stats >= observed) + 1) / (n_permutations + 1)
    return {"test_statistic": float(observed), "p_value": float(p_value), "n_permutations": n_permutations}
