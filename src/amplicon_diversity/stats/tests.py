"""Group-comparison statistics shared by the functional and diversity modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import skbio
import skbio.stats.composition as composition


def differential_abundance(table: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Two-group differential abundance testing via ANCOM.

    Wraps ``skbio.stats.composition.ancom`` — Analysis of Composition of
    Microbiomes (Mandal et al. 2015), the standard compositionally-aware
    differential abundance test for microbiome data. Unlike a naive per-feature
    test on relative abundances, ANCOM tests all pairwise log-ratios between
    features, which avoids spurious "differences" created by the fact that
    relative abundances are constrained to sum to 1 (closure) — the same
    issue R's ``ALDEx2``/``ANCOM-BC`` are built to address.

    Zeros are handled with multiplicative replacement
    (``skbio.stats.composition.multi_replace``) rather than an arbitrary
    pseudocount, since it preserves the compositional (simplex) geometry
    instead of distorting it with an ad hoc constant.

    Parameters
    ----------
    table : DataFrame
        Samples x features count or relative-abundance table.
    groups : Series
        Index-aligned with ``table``; must have exactly 2 unique values.

    Returns
    -------
    DataFrame indexed by feature, columns:
        W : ANCOM W-statistic — the number of pairwise log-ratio tests (out of
            n_features - 1) in which this feature came out significantly
            different between groups.
        reject_null : bool, whether ANCOM declares the feature differentially
            abundant.
        median_<group1>, median_<group2> : median relative abundance in each
            group, for interpretability/plotting.
    Sorted by W descending. ``result.attrs["group1"]``/``["group2"]`` record
    which group is which.
    """
    groups = groups.reindex(table.index)
    levels = sorted(groups.unique())
    if len(levels) != 2:
        raise ValueError(f"expected exactly 2 groups, got {levels}")
    g1, g2 = levels

    rel = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    positive = pd.DataFrame(
        composition.multi_replace(rel.values), index=table.index, columns=table.columns
    )

    ancom_result, percentiles = composition.ancom(
        positive, groups, sig_test="mannwhitneyu", percentiles=[50]
    )

    result = pd.DataFrame(
        {
            "W": ancom_result["W"],
            "reject_null": ancom_result["Signif"],
            f"median_{g1}": percentiles[(50.0, g1)],
            f"median_{g2}": percentiles[(50.0, g2)],
        },
        index=table.columns,
    )
    result.attrs["group1"] = g1
    result.attrs["group2"] = g2
    return result.sort_values("W", ascending=False)


def permanova(
    distance_matrix: pd.DataFrame,
    groups: pd.Series,
    n_permutations: int = 999,
    seed: int | None = 0,
) -> dict:
    """PERMANOVA (Anderson 2001): test whether group centroids differ in beta-diversity space.

    Wraps ``skbio.stats.distance.permanova`` — the same permutation-based
    pseudo-F test ``vegan::adonis2()`` performs in R.

    Returns
    -------
    dict with keys "test_statistic" (pseudo-F), "p_value", "n_permutations".
    """
    groups = groups.reindex(distance_matrix.index)
    values = np.ascontiguousarray(distance_matrix.values, dtype=float)
    dm = skbio.DistanceMatrix(values, ids=distance_matrix.index.tolist())
    result = skbio.stats.distance.permanova(
        dm, grouping=groups.values, permutations=n_permutations, seed=seed
    )
    return {
        "test_statistic": float(result["test statistic"]),
        "p_value": float(result["p-value"]),
        "n_permutations": int(result["number of permutations"]),
    }
