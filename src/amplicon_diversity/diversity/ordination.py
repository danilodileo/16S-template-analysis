"""Ordination methods to embed a beta-diversity distance matrix in low dimensions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import skbio
from sklearn.manifold import MDS


def pcoa(distance_matrix: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """Classical (metric) PCoA via eigendecomposition of the double-centered distance matrix.

    Wraps ``skbio.stats.ordination.pcoa`` — the same computation
    ``ape::pcoa()`` / ``vegan::wcmdscale()`` perform in R.

    Parameters
    ----------
    distance_matrix : DataFrame
        Symmetric samples x samples dissimilarity matrix (e.g. from ``beta_diversity``).
    n_components : int
        Number of ordination axes to return.

    Returns
    -------
    coords : DataFrame, shape (n_samples, n_components)
        Columns named "PC1", "PC2", ...
    explained_variance_ratio : ndarray, shape (n_components,)
        Fraction of total (signed) eigenvalue sum explained by each axis.
    """
    dm = skbio.DistanceMatrix(distance_matrix.values, ids=distance_matrix.index.tolist())
    result = skbio.stats.ordination.pcoa(dm, dimensions=n_components)
    coords = result.samples.loc[distance_matrix.index]
    explained = result.proportion_explained.values
    return coords, explained


def nmds(
    distance_matrix: pd.DataFrame,
    n_components: int = 2,
    random_state: int | None = 0,
    n_init: int = 8,
    max_iter: int = 300,
) -> tuple[pd.DataFrame, float]:
    """Non-metric multidimensional scaling (NMDS) of a distance matrix.

    Uses ``sklearn.manifold.MDS`` deliberately, not a scikit-bio equivalent —
    scikit-bio doesn't implement NMDS, and ``scikit-learn``'s MDS (metric=False)
    is the standard Python-side stand-in for ``vegan::metaMDS()``.

    Returns
    -------
    coords : DataFrame, shape (n_samples, n_components)
    stress : float
        Kruskal stress-1 of the final embedding (lower is a better fit).
    """
    model = MDS(
        n_components=n_components,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        normalized_stress=True,
    )
    embedding = model.fit_transform(distance_matrix.values)
    columns = [f"NMDS{i+1}" for i in range(n_components)]
    coords_df = pd.DataFrame(embedding, index=distance_matrix.index, columns=columns)
    return coords_df, float(model.stress_)
