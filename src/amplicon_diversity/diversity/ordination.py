"""Ordination methods to embed a beta-diversity distance matrix in low dimensions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import MDS


def pcoa(distance_matrix: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """Classical (metric) PCoA via eigendecomposition of the double-centered distance matrix.

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
    d = distance_matrix.values.astype(float)
    n = d.shape[0]
    d2 = d**2
    centering = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * centering @ d2 @ centering

    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = np.clip(eigvals, 0, None)
    total = positive.sum() if positive.sum() > 0 else 1.0
    explained = positive[:n_components] / total

    top_vals = np.clip(eigvals[:n_components], 0, None)
    coords = eigvecs[:, :n_components] * np.sqrt(top_vals)
    columns = [f"PC{i+1}" for i in range(n_components)]
    coords_df = pd.DataFrame(coords, index=distance_matrix.index, columns=columns)
    return coords_df, explained


def nmds(
    distance_matrix: pd.DataFrame,
    n_components: int = 2,
    random_state: int | None = 0,
    n_init: int = 8,
    max_iter: int = 300,
) -> tuple[pd.DataFrame, float]:
    """Non-metric multidimensional scaling (NMDS) of a distance matrix.

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
