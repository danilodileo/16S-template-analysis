"""Synthetic 16S/18S amplicon (ASV-level) community data generators.

These functions produce reproducible, realistically-structured data
(compositional, overdispersed, with a controllable "signal" between groups)
so the rest of the package can be unit-tested and demonstrated without
depending on external downloads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_GENERA = [
    "Bacteroides", "Prevotella", "Faecalibacterium", "Bifidobacterium",
    "Escherichia", "Akkermansia", "Ruminococcus", "Lactobacillus",
    "Blautia", "Roseburia", "Alistipes", "Parabacteroides",
    "Clostridium", "Enterococcus", "Fusobacterium", "Veillonella",
    "Streptococcus", "Klebsiella", "Bilophila", "Desulfovibrio",
]


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def simulate_asv_table(
    n_samples: int = 40,
    n_features: int = 60,
    n_groups: int = 2,
    n_differential: int = 8,
    effect_size: float = 3.0,
    library_size: tuple[int, int] = (5_000, 20_000),
    dispersion: float = 0.3,
    seed: int | None = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate an ASV (feature) x sample count table plus a group label.

    Counts are drawn from a Dirichlet-multinomial model: each group has its
    own mean composition, with ``n_differential`` features perturbed by
    ``effect_size`` (fold change) between groups, on top of Gamma-distributed
    per-feature dispersion (``dispersion``) that gives realistic overdispersed,
    zero-inflated-looking counts — the same statistical shape as a real
    DADA2/Deblur ASV table.

    Returns
    -------
    counts : DataFrame, shape (n_samples, n_features)
        Raw counts, samples as rows, features (ASVs) as columns.
    groups : Series, index-aligned with ``counts``
        Group label per sample ("group_0", "group_1", ...).
    """
    rng = _rng(seed)
    feature_ids = [f"ASV_{i:04d}" for i in range(n_features)]
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]

    base_mean = rng.dirichlet(np.full(n_features, 0.5))

    diff_idx = rng.choice(n_features, size=min(n_differential, n_features), replace=False)
    group_means = []
    for g in range(n_groups):
        mean_g = base_mean.copy()
        sign = 1.0 if g % 2 == 0 else -1.0
        fold = effect_size ** (sign * (1 + (g // 2)))
        mean_g[diff_idx] = mean_g[diff_idx] * fold
        mean_g = mean_g / mean_g.sum()
        group_means.append(mean_g)

    counts = np.zeros((n_samples, n_features), dtype=int)
    group_labels = np.array([f"group_{i % n_groups}" for i in range(n_samples)])
    rng.shuffle(group_labels)

    for i, g_label in enumerate(group_labels):
        g = int(g_label.split("_")[1])
        mean_g = group_means[g]
        alpha = mean_g * (1.0 / dispersion)
        alpha = np.clip(alpha, 1e-3, None)
        p = rng.dirichlet(alpha)
        depth = rng.integers(library_size[0], library_size[1])
        counts[i] = rng.multinomial(depth, p)

    counts_df = pd.DataFrame(counts, index=sample_ids, columns=feature_ids)
    groups = pd.Series(group_labels, index=sample_ids, name="group")
    counts_df.attrs["differential_features"] = [feature_ids[i] for i in diff_idx]
    return counts_df, groups


def simulate_taxonomy(feature_ids: list[str], seed: int | None = 0) -> pd.DataFrame:
    """Attach a plausible 7-rank taxonomy string to each ASV id.

    16S amplicon data is typically only reliably resolved to genus level
    (species assignment from a single hypervariable region is unreliable),
    so ``species`` is left blank here — unlike a shotgun profiler's output.

    Useful for testing rank-collapsing and taxonomic-barplot code without
    needing a real QIIME2 taxonomy.tsv on disk.
    """
    rng = _rng(seed)
    rows = []
    for fid in feature_ids:
        genus = rng.choice(_GENERA)
        rows.append(
            {
                "feature_id": fid,
                "domain": "Bacteria",
                "phylum": rng.choice(
                    ["Bacillota", "Bacteroidota", "Pseudomonadota", "Actinomycetota", "Verrucomicrobiota"]
                ),
                "class": f"class_{rng.integers(1, 6)}",
                "order": f"order_{rng.integers(1, 10)}",
                "family": f"{genus}aceae",
                "genus": genus,
                "species": "",
            }
        )
    return pd.DataFrame(rows).set_index("feature_id")
