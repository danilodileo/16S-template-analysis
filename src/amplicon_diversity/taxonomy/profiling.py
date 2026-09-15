"""Taxonomic profiling utilities: relative abundance, rank collapsing, core/top taxa."""
from __future__ import annotations

import pandas as pd


def relative_abundance(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize each sample (row) to sum to 1."""
    row_sums = table.sum(axis=1)
    return table.div(row_sums.replace(0, 1), axis=0)


def collapse_to_rank(table: pd.DataFrame, taxonomy: pd.DataFrame, rank: str) -> pd.DataFrame:
    """Sum feature-level abundances up to a coarser taxonomic rank.

    Parameters
    ----------
    table : DataFrame
        Samples x features (e.g. OTUs/ASVs) abundance/count table.
    taxonomy : DataFrame
        Indexed by the same feature ids as ``table``'s columns, with a column
        named ``rank`` giving each feature's assignment at that rank
        (e.g. as produced by :func:`metaomics.simulate.simulate_taxonomy`).
    rank : str
        Column of ``taxonomy`` to group by, e.g. "phylum" or "genus".

    Returns
    -------
    DataFrame, samples x taxa-at-rank, values summed from all matching features.
    """
    if rank not in taxonomy.columns:
        raise ValueError(f"{rank!r} not found in taxonomy columns: {list(taxonomy.columns)}")
    mapping = taxonomy[rank].reindex(table.columns)
    collapsed = table.T.groupby(mapping).sum().T
    return collapsed


def top_taxa(table: pd.DataFrame, n: int = 10, other_label: str = "Other") -> pd.DataFrame:
    """Keep the ``n`` most abundant taxa overall and bin the rest into ``other_label``.

    Handy before plotting a stacked taxonomic barplot with a manageable legend.
    """
    mean_abundance = table.mean(axis=0).sort_values(ascending=False)
    keep = mean_abundance.index[:n]
    result = table[keep].copy()
    remainder = table.drop(columns=keep).sum(axis=1)
    if (remainder > 0).any():
        result[other_label] = remainder
    return result


def core_taxa(table: pd.DataFrame, prevalence: float = 0.8, min_abundance: float = 0.0) -> list[str]:
    """Return features present in at least ``prevalence`` fraction of samples.

    A feature counts as "present" in a sample if its relative abundance
    exceeds ``min_abundance`` there. Useful for defining a "core microbiome".
    """
    rel = relative_abundance(table)
    presence = (rel > min_abundance).mean(axis=0)
    return presence[presence >= prevalence].index.tolist()
