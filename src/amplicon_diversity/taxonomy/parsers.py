"""Parser for QIIME2-style ASV taxonomy assignments (e.g. classify-sklearn output)."""
from __future__ import annotations

from typing import IO

import pandas as pd

_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
_PREFIXES = ["k__", "p__", "c__", "o__", "f__", "g__", "s__"]


def parse_qiime2_taxonomy(source: str | IO) -> pd.DataFrame:
    """Parse a QIIME2 ``taxonomy.tsv`` (exported from a ``taxonomy.qza``) into a tidy DataFrame.

    Expects the standard 3-column export: ``Feature ID``, ``Taxon`` (a
    semicolon-separated ``k__...;p__...;...;s__...`` lineage string, as
    produced by e.g. ``qiime feature-classifier classify-sklearn``), and
    ``Confidence``.

    Parameters
    ----------
    source : str or file-like
        Path to a QIIME2-exported taxonomy TSV, or an open file-like/StringIO
        with the same content.

    Returns
    -------
    DataFrame indexed by feature (ASV) id, with one column per rank
    (domain, phylum, class, order, family, genus, species) plus
    ``confidence``. Ranks the classifier could not resolve are empty strings
    — 16S data is typically only reliable down to genus.
    """
    raw = pd.read_csv(source, sep="\t")
    raw = raw.rename(columns={"Feature ID": "feature_id", "Taxon": "taxon", "Confidence": "confidence"})
    raw = raw.set_index("feature_id")

    records = []
    for fid, taxon in raw["taxon"].items():
        parts = [p.strip() for p in str(taxon).split(";")]
        row = {"feature_id": fid}
        for rank, prefix in zip(_RANKS, _PREFIXES):
            row[rank] = next((p[len(prefix):] for p in parts if p.startswith(prefix)), "")
        records.append(row)

    taxonomy = pd.DataFrame(records).set_index("feature_id")
    taxonomy["confidence"] = raw["confidence"]
    return taxonomy
