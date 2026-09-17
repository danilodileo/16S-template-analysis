"""Loading, saving, and validating samples x features abundance tables."""
from __future__ import annotations

from pathlib import Path
from typing import IO

import biom
import pandas as pd


def load_abundance_table(source: str | Path | IO, samples_as_rows: bool = True) -> pd.DataFrame:
    """Load a tab- or comma-separated abundance table, with samples as rows.

    Parameters
    ----------
    source : str, Path, or file-like
        Delimiter is inferred from the file extension (".csv" -> comma,
        else tab), defaulting to tab for file-like objects.
    samples_as_rows : bool
        Set to False if the file has features as rows and samples as
        columns (common for OTU tables); the result is transposed to the
        samples-as-rows convention used throughout this package.
    """
    sep = ","
    if hasattr(source, "suffix"):
        sep = "," if source.suffix.lower() == ".csv" else "\t"
    elif isinstance(source, str) and not source.lower().endswith(".csv"):
        sep = "\t"

    table = pd.read_csv(source, sep=sep, index_col=0)
    if not samples_as_rows:
        table = table.T
    return table


def load_biom_table(path: str | Path) -> pd.DataFrame:
    """Load a QIIME2/DADA2-native ``.biom`` feature table (e.g. unzipped from
    a QIIME2 ``table.qza``) via the ``biom-format`` package, the same file
    format a real amplicon pipeline outputs.

    Returns a samples x features DataFrame of integer counts, matching this
    package's convention throughout.
    """
    table = biom.load_table(str(path))
    asv = table.to_dataframe(dense=True).T
    asv.index.name = "sample_id"
    return asv.astype(int)


def save_abundance_table(table: pd.DataFrame, destination: str | Path) -> None:
    """Save a samples x features table as TSV (or CSV if the path ends in .csv)."""
    destination = Path(destination)
    sep = "," if destination.suffix.lower() == ".csv" else "\t"
    table.to_csv(destination, sep=sep)


def validate_table(table: pd.DataFrame) -> None:
    """Sanity-check a samples x features table; raises ValueError on problems.

    Checks for: empty table, non-numeric values, negative values, and
    duplicate sample or feature ids.
    """
    if table.empty:
        raise ValueError("table is empty")
    if not table.index.is_unique:
        raise ValueError("duplicate sample ids in table index")
    if not table.columns.is_unique:
        raise ValueError("duplicate feature ids in table columns")
    non_numeric = table.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"non-numeric columns found: {non_numeric}")
    if (table.values < 0).any():
        raise ValueError("table contains negative values")
