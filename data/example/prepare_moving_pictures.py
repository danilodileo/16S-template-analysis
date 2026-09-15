"""Rebuild moving_pictures_{asv_table,taxonomy,metadata}.csv from the raw QIIME2 artifacts.

The CSVs in this folder are already committed, so you do not need to run this
to use the toolkit. It is included for transparency/reproducibility: it shows
exactly how the official QIIME2 tutorial artifacts were turned into the tidy
tables used by examples/04_real_data_moving_pictures.py.

Requires the `biom-format` package (not a runtime dependency of this
toolkit — install it separately: `pip install biom-format`).

Usage
-----
1. Download the three official artifacts:
   curl -O https://docs.qiime2.org/2024.2/data/tutorials/moving-pictures/table.qza
   curl -O https://docs.qiime2.org/2024.2/data/tutorials/moving-pictures/taxonomy.qza
   curl -O https://data.qiime2.org/2024.2/tutorials/moving-pictures/sample_metadata.tsv
2. Unzip the .qza files (they are just zip archives) to get:
   - table.qza -> */data/feature-table.biom
   - taxonomy.qza -> */data/taxonomy.tsv
3. python prepare_moving_pictures.py /path/to/feature-table.biom /path/to/taxonomy.tsv /path/to/sample_metadata.tsv
"""
import sys
from pathlib import Path

import biom
import pandas as pd

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
PREFIXES = ["k__", "p__", "c__", "o__", "f__", "g__", "s__"]


def main(biom_path: Path, taxonomy_path: Path, metadata_path: Path, out_dir: Path):
    table = biom.load_table(str(biom_path))
    asv = table.to_dataframe(dense=True).T
    asv.index.name = "sample_id"
    asv = asv.astype(int)

    meta = pd.read_csv(metadata_path, sep="\t")
    meta = meta[meta["sample-id"] != "#q2:types"]
    meta = meta.rename(columns={"sample-id": "sample_id"}).set_index("sample_id")
    meta = meta[
        ["body-site", "subject", "year", "month", "day", "days-since-experiment-start", "reported-antibiotic-usage"]
    ]
    meta.columns = ["body_site", "subject", "year", "month", "day", "days_since_start", "antibiotic_usage"]

    tax_raw = pd.read_csv(taxonomy_path, sep="\t")
    tax_raw = tax_raw.rename(columns={"Feature ID": "feature_id", "Taxon": "taxon", "Confidence": "confidence"})
    tax_raw = tax_raw.set_index("feature_id")

    records = []
    for fid, taxon in tax_raw["taxon"].items():
        parts = [p.strip() for p in str(taxon).split(";")]
        row = {"feature_id": fid}
        for rank, prefix in zip(RANKS, PREFIXES):
            row[rank] = next((p[len(prefix):] for p in parts if p.startswith(prefix)), "")
        records.append(row)
    taxonomy = pd.DataFrame(records).set_index("feature_id")
    taxonomy["confidence"] = tax_raw["confidence"]

    shared_samples = asv.index.intersection(meta.index)
    asv = asv.loc[shared_samples]
    meta = meta.loc[shared_samples]
    asv = asv.loc[:, asv.sum(axis=0) > 0]
    taxonomy = taxonomy.reindex(asv.columns)

    # standard QC: drop host-derived (mitochondria/chloroplast) ASVs
    is_contaminant = (taxonomy["family"] == "mitochondria") | (taxonomy["class"] == "Chloroplast")
    print(f"Removing {is_contaminant.sum()} host-derived ASVs of {taxonomy.shape[0]}")
    taxonomy = taxonomy[~is_contaminant]
    asv = asv[taxonomy.index.tolist()]
    asv = asv.loc[:, asv.sum(axis=0) > 0]
    taxonomy = taxonomy.loc[asv.columns]

    asv.to_csv(out_dir / "moving_pictures_asv_table.csv")
    taxonomy.to_csv(out_dir / "moving_pictures_taxonomy.csv")
    meta.to_csv(out_dir / "moving_pictures_metadata.csv")
    print(f"Wrote {asv.shape[0]} samples x {asv.shape[1]} ASVs to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python prepare_moving_pictures.py feature-table.biom taxonomy.tsv sample_metadata.tsv")
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(__file__).parent)
