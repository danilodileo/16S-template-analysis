"""Rebuild data/example/{feature-table.biom, taxonomy.tsv, tree.nwk, sample_metadata.tsv}
from the raw QIIME2 artifacts.

These files are already committed, so you do not need to run this to use the
toolkit. It is included for transparency/reproducibility: it shows exactly
how the official QIIME2 tutorial artifacts were filtered for
examples/04_real_data_moving_pictures.py. Unlike an earlier version of this
script, it does not flatten anything to CSV — the committed files stay in
their native QIIME2 formats (.biom feature table, a raw Feature ID/Taxon/
Confidence taxonomy.tsv, a Newick tree), the same formats
amplicon_diversity.io.load_biom_table / .taxonomy.parse_qiime2_taxonomy are
built to read, and the same formats a real QIIME2 pipeline hands off to
downstream analysis.

Requires this package installed (`pip install -e .` from the repo root,
already done in its own dev environment) plus `biom-format`.

Usage
-----
1. Download the four official artifacts:
   curl -O https://docs.qiime2.org/2024.2/data/tutorials/moving-pictures/table.qza
   curl -O https://docs.qiime2.org/2024.2/data/tutorials/moving-pictures/taxonomy.qza
   curl -O https://docs.qiime2.org/2024.2/data/tutorials/moving-pictures/rooted-tree.qza
   curl -O https://data.qiime2.org/2024.2/tutorials/moving-pictures/sample_metadata.tsv
2. Unzip the .qza files (they are just zip archives) to get:
   - table.qza -> */data/feature-table.biom
   - taxonomy.qza -> */data/taxonomy.tsv
   - rooted-tree.qza -> */data/tree.nwk
3. python prepare_moving_pictures.py feature-table.biom taxonomy.tsv tree.nwk sample_metadata.tsv
"""
import sys
from pathlib import Path

import biom
import pandas as pd
from biom.util import biom_open

from amplicon_diversity.taxonomy import parse_qiime2_taxonomy


def main(biom_path: Path, taxonomy_path: Path, tree_path: Path, metadata_path: Path, out_dir: Path):
    table = biom.load_table(str(biom_path))
    asv = table.to_dataframe(dense=True).T
    asv.index.name = "sample_id"
    asv = asv.astype(int)

    meta = pd.read_csv(metadata_path, sep="\t")
    meta = meta[meta["sample-id"] != "#q2:types"]
    meta = meta.rename(columns={"sample-id": "sample_id"}).set_index("sample_id")
    meta = meta[
        [
            "body-site", "subject", "year", "month", "day",
            "days-since-experiment-start", "reported-antibiotic-usage",
        ]
    ]
    meta.columns = ["body_site", "subject", "year", "month", "day", "days_since_start", "antibiotic_usage"]

    # parsed only to drive the QC filter below -- the committed taxonomy.tsv
    # stays in raw QIIME2 lineage-string form, parsed at runtime by the
    # example script itself via `taxonomy.parse_qiime2_taxonomy`.
    taxonomy = parse_qiime2_taxonomy(taxonomy_path)

    shared_samples = asv.index.intersection(meta.index)
    asv = asv.loc[shared_samples]
    meta = meta.loc[shared_samples]
    asv = asv.loc[:, asv.sum(axis=0) > 0]

    # standard QC: drop host-derived (mitochondria/chloroplast) ASVs
    ranks = taxonomy.reindex(asv.columns)
    is_contaminant = (ranks["family"] == "mitochondria") | (ranks["class"] == "Chloroplast")
    print(f"Removing {is_contaminant.sum()} host-derived ASVs of {asv.shape[1]}")
    asv = asv.loc[:, ~is_contaminant.values]
    asv = asv.loc[:, asv.sum(axis=0) > 0]

    tax_raw = pd.read_csv(taxonomy_path, sep="\t").set_index("Feature ID")
    tax_raw = tax_raw.loc[asv.columns]
    tax_raw.index.name = "Feature ID"  # .loc[] with an Index arg overwrites the index name

    filtered_table = biom.Table(
        asv.T.values, observation_ids=asv.columns.tolist(), sample_ids=asv.index.tolist()
    )
    with biom_open(out_dir / "feature-table.biom", "w") as f:
        filtered_table.to_hdf5(f, generated_by="prepare_moving_pictures.py")

    tax_raw.to_csv(out_dir / "taxonomy.tsv", sep="\t")
    meta.to_csv(out_dir / "sample_metadata.tsv", sep="\t")
    # the tree's tip set may be (and here, is) a superset of the QC-filtered
    # ASVs -- skbio's UniFrac only requires the tree to cover the taxa used,
    # not match them exactly -- so it's copied through unfiltered.
    (out_dir / "tree.nwk").write_bytes(tree_path.read_bytes())

    print(f"Wrote {asv.shape[0]} samples x {asv.shape[1]} ASVs to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(
            "usage: python prepare_moving_pictures.py "
            "feature-table.biom taxonomy.tsv tree.nwk sample_metadata.tsv"
        )
    main(
        Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(__file__).parent
    )
