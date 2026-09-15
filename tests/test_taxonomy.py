from io import StringIO

import pandas as pd
import pytest

from amplicon_diversity.taxonomy import (
    collapse_to_rank,
    core_taxa,
    parse_qiime2_taxonomy,
    relative_abundance,
    top_taxa,
)

QIIME2_TAXONOMY = """Feature ID\tTaxon\tConfidence
asv1\tk__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Ruminococcaceae; g__Faecalibacterium; s__\t0.999
asv2\tk__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Bacteroides; s__fragilis\t0.987
asv3\tk__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacteriales; f__Enterobacteriaceae; g__; s__\t0.62
"""


def test_parse_qiime2_taxonomy_builds_lineage():
    df = parse_qiime2_taxonomy(StringIO(QIIME2_TAXONOMY))
    assert df.loc["asv1", "domain"] == "Bacteria"
    assert df.loc["asv1", "phylum"] == "Firmicutes"
    assert df.loc["asv1", "genus"] == "Faecalibacterium"
    assert df.loc["asv2", "species"] == "fragilis"


def test_parse_qiime2_taxonomy_leaves_unresolved_ranks_empty():
    df = parse_qiime2_taxonomy(StringIO(QIIME2_TAXONOMY))
    assert df.loc["asv3", "genus"] == ""
    assert df.loc["asv3", "confidence"] == pytest.approx(0.62)


def test_relative_abundance_rows_sum_to_one(amplicon_data):
    counts, _ = amplicon_data
    rel = relative_abundance(counts)
    row_sums = rel.sum(axis=1)
    assert row_sums.round(6).eq(1.0).all()


def test_collapse_to_rank_preserves_total(amplicon_data, taxonomy_data):
    counts, _ = amplicon_data
    collapsed = collapse_to_rank(counts, taxonomy_data, rank="phylum")
    assert collapsed.shape[1] <= counts.shape[1]
    pd.testing.assert_series_equal(
        collapsed.sum(axis=1).sort_index(), counts.sum(axis=1).sort_index(), check_names=False
    )


def test_top_taxa_caps_columns_and_adds_other(amplicon_data):
    counts, _ = amplicon_data
    top = top_taxa(counts, n=5)
    assert top.shape[1] <= 6  # 5 top + "Other"


def test_core_taxa_returns_subset_of_features(amplicon_data):
    counts, _ = amplicon_data
    core = core_taxa(counts, prevalence=0.5)
    assert set(core).issubset(set(counts.columns))
