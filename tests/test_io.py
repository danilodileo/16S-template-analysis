import biom
import pandas as pd
import pytest
from biom.util import biom_open

from amplicon_diversity.io import load_abundance_table, load_biom_table, save_abundance_table, validate_table


def test_save_and_load_roundtrip(tmp_path, amplicon_data):
    counts, _ = amplicon_data
    path = tmp_path / "table.tsv"
    save_abundance_table(counts, path)
    loaded = load_abundance_table(path)
    pd.testing.assert_frame_equal(loaded, counts, check_dtype=False)


def test_save_and_load_csv_roundtrip(tmp_path, amplicon_data):
    counts, _ = amplicon_data
    path = tmp_path / "table.csv"
    save_abundance_table(counts, path)
    loaded = load_abundance_table(path)
    pd.testing.assert_frame_equal(loaded, counts, check_dtype=False)


def test_load_features_as_rows_transposes(tmp_path, amplicon_data):
    counts, _ = amplicon_data
    path = tmp_path / "features_by_sample.tsv"
    save_abundance_table(counts.T, path)
    loaded = load_abundance_table(path, samples_as_rows=False)
    pd.testing.assert_frame_equal(loaded, counts, check_dtype=False)


def test_load_biom_table_roundtrip(tmp_path, amplicon_data):
    counts, _ = amplicon_data
    table = biom.Table(
        counts.T.values, observation_ids=counts.columns.tolist(), sample_ids=counts.index.tolist()
    )
    path = tmp_path / "table.biom"
    with biom_open(path, "w") as f:
        table.to_hdf5(f, generated_by="test")
    loaded = load_biom_table(path)
    pd.testing.assert_frame_equal(loaded.loc[counts.index, counts.columns], counts, check_dtype=False)


def test_validate_table_passes_on_good_table(amplicon_data):
    counts, _ = amplicon_data
    validate_table(counts)  # should not raise


def test_validate_table_rejects_negative_values(amplicon_data):
    counts, _ = amplicon_data
    bad = counts.copy()
    bad.iloc[0, 0] = -5
    with pytest.raises(ValueError):
        validate_table(bad)


def test_validate_table_rejects_empty():
    with pytest.raises(ValueError):
        validate_table(pd.DataFrame())


def test_validate_table_rejects_duplicate_ids(amplicon_data):
    counts, _ = amplicon_data
    bad = counts.copy()
    bad.columns = ["dup"] * bad.shape[1]
    with pytest.raises(ValueError):
        validate_table(bad)
