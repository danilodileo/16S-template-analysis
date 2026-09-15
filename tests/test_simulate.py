
from amplicon_diversity.simulate import simulate_asv_table


def test_asv_table_shape_and_reproducibility():
    counts1, groups1 = simulate_asv_table(n_samples=10, n_features=20, seed=42)
    counts2, groups2 = simulate_asv_table(n_samples=10, n_features=20, seed=42)
    assert counts1.shape == (10, 20)
    assert (counts1.values == counts2.values).all()
    assert (groups1.values == groups2.values).all()


def test_asv_table_nonnegative_and_within_library_size():
    counts, _ = simulate_asv_table(n_samples=15, n_features=25, library_size=(1000, 2000), seed=0)
    assert (counts.values >= 0).all()
    totals = counts.sum(axis=1)
    assert (totals >= 1000).all() and (totals < 2000 + 1).all()


def test_asv_ids_look_like_asvs_not_otus():
    counts, _ = simulate_asv_table(n_samples=5, n_features=10, seed=0)
    assert all(col.startswith("ASV_") for col in counts.columns)
