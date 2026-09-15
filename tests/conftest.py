import matplotlib

matplotlib.use("Agg")  # headless backend for tests/CI

import pytest

from amplicon_diversity.simulate import simulate_asv_table, simulate_taxonomy


@pytest.fixture
def amplicon_data():
    counts, groups = simulate_asv_table(n_samples=24, n_features=30, seed=1)
    return counts, groups


@pytest.fixture
def taxonomy_data(amplicon_data):
    counts, _ = amplicon_data
    return simulate_taxonomy(list(counts.columns), seed=1)
