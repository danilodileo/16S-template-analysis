import numpy as np
import pytest

from amplicon_diversity.diversity import beta_diversity, bray_curtis, jaccard, nmds, pcoa


def test_bray_curtis_identical_samples_is_zero():
    x = np.array([1, 2, 3, 4])
    assert bray_curtis(x, x) == pytest.approx(0.0)


def test_bray_curtis_disjoint_samples_is_one():
    x = np.array([10, 0, 0])
    y = np.array([0, 10, 0])
    assert bray_curtis(x, y) == pytest.approx(1.0)


def test_jaccard_bounds_and_identity():
    x = np.array([1, 0, 3])
    y = np.array([0, 5, 3])
    d = jaccard(x, y)
    assert 0 <= d <= 1
    assert jaccard(x, x) == pytest.approx(0.0)


def test_beta_diversity_matrix_is_symmetric_with_zero_diagonal(amplicon_data):
    counts, _ = amplicon_data
    dist = beta_diversity(counts, metric="bray_curtis")
    assert dist.shape == (counts.shape[0], counts.shape[0])
    np.testing.assert_allclose(dist.values, dist.values.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(dist.values), 0.0, atol=1e-10)


def test_pcoa_recovers_expected_number_of_axes(amplicon_data):
    counts, _ = amplicon_data
    dist = beta_diversity(counts, metric="bray_curtis")
    coords, explained = pcoa(dist, n_components=3)
    assert coords.shape == (counts.shape[0], 3)
    assert len(explained) == 3
    assert (explained >= -1e-9).all()


def test_pcoa_separates_known_groups(amplicon_data):
    counts, groups = amplicon_data
    dist = beta_diversity(counts, metric="bray_curtis")
    coords, _ = pcoa(dist, n_components=2)
    g0 = coords.loc[groups[groups == "group_0"].index, "PC1"].mean()
    g1 = coords.loc[groups[groups == "group_1"].index, "PC1"].mean()
    assert abs(g0 - g1) > 1e-6


def test_nmds_returns_valid_stress(amplicon_data):
    counts, _ = amplicon_data
    dist = beta_diversity(counts, metric="bray_curtis")
    coords, stress = nmds(dist, n_components=2, n_init=2, max_iter=100)
    assert coords.shape == (counts.shape[0], 2)
    assert stress >= 0
