import numpy as np
import pytest

from amplicon_diversity.diversity import (
    alpha_diversity,
    chao1,
    observed_features,
    pielou_evenness,
    shannon,
    simpson,
)


def test_shannon_uniform_vs_single_taxon():
    uniform = np.array([25, 25, 25, 25])
    single = np.array([100, 0, 0, 0])
    assert shannon(uniform) == pytest.approx(np.log(4))
    assert shannon(single) == pytest.approx(0.0)


def test_simpson_bounds():
    uniform = np.array([10, 10, 10, 10, 10])
    single = np.array([50, 0, 0, 0, 0])
    assert 0 <= simpson(uniform) <= 1
    assert simpson(single) == pytest.approx(0.0)
    assert simpson(uniform) > simpson(single)


def test_pielou_evenness_perfect_evenness_is_one():
    uniform = np.array([10, 10, 10, 10])
    assert pielou_evenness(uniform) == pytest.approx(1.0, abs=1e-9)


def test_observed_features_counts_nonzero():
    counts = np.array([5, 0, 3, 0, 1])
    assert observed_features(counts) == 3


def test_chao1_at_least_observed_richness():
    counts = np.array([1, 1, 1, 5, 0, 0])
    assert chao1(counts) >= observed_features(counts)


def test_empty_sample_returns_zero_diversity():
    empty = np.zeros(5)
    assert shannon(empty) == 0.0
    assert simpson(empty) == 0.0
    assert observed_features(empty) == 0


def test_alpha_diversity_on_table(amplicon_data):
    counts, _ = amplicon_data
    result = alpha_diversity(counts, metric="shannon")
    assert len(result) == counts.shape[0]
    assert (result >= 0).all()


def test_alpha_diversity_rejects_unknown_metric(amplicon_data):
    counts, _ = amplicon_data
    with pytest.raises(ValueError):
        alpha_diversity(counts, metric="not_a_metric")
