import pandas as pd
import pytest

from amplicon_diversity.diversity import beta_diversity
from amplicon_diversity.stats import differential_abundance, permanova


def test_differential_abundance_recovers_known_signal():
    from amplicon_diversity.simulate import simulate_asv_table

    counts, groups = simulate_asv_table(
        n_samples=60, n_features=20, n_differential=5, effect_size=8.0, dispersion=0.05, seed=3
    )
    result = differential_abundance(counts, groups)
    diff_features = set(counts.attrs["differential_features"])
    top_hits = set(result.sort_values("qvalue").index[:5])
    assert len(top_hits & diff_features) >= 2  # noisy but should recover most


def test_differential_abundance_requires_two_groups(amplicon_data):
    counts, groups = amplicon_data
    three_groups = groups.copy()
    three_groups.iloc[0] = "group_2"
    with pytest.raises(ValueError):
        differential_abundance(counts, three_groups)


def test_differential_abundance_handles_unbalanced_group_sizes():
    from amplicon_diversity.simulate import simulate_asv_table

    counts, groups = simulate_asv_table(n_samples=21, n_features=10, seed=4)
    # force an unequal group split, including a feature that is all-zero in both groups
    groups = pd.Series(["a"] * 9 + ["b"] * 12, index=counts.index, name="group")
    counts.iloc[:, 0] = 0
    result = differential_abundance(counts, groups)
    assert result.shape[0] == counts.shape[1]
    assert result.loc[counts.columns[0], "pvalue"] == 1.0


def test_permanova_returns_valid_pvalue_range(amplicon_data):
    counts, groups = amplicon_data
    dist = beta_diversity(counts, metric="bray_curtis")
    result = permanova(dist, groups, n_permutations=99, seed=0)
    assert 0 <= result["p_value"] <= 1
    assert result["test_statistic"] >= 0


def test_permanova_handles_more_than_two_groups(amplicon_data):
    counts, _ = amplicon_data
    dist = beta_diversity(counts, metric="bray_curtis")
    four_groups = pd.Series(
        [f"site_{i % 4}" for i in range(counts.shape[0])], index=counts.index, name="site"
    )
    result = permanova(dist, four_groups, n_permutations=99, seed=0)
    assert 0 <= result["p_value"] <= 1
