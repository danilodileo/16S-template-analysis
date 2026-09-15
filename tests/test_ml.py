import numpy as np

from amplicon_diversity.ml import clr_transform, cross_validate_classifier, feature_importance
from amplicon_diversity.simulate import simulate_asv_table


def test_clr_transform_rows_sum_to_zero(amplicon_data):
    counts, _ = amplicon_data
    clr = clr_transform(counts)
    assert clr.shape == counts.shape
    np.testing.assert_allclose(clr.sum(axis=1).values, 0.0, atol=1e-8)


def test_cross_validate_classifier_beats_chance_on_separable_data():
    counts, groups = simulate_asv_table(
        n_samples=80, n_features=15, n_differential=6, effect_size=10.0, dispersion=0.05, seed=7
    )
    result = cross_validate_classifier(counts, groups, n_splits=5, seed=0)
    assert result["accuracy"] > 0.6
    assert result["roc_auc"] is not None
    assert result["roc_auc"] > 0.6
    assert len(result["predictions"]) == counts.shape[0]


def test_feature_importance_matches_feature_count(amplicon_data):
    counts, groups = amplicon_data
    result = cross_validate_classifier(counts, groups, n_splits=3, seed=0)
    importance = feature_importance(result["fitted_model"], list(counts.columns))
    assert len(importance) == counts.shape[1]
    assert (importance >= 0).all()
    assert importance.index[0] in counts.columns
