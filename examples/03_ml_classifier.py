"""Machine learning workflow: classify samples from ASV composition.

Simulates a case/control ASV table with a moderate compositional signal,
CLR-transforms it, cross-validates a random forest classifier, and reports
the ROC curve and the ASVs driving the classification.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from amplicon_diversity.ml import cross_validate_classifier, feature_importance
from amplicon_diversity.simulate import simulate_asv_table
from amplicon_diversity.viz import plot_feature_importance, plot_roc_curve

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)


def main():
    counts, groups = simulate_asv_table(
        n_samples=100, n_features=60, n_differential=10, effect_size=4.0, dispersion=0.35, seed=99
    )
    print(f"Simulated cohort: {counts.shape[0]} samples ({groups.value_counts().to_dict()})")

    result = cross_validate_classifier(counts, groups, n_splits=5, seed=0)
    print(f"5-fold CV accuracy: {result['accuracy']:.3f}, ROC AUC: {result['roc_auc']:.3f}")

    positive_class = sorted(groups.unique())[1]
    y_true = (groups == positive_class).astype(int)
    y_proba = result["probabilities"][positive_class]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_roc_curve(y_true, y_proba, ax=ax, label="Random Forest", title="Disease classifier ROC")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_roc_curve.png", dpi=150)
    print(f"Saved {FIGURES / '04_roc_curve.png'}")

    importance = feature_importance(result["fitted_model"], list(counts.columns))
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_feature_importance(importance, top_n=15, ax=ax, title="Top predictive ASVs")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_feature_importance.png", dpi=150)
    print(f"Saved {FIGURES / '05_feature_importance.png'}")


if __name__ == "__main__":
    main()
