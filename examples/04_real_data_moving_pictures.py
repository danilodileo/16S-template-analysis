"""Real-data case study: QIIME2 "Moving Pictures" — human microbiome across body sites.

Runs the full toolkit end to end on a real, published, ASV-level 16S dataset
(see data/example/README.md for provenance and citation): alpha/beta
diversity and PERMANOVA across all four body sites, then a focused
gut-vs-tongue differential abundance and classification analysis.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from amplicon_diversity.diversity import alpha_diversity, beta_diversity, pcoa
from amplicon_diversity.io import load_abundance_table
from amplicon_diversity.ml import cross_validate_classifier, feature_importance
from amplicon_diversity.stats import differential_abundance, permanova
from amplicon_diversity.taxonomy import collapse_to_rank
from amplicon_diversity.viz import (
    plot_alpha_diversity,
    plot_diff_abundance,
    plot_feature_importance,
    plot_ordination,
    plot_roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "example"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def main():
    counts = load_abundance_table(DATA / "moving_pictures_asv_table.csv")
    taxonomy = load_abundance_table(DATA / "moving_pictures_taxonomy.csv")
    metadata = load_abundance_table(DATA / "moving_pictures_metadata.csv")
    body_site = metadata["body_site"]
    print(f"Loaded {counts.shape[0]} samples x {counts.shape[1]} ASVs across body sites:")
    print(body_site.value_counts())

    genus_table = collapse_to_rank(counts, taxonomy, rank="genus")
    genus_table = genus_table.drop(columns=[c for c in [""] if c in genus_table.columns])
    print(f"Collapsed to {genus_table.shape[1]} named genera")

    # --- diversity across all 4 body sites ---
    shannon = alpha_diversity(counts, metric="shannon")
    dist = beta_diversity(counts, metric="bray_curtis")
    coords, explained = pcoa(dist, n_components=2)
    permanova_result = permanova(dist, body_site, n_permutations=999, seed=0)
    print(
        f"PERMANOVA (body site, 4 groups): pseudo-F={permanova_result['test_statistic']:.2f}, "
        f"p={permanova_result['p_value']:.4f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_alpha_diversity(shannon, body_site, ax=axes[0], title="Shannon diversity by body site")
    plot_ordination(
        coords, body_site, explained_variance=explained, ax=axes[1],
        title=f"PCoA — PERMANOVA p={permanova_result['p_value']:.3f}",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "06_real_data_diversity.png", dpi=150)
    print(f"Saved {FIGURES / '06_real_data_diversity.png'}")

    # --- focused comparison: gut vs. tongue ---
    two_site_mask = body_site.isin(["gut", "tongue"])
    genus_two = genus_table.loc[two_site_mask]
    groups_two = body_site.loc[two_site_mask]
    print(f"\nGut vs. tongue subset: {groups_two.value_counts().to_dict()}")

    result = differential_abundance(genus_two, groups_two)
    n_sig = (result["qvalue"] < 0.05).sum()
    print(f"{n_sig} / {result.shape[0]} genera differ at q<0.05 between gut and tongue")
    print(result.head(10))

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_diff_abundance(result, alpha=0.05, ax=ax, title="Gut vs. tongue: genus-level differential abundance")
    fig.tight_layout()
    fig.savefig(FIGURES / "07_real_data_diff_abundance.png", dpi=150)
    print(f"Saved {FIGURES / '07_real_data_diff_abundance.png'}")

    cv_result = cross_validate_classifier(genus_two, groups_two, n_splits=5, seed=0)
    print(f"5-fold CV accuracy: {cv_result['accuracy']:.3f}, ROC AUC: {cv_result['roc_auc']:.3f}")

    positive_class = "tongue"
    y_true = (groups_two == positive_class).astype(int)
    y_proba = cv_result["probabilities"][positive_class]
    importance = feature_importance(cv_result["fitted_model"], list(genus_two.columns))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_roc_curve(y_true, y_proba, ax=axes[0], label="Random Forest", title="Gut vs. tongue classifier ROC")
    plot_feature_importance(importance, top_n=15, ax=axes[1], title="Top predictive genera")
    fig.tight_layout()
    fig.savefig(FIGURES / "08_real_data_classifier.png", dpi=150)
    print(f"Saved {FIGURES / '08_real_data_classifier.png'}")


if __name__ == "__main__":
    main()
