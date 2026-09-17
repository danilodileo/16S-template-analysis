"""Real-data case study: QIIME2 "Moving Pictures" — human microbiome across body sites.

Runs the full toolkit end to end on a real, published, ASV-level 16S dataset
(see data/example/README.md for provenance and citation): alpha/beta
diversity (Bray-Curtis and phylogenetic UniFrac) and PERMANOVA across all
four body sites, then a focused gut-vs-tongue differential abundance
(ANCOM) and classification analysis. Loads the QIIME2-native feature table,
taxonomy, and tree formats directly, the same way a real pipeline hands them
off to downstream analysis.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import skbio

from amplicon_diversity.diversity import alpha_diversity, beta_diversity, pcoa
from amplicon_diversity.io import load_biom_table
from amplicon_diversity.ml import cross_validate_classifier, feature_importance
from amplicon_diversity.stats import differential_abundance, permanova
from amplicon_diversity.taxonomy import collapse_to_rank, parse_qiime2_taxonomy
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
    counts = load_biom_table(DATA / "feature-table.biom")
    taxonomy = parse_qiime2_taxonomy(DATA / "taxonomy.tsv")
    tree = skbio.TreeNode.read(str(DATA / "tree.nwk"))
    metadata = pd.read_csv(DATA / "sample_metadata.tsv", sep="\t", index_col=0)
    body_site = metadata["body_site"]
    print(f"Loaded {counts.shape[0]} samples x {counts.shape[1]} ASVs across body sites:")
    print(body_site.value_counts())

    genus_table = collapse_to_rank(counts, taxonomy, rank="genus")
    genus_table = genus_table.drop(columns=[c for c in [""] if c in genus_table.columns])
    print(f"Collapsed to {genus_table.shape[1]} named genera")

    # --- diversity across all 4 body sites: Bray-Curtis (composition) vs.
    # weighted UniFrac (composition + phylogenetic relatedness) ---
    shannon = alpha_diversity(counts, metric="shannon")

    bray_dist = beta_diversity(counts, metric="bray_curtis")
    bray_coords, bray_explained = pcoa(bray_dist, n_components=2)
    bray_permanova = permanova(bray_dist, body_site, n_permutations=999, seed=0)
    print(
        f"PERMANOVA bray_curtis (body site, 4 groups): "
        f"pseudo-F={bray_permanova['test_statistic']:.2f}, p={bray_permanova['p_value']:.4f}"
    )

    unifrac_dist = beta_diversity(counts, metric="weighted_unifrac", tree=tree)
    unifrac_coords, unifrac_explained = pcoa(unifrac_dist, n_components=2)
    unifrac_permanova = permanova(unifrac_dist, body_site, n_permutations=999, seed=0)
    print(
        f"PERMANOVA weighted_unifrac (body site, 4 groups): "
        f"pseudo-F={unifrac_permanova['test_statistic']:.2f}, p={unifrac_permanova['p_value']:.4f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    plot_alpha_diversity(shannon, body_site, ax=axes[0], title="Shannon diversity by body site")
    plot_ordination(
        bray_coords, body_site, explained_variance=bray_explained, ax=axes[1],
        title=f"Bray-Curtis PCoA — PERMANOVA p={bray_permanova['p_value']:.3f}",
    )
    plot_ordination(
        unifrac_coords, body_site, explained_variance=unifrac_explained, ax=axes[2],
        title=f"Weighted UniFrac PCoA — PERMANOVA p={unifrac_permanova['p_value']:.3f}",
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
    n_sig = result["reject_null"].sum()
    print(f"{n_sig} / {result.shape[0]} genera flagged as differentially abundant by ANCOM")
    print(result.head(10))

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_diff_abundance(result, ax=ax, title="Gut vs. tongue: genus-level differential abundance")
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
