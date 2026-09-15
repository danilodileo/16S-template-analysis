"""Core amplicon ecology workflow: alpha diversity, beta diversity, ordination, PERMANOVA.

Simulates an ASV table for two conditions (e.g. healthy vs. disease), then
reproduces the core steps of a standard amplicon diversity analysis, saving
each figure to ../figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from amplicon_diversity.diversity import alpha_diversity, beta_diversity, pcoa
from amplicon_diversity.simulate import simulate_asv_table
from amplicon_diversity.stats import permanova
from amplicon_diversity.viz import plot_alpha_diversity, plot_ordination

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)


def main():
    counts, groups = simulate_asv_table(
        n_samples=48, n_features=80, n_differential=20, effect_size=8.0, dispersion=0.15, seed=42
    )
    print(f"Simulated ASV table: {counts.shape[0]} samples x {counts.shape[1]} ASVs")

    shannon = alpha_diversity(counts, metric="shannon")
    chao1 = alpha_diversity(counts, metric="chao1")
    print(f"Mean Shannon diversity by group:\n{shannon.groupby(groups).mean()}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_alpha_diversity(shannon, groups, ax=axes[0], title="Shannon diversity")
    plot_alpha_diversity(chao1, groups, ax=axes[1], title="Chao1 richness")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_alpha_diversity.png", dpi=150)
    print(f"Saved {FIGURES / '01_alpha_diversity.png'}")

    dist = beta_diversity(counts, metric="bray_curtis")
    coords, explained = pcoa(dist, n_components=2)
    permanova_result = permanova(dist, groups, n_permutations=999, seed=0)
    print(
        f"PERMANOVA: pseudo-F={permanova_result['test_statistic']:.2f}, "
        f"p={permanova_result['p_value']:.4f}"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    title = f"PCoA (Bray-Curtis) — PERMANOVA p={permanova_result['p_value']:.3f}"
    plot_ordination(coords, groups, explained_variance=explained, ax=ax, title=title)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_pcoa_bray_curtis.png", dpi=150)
    print(f"Saved {FIGURES / '02_pcoa_bray_curtis.png'}")


if __name__ == "__main__":
    main()
