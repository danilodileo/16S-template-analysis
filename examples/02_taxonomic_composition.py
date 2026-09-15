"""Taxonomic composition workflow: rank collapsing, composition barplot, core microbiome.

Simulates an ASV table with genus-level taxonomy assignments (as if produced
by QIIME2's classify-sklearn against a reference database) and reproduces a
typical taxonomic-profiling figure plus a core-microbiome summary.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from amplicon_diversity.simulate import simulate_asv_table, simulate_taxonomy
from amplicon_diversity.taxonomy import collapse_to_rank, core_taxa, top_taxa
from amplicon_diversity.viz import plot_taxa_barplot

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)


def main():
    counts, groups = simulate_asv_table(n_samples=30, n_features=100, seed=7)
    taxonomy = simulate_taxonomy(list(counts.columns), seed=7)
    print(f"Simulated ASV table: {counts.shape[0]} samples x {counts.shape[1]} ASVs")

    genus_table = collapse_to_rank(counts, taxonomy, rank="genus")
    print(f"Collapsed to {genus_table.shape[1]} genera")

    top_genera = top_taxa(genus_table, n=12)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_taxa_barplot(top_genera, groups, ax=ax, title="Genus-level composition")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_taxa_barplot_genus.png", dpi=150)
    print(f"Saved {FIGURES / '03_taxa_barplot_genus.png'}")

    core = core_taxa(genus_table, prevalence=0.8, min_abundance=0.001)
    print(f"Core genera (present in >=80% of samples): {len(core)} of {genus_table.shape[1]}")
    print(sorted(core))


if __name__ == "__main__":
    main()
