# amplicon-diversity

A small, well-tested Python toolkit for **16S/18S rRNA amplicon (ASV-based)**
microbial community analysis: diversity ecology, taxonomic profiling, and
machine learning on compositional data — with runnable examples, figures,
and unit tests.

Built as a thin, well-documented orchestration layer over the same tools a
real Python bioinformatics workflow reaches for —
[`scikit-bio`](https://scikit.bio) (the Python analogue of R's `vegan`),
`biom-format`, and `scikit-learn` — rather than reimplementing statistics
from scratch. The goal is to show a coherent, real pipeline built on
standard, peer-reviewed libraries: `diversity`/`stats`/`taxonomy`/`ml`/`viz`/
`io` give this package a clean, stable API, while `scikit-bio` etc. do the
actual computation (see [Design notes](#design-notes)).

This is scoped to amplicon data specifically (working at **ASV**, not
97%-similarity OTU, resolution — see [Design notes](#design-notes)). It's
one repo in a small set of focused, single-topic portfolio projects rather
than one large toolkit spanning every kind of -omics data.

## Why this exists

This is a portfolio project: a compact demonstration of how I approach
scientific software — package structure, reproducible synthetic data for
testing, an honest end-to-end run on real published data, and figures that
would hold up in a lab meeting.

## Install

```bash
git clone https://github.com/danilodileo/16S-template-analysis.git
cd 16S-template-analysis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Quickstart

```python
from amplicon_diversity.simulate import simulate_asv_table
from amplicon_diversity.diversity import alpha_diversity, beta_diversity, pcoa
from amplicon_diversity.stats import permanova
from amplicon_diversity.viz import plot_ordination

counts, groups = simulate_asv_table(n_samples=48, n_features=80, seed=42)

shannon = alpha_diversity(counts, metric="shannon")
dist = beta_diversity(counts, metric="bray_curtis")
coords, explained = pcoa(dist)
result = permanova(dist, groups)

plot_ordination(coords, groups, explained_variance=explained,
                 title=f"PCoA — PERMANOVA p={result['p_value']:.3f}")
```

## What's in the package

| Module | Covers | Key functions | Backed by |
|---|---|---|---|
| [`amplicon_diversity.diversity`](src/amplicon_diversity/diversity) | Alpha/beta diversity, ordination, UniFrac | `shannon`, `chao1`, `bray_curtis`, `beta_diversity`, `pcoa`, `nmds` | `scikit-bio` (NMDS: `scikit-learn`) |
| [`amplicon_diversity.taxonomy`](src/amplicon_diversity/taxonomy) | ASV taxonomy & composition | `parse_qiime2_taxonomy`, `collapse_to_rank`, `top_taxa`, `core_taxa` | pandas |
| [`amplicon_diversity.stats`](src/amplicon_diversity/stats) | Group-comparison statistics | `differential_abundance` (ANCOM), `permanova` | `scikit-bio` |
| [`amplicon_diversity.ml`](src/amplicon_diversity/ml) | ML on compositional data | `clr_transform`, `cross_validate_classifier`, `feature_importance` | `scikit-bio`, `scikit-learn` |
| [`amplicon_diversity.viz`](src/amplicon_diversity/viz) | Shared plotting | boxplots, ordination, taxa barplots, ANCOM plot, heatmap, ROC | matplotlib/seaborn |
| [`amplicon_diversity.io`](src/amplicon_diversity/io) | Table I/O & validation | `load_abundance_table`, `load_biom_table`, `save_abundance_table`, `validate_table` | `biom-format` |
| [`amplicon_diversity.simulate`](src/amplicon_diversity/simulate) | Synthetic data generators | `simulate_asv_table`, `simulate_taxonomy` | NumPy |

All abundance data follows one convention throughout: **samples as rows,
features (ASVs) as columns**.

## Examples

Each script in [`examples/`](examples) is runnable end to end and saves its
figures to [`figures/`](figures).

1. [`01_diversity_ecology.py`](examples/01_diversity_ecology.py) — alpha/beta diversity, PCoA, PERMANOVA
2. [`02_taxonomic_composition.py`](examples/02_taxonomic_composition.py) — rank collapsing, composition barplot, core microbiome
3. [`03_ml_classifier.py`](examples/03_ml_classifier.py) — CLR + random forest + ROC + feature importance
4. [`04_real_data_moving_pictures.py`](examples/04_real_data_moving_pictures.py) — the whole pipeline on a **real published ASV dataset** (see below)

### Diversity & ordination (synthetic)

Two simulated groups with a known compositional shift — alpha diversity is
higher in group 1, and PERMANOVA confirms the groups separate in Bray-Curtis
space (p = 0.001):

![Alpha diversity](figures/01_alpha_diversity.png)
![PCoA ordination](figures/02_pcoa_bray_curtis.png)

### Taxonomic composition (synthetic)

Genus-level composition across samples, and the "core" genera present in
≥80% of them:

![Taxonomic barplot](figures/03_taxa_barplot_genus.png)

### Machine learning: disease classifier (synthetic)

CLR-transformed abundances, 5-fold cross-validated random forest:

![ROC curve](figures/04_roc_curve.png)
![Feature importance](figures/05_feature_importance.png)

### Real data: QIIME2 "Moving Pictures" — human microbiome across body sites

[`data/example/`](data/example) bundles the official **QIIME2 tutorial**
ASV dataset (Caporaso et al. 2011, *Genome Biology* — see
[`data/example/README.md`](data/example/README.md) for full provenance and
licensing notes), kept in its **native QIIME2 formats** (`.biom` feature
table, raw `taxonomy.tsv`, Newick tree — not flattened to CSV): 34 real
samples from 2 subjects across 4 body sites (gut, tongue, left palm, right
palm), denoised to **750 real ASVs** with DADA2 — not OTU clustering.

Body site is one of the strongest signals in human microbiome ecology, and
the toolkit recovers it cleanly two independent ways: PERMANOVA p = 0.001 on
Bray-Curtis dissimilarity *and* on phylogenetic weighted UniFrac (pseudo-F
14.0 vs. 6.7 — UniFrac separates the sites even more cleanly once
evolutionary relatedness between ASVs is taken into account). ANCOM flags 10
genera as differentially abundant between gut and tongue — textbook gut taxa
(*Faecalibacterium*, *Roseburia*, *Ruminococcus*) — and a gut-vs-tongue
random forest classifier hits 100% cross-validated accuracy on them: not
overfitting, just genuinely easy biology.

![Real data diversity](figures/06_real_data_diversity.png)
![Real data classifier](figures/08_real_data_classifier.png)

## Testing

49 tests, 96% coverage, run in CI on Python 3.9–3.12 ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```bash
pytest --cov=amplicon_diversity --cov-report=term-missing
```

Every statistical function is tested against known analytical properties
(e.g. Shannon entropy of a uniform distribution equals log(richness),
Bray-Curtis of identical samples is 0) rather than just "does it run", and
`amplicon_diversity.simulate` gives every test a reproducible, known ground
truth (e.g. which ASVs are truly differentially abundant) to check recovery
against.

## Design notes

- **ASVs, not OTUs**: every example — synthetic and real — works at
  amplicon sequence variant (exact-sequence) resolution, the modern
  DADA2/Deblur replacement for 97%-similarity OTU clustering. The real
  dataset is QIIME2's own DADA2 output, and `taxonomy.parse_qiime2_taxonomy`
  reads the format QIIME2's `classify-sklearn` actually produces — not a
  shotgun-metagenomics classifier's output (that's a different data type
  and belongs in a separate repo).
- **Compositional-data aware**: `ml.clr_transform` uses CLR with
  multiplicative zero-replacement (not an arbitrary pseudocount, which
  distorts the simplex), and `stats.differential_abundance` uses ANCOM
  rather than a naive per-feature test on relative abundances — both avoid
  the spurious correlations that closure (relative abundances summing to 1)
  otherwise creates in compositional data.
- **Built on standard libraries, not reimplemented from scratch**: diversity
  indices, PCoA, PERMANOVA, UniFrac, and ANCOM are thin wrappers around
  `scikit-bio` — the same library (and often the same underlying methods)
  `vegan`/`phyloseq` provide in R (ANCOM's own multiple-comparison
  correction runs through `statsmodels` internally, a transitive
  dependency). This package's own code is the orchestration layer — a
  stable, table-in/table-out API and the plotting on top — not a
  reimplementation of the statistics: a real analysis pipeline should use
  peer-reviewed, battle-tested statistical machinery, not a bespoke one.
- **Species-level taxonomy is deliberately left blank** in the synthetic
  generator: a single 16S hypervariable region genuinely can't resolve
  species reliably, and pretending otherwise would be a modeling error, not
  a simplification.

## License

MIT for the code (see [LICENSE](LICENSE)). The example dataset in
`data/example/` has separate, more restrictive provenance notes — see
[`data/example/README.md`](data/example/README.md) before reusing it
outside this repo.
