# Example data: QIIME2 "Moving Pictures" (Caporaso et al. 2011)

`moving_pictures_asv_table.csv`, `moving_pictures_taxonomy.csv`, and
`moving_pictures_metadata.csv` are a cleaned export of the official
**QIIME2 "Moving Pictures" tutorial** dataset — the canonical example
dataset for the QIIME2 amplicon pipeline, originally from **Caporaso et al.,
"Moving pictures of the human microbiome"**, *Genome Biology*, 2011
([doi:10.1186/gb-2011-12-5-r50](https://doi.org/10.1186/gb-2011-12-5-r50)).

Downloaded directly from QIIME2's own hosting (2024.2 release):

- ASV feature table: <https://docs.qiime2.org/2024.2/data/tutorials/moving-pictures/table.qza>
  (DADA2-denoised — genuine **ASVs**, not 97%-similarity OTUs)
- Taxonomy: <https://docs.qiime2.org/2024.2/data/tutorials/moving-pictures/taxonomy.qza>
  (`classify-sklearn` against Greengenes 13-8, 515F/806R region)
- Metadata: <https://data.qiime2.org/2024.2/tutorials/moving-pictures/sample_metadata.tsv>

34 stool/skin/oral samples from 2 subjects, sampled repeatedly over ~18
months at 4 body sites (gut, tongue, left palm, right palm) — one of the
strongest, most reproducible ecological signals in human microbiome data
(body habitat shapes community composition far more than almost any other
variable), which makes it a good sanity-check dataset: if a pipeline can't
recover *this* signal, something is wrong with the pipeline, not the
biology.

## Files

| File | Shape | Description |
|---|---|---|
| `moving_pictures_asv_table.csv` | 34 samples x 750 ASVs | Raw DADA2 counts, samples as rows. 20 host-derived (mitochondria/chloroplast) ASVs removed as a standard QC step. |
| `moving_pictures_taxonomy.csv` | 750 ASVs x 8 columns | domain -> species lineage (Greengenes 13-8) + classifier confidence. `species` is mostly blank — 16S resolves reliably only to genus. |
| `moving_pictures_metadata.csv` | 34 samples x 6 columns | `body_site`, `subject`, `year`/`month`/`day`, `days_since_start`, `antibiotic_usage` |

Reproduced from the raw QIIME2 artifacts with
[`prepare_moving_pictures.py`](prepare_moving_pictures.py): the feature
table was extracted from the `.biom` file inside `table.qza` (via the
`biom-format` package), taxonomy was parsed out of the
`k__...;p__...;...;s__...` lineage string in `taxonomy.qza`, and both were
aligned to the sample IDs in the metadata. See
[`examples/04_real_data_moving_pictures.py`](../../examples/04_real_data_moving_pictures.py)
for the full analysis built on top of it.

## License

The underlying study (Caporaso et al. 2011, *Genome Biology*) is open access
under CC-BY. The exact redistribution terms for these specific
QIIME2-processed artifacts are not stated on docs.qiime2.org itself; a
related community tutorial site (moving-pictures-tutorial.readthedocs.io)
lists its own materials as CC BY-NC-ND, which would restrict derivatives —
out of caution, treat this data as **non-commercial, attribution-required,
personal/portfolio use only**, not for redistribution, and re-verify the
license before any public or commercial use.
