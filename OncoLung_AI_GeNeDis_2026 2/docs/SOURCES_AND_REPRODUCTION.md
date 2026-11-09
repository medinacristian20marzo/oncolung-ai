# Data sources and reproduction

## Source

- Study: **Lung Adenocarcinoma (TCGA, PanCancer Atlas)**, cBioPortal ID `luad_tcga_pan_can_atlas_2018`
- [Study page](https://www.cbioportal.org/study/summary?id=luad_tcga_pan_can_atlas_2018) · [files on cBioPortal DataHub](https://github.com/cBioPortal/datahub/tree/master/public/luad_tcga_pan_can_atlas_2018) · [file formats](https://docs.cbioportal.org/file-formats/)
- Downloaded in November 2026.

## Files used

| File | Used for | In this folder |
|---|---|---|
| `data_clinical_patient.txt` | Age, sex, stage, overall survival | yes |
| `data_clinical_sample.txt` | Patient–sample links, sample type | yes |
| `data_gene_panel_matrix.txt`, `case_lists/cases_sequenced.txt` | Which samples were exome-sequenced | yes |
| `data_timeline_sample_acquisition.txt` | Day of sample acquisition | yes |
| `mutations_four_gene_extract.tsv` | Mutation rows for EGFR, KRAS, TP53, STK11 (668 rows) from `data_mutations.txt` | yes |
| `data_mrna_seq_v2_rsem.txt` | Expression matrix, 20,531 × 510 | no (73 MB) |
| `data/processed/rna_four_gene_rsem.csv` | Expression of the four genes in all 510 samples | yes |

SHA-256 hashes of the input files are in `source_input_hashes.json`.

## Cohort

1. 566 patients in the study.
2. 489 with age, sex, pathological stage, and overall survival with follow-up > 0.
3. 295 with sample acquisition at day zero (`START_DATE == 0` in the timeline file).

Stage sub-categories (IA, IB, ...) are grouped into I–IV.

## Mutation coding

A gene is coded 1 if the sample has at least one variant with `FILTER == "PASS"` and `Variant_Classification` in:

```text
Missense_Mutation, Nonsense_Mutation, Nonstop_Mutation,
Frame_Shift_Del, Frame_Shift_Ins, In_Frame_Del, In_Frame_Ins,
Splice_Site, Translation_Start_Site
```

Otherwise 0, provided the sample is in the sequenced list. `Splice_Region` variants are not counted. Of the 668 extracted rows, 582 qualify.

## Rebuild

```bash
python scripts/reconstruct_cohort.py     # writes outputs/reconstructed/ and checks it against tcga_luad_day0.csv
python scripts/rna_extension.py          # uses the four-gene expression file
```

To start from the full expression matrix, download `data_mrna_seq_v2_rsem.txt` from the study and run:

```bash
python scripts/rna_extension.py --rna /path/to/data_mrna_seq_v2_rsem.txt --out outputs/rna_from_full_matrix
```

The four-gene file and the full matrix give identical results.

## References

1. Cancer Genome Atlas Research Network. Comprehensive molecular profiling of lung adenocarcinoma. *Nature* 2014;511:543–550. [doi:10.1038/nature13385](https://doi.org/10.1038/nature13385)
2. [TCGA PanCancer Atlas](https://gdc.cancer.gov/about-data/publications/pancanatlas)
3. Cerami E et al. The cBio cancer genomics portal. *Cancer Discov* 2012;2:401–404.
4. Iba K et al. Bootstrap-based optimism correction methods. *BMC Med Res Methodol* 2021;21:9. [doi:10.1186/s12874-020-01201-w](https://doi.org/10.1186/s12874-020-01201-w)
5. Benjamini Y, Hochberg Y. Controlling the false discovery rate. *J R Stat Soc B* 1995;57:289–300.
6. [lifelines](https://lifelines.readthedocs.io) and [SciPy](https://docs.scipy.org) documentation.
