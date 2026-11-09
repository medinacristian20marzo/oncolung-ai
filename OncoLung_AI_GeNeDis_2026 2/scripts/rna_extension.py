#!/usr/bin/env python3
"""Fixed four-gene mutation/expression exploration, separate from survival models.

Default: reproduce from the included 510-sample four-gene expression subset.
Optional --rna: extract that subset from the complete cBioPortal RSEM matrix.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu, false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
GENES = ["EGFR", "KRAS", "TP53", "STK11"]
ENTREZ = {"EGFR": 1956, "KRAS": 3845, "TP53": 7157, "STK11": 6794}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_matrix(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f, delimiter="\t"))
    if len(header) != len(set(header)) or header[:2] != ["Hugo_Symbol", "Entrez_Gene_Id"]:
        raise ValueError("Duplicate or unexpected expression matrix headers")
    frame = pd.read_csv(path, sep="\t", float_precision="round_trip")
    values = frame.iloc[:, 2:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    eid = pd.to_numeric(frame["Entrez_Gene_Id"], errors="coerce")
    qc = {
        "source_file": path.name, "source_sha256": sha(path), "source_bytes": path.stat().st_size,
        "gene_rows": len(frame), "sample_columns": len(header) - 2,
        "missing_expression_values": int(np.isnan(values).sum()),
        "infinite_expression_values": int(np.isinf(values).sum()),
        "negative_expression_values": int((values < 0).sum()),
        "zero_expression_values": int((values == 0).sum()),
        "minimum_expression": float(np.nanmin(values)), "maximum_expression": float(np.nanmax(values)),
        "missing_gene_symbols": int(frame["Hugo_Symbol"].isna().sum()),
        "duplicate_nonmissing_symbol_rows_after_first": int(frame["Hugo_Symbol"].dropna().duplicated().sum()),
        "duplicate_nonmissing_entrez_rows_after_first": int(eid.dropna().duplicated().sum()),
        "gene_selection": "Only four preselected genes; exact symbol and Entrez ID; no aggregation of duplicate genes.",
    }
    result = pd.DataFrame({"SAMPLE_ID": header[2:]})
    for gene in GENES:
        symbol_rows = frame["Hugo_Symbol"].eq(gene)
        id_rows = eid.eq(ENTREZ[gene])
        if symbol_rows.sum() != 1 or id_rows.sum() != 1 or not symbol_rows.equals(id_rows):
            raise ValueError(f"Ambiguous symbol/Entrez mapping: {gene}")
        result[f"{gene}_RSEM"] = frame.loc[symbol_rows].iloc[0, 2:].to_numpy(dtype=float)
    return result, qc


def run(cohort_path, map_path, expression_path, raw_rna, out):
    cohort = pd.read_csv(cohort_path, float_precision="round_trip")
    mapping = pd.read_csv(map_path)
    source_qc = None
    if raw_rna is not None:
        expression, source_qc = extract_matrix(raw_rna)
        input_hash = sha(raw_rna)
        mode = "full_matrix_extraction"
    else:
        expression = pd.read_csv(expression_path, float_precision="round_trip")
        input_hash = sha(expression_path)
        mode = "included_four_gene_subset"
    for data, key in [(cohort, "PATIENT_ID"), (mapping, "PATIENT_ID"),
                      (mapping, "SAMPLE_ID"), (expression, "SAMPLE_ID")]:
        if data[key].isna().any() or data[key].duplicated().any():
            raise ValueError(f"Missing or duplicate {key}; explicit resolution required")
    if not cohort[GENES].isin([0, 1]).all().all():
        raise ValueError("Missing or nonbinary mutation indicators")
    if not set(cohort["PATIENT_ID"]).issubset(mapping["PATIENT_ID"]):
        raise ValueError("Cohort patient absent from sample mapping")
    expr_columns = [f"{g}_RSEM" for g in GENES]
    x = expression[expr_columns].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Target-gene expression must be finite and nonnegative; no imputation")
    joined = cohort.merge(mapping, on="PATIENT_ID", how="left", validate="one_to_one")
    if not joined["MUTATION_PROFILE"].eq("WXS").all() or not joined["SAMPLE_TYPE"].eq("Primary").all():
        raise ValueError("Unexpected sample type or mutation profile")
    availability = joined[["PATIENT_ID", "SAMPLE_ID", "OS_STATUS"]].copy()
    availability["RNA_AVAILABLE"] = availability["SAMPLE_ID"].isin(expression["SAMPLE_ID"])
    integrated = joined.merge(expression, on="SAMPLE_ID", how="inner", validate="one_to_one")
    if integrated.empty:
        raise ValueError("No exact RNA/clinical sample-ID matches")
    for gene in GENES:
        integrated[f"{gene}_log2_RSEM_plus1"] = np.log2(integrated[f"{gene}_RSEM"] + 1.0)
    rows = []
    for gene in GENES:
        col = f"{gene}_log2_RSEM_plus1"
        x1 = integrated.loc[integrated[gene].eq(1), col].to_numpy()
        x0 = integrated.loc[integrated[gene].eq(0), col].to_numpy()
        if min(len(x1), len(x0)) < 10:
            raise ValueError(f"Insufficient group size for the specified asymptotic comparison: {gene}")
        u, p = mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic", use_continuity=True)
        q1 = np.quantile(x1, [.25, .5, .75], method="linear")
        q0 = np.quantile(x0, [.25, .5, .75], method="linear")
        rows.append({
            "gene": gene, "n_mutation_0": len(x0), "n_mutation_1": len(x1),
            "q25_0": q0[0], "median_0": q0[1], "q75_0": q0[2],
            "q25_1": q1[0], "median_1": q1[1], "q75_1": q1[2],
            "median_difference_1_minus_0": q1[1] - q0[1],
            "rank_biserial_1_vs_0": 2 * u / (len(x1) * len(x0)) - 1,
            "mann_whitney_U_mutation_1": u, "p_two_sided": p,
        })
    results = pd.DataFrame(rows)
    results["p_adjusted_BH_4_tests"] = false_discovery_control(results["p_two_sided"].to_numpy(), method="bh")
    out.mkdir(parents=True, exist_ok=True)
    expression.to_csv(out / "rna_four_gene_rsem.csv", index=False)
    integrated.to_csv(out / "clinical_mutation_rna_293.csv", index=False)
    availability.to_csv(out / "rna_availability_295.csv", index=False)
    results.to_csv(out / "mutation_expression_results.csv", index=False)
    if source_qc is not None:
        (out / "full_matrix_qc.json").write_text(json.dumps(source_qc, indent=2) + "\n")
    audit = {
        "analysis": "Exploratory unadjusted within-gene mutation/expression comparisons; no RNA survival model",
        "input_mode": mode, "input_sha256": input_hash, "cohort_sha256": sha(cohort_path),
        "sample_map_sha256": sha(map_path), "cohort_n": len(cohort), "rna_samples_in_input": len(expression),
        "matched_n": len(integrated), "matched_events": int(integrated["OS_STATUS"].sum()),
        "unmatched": availability.loc[~availability["RNA_AVAILABLE"]].to_dict(orient="records"),
        "sample_matching": "Exact clinical SAMPLE_ID = mutation sample ID = RNA matrix column; one sample per patient",
        "matched_sample_types": integrated["SAMPLE_TYPE"].value_counts().to_dict(),
        "matched_mutation_profiles": integrated["MUTATION_PROFILE"].value_counts().to_dict(),
        "genes": ENTREZ, "normalization": "Input cBioPortal metadata: batch-normalized RSEM, Illumina HiSeq RNASeqV2; log2(RSEM+1) for analysis",
        "tests": "Four two-sided Mann-Whitney U tests, asymptotic tie correction and continuity correction; Benjamini-Hochberg across these four tests only",
        "seed_plot_jitter_only": 42,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    }
    (out / "rna_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    jitter = np.random.default_rng(42)
    for ax, row in zip(axes.flat, rows):
        gene = row["gene"]
        groups = [integrated.loc[integrated[gene].eq(i), f"{gene}_log2_RSEM_plus1"].to_numpy() for i in [0, 1]]
        colors = ["#28718A", "#BA5D37"]
        bp = ax.boxplot(groups, positions=[0, 1], widths=.45, patch_artist=True,
                        showfliers=False, medianprops={"color": "black", "linewidth": 1.6})
        for i, (ys, color, box) in enumerate(zip(groups, colors, bp["boxes"])):
            box.set_facecolor(color); box.set_alpha(.24)
            ax.scatter(i + jitter.uniform(-.17, .17, size=len(ys)), ys, color=color,
                       s=12, alpha=.45, linewidths=0, zorder=2)
        adjusted = results.loc[results["gene"].eq(gene), "p_adjusted_BH_4_tests"].iloc[0]
        ax.set_title(f"{gene}  |  BH-adjusted p = {adjusted:.3g}", fontweight="bold")
        ax.set_xticks([0, 1], [f"No qualifying variant\n(n={len(groups[0])})", f"Qualifying variant\n(n={len(groups[1])})"])
        ax.set_ylabel("log2(batch-normalized RSEM + 1)")
        ax.grid(axis="y", alpha=.18)
    fig.suptitle(f"Matched mutation and RNA expression in TCGA-LUAD (n={len(integrated)})", fontsize=15, fontweight="bold", y=.98)
    fig.text(.5, .018, "Exploratory, unadjusted comparisons. Boxes: median and IQR; whiskers: 1.5 IQR. Each point is one patient.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94), h_pad=2)
    fig.savefig(out / "mutation_expression.png", dpi=180)
    plt.close(fig)
    print(f"RNA matched patients: {len(integrated)}/{len(cohort)}; deaths: {int(integrated['OS_STATUS'].sum())}")
    print(results[["gene", "n_mutation_0", "n_mutation_1", "median_difference_1_minus_0",
                   "rank_biserial_1_vs_0", "p_two_sided", "p_adjusted_BH_4_tests"]].to_string(index=False))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, default=ROOT / "tcga_luad_day0.csv")
    ap.add_argument("--mapping", type=Path, default=ROOT / "data/processed/cohort_sample_map.csv")
    ap.add_argument("--expression", type=Path, default=ROOT / "data/processed/rna_four_gene_rsem.csv")
    ap.add_argument("--rna", type=Path, default=None, help="Optional full cBioPortal data_mrna_seq_v2_rsem.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/rna_reproduced")
    a = ap.parse_args()
    run(a.cohort, a.mapping, a.expression, a.rna, a.out)
