#!/usr/bin/env python3
"""Rebuild the frozen 295-patient cohort from included cBioPortal source tables.

Mutation input is a documented four-gene extract, not an independently downloaded
complete MAF. No model is fitted and no date alignment is inferred here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENES = ["EGFR", "KRAS", "TP53", "STK11"]
ENTREZ = {"EGFR": 1956, "KRAS": 3845, "TP53": 7157, "STK11": 6794}
CORE = {
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Translation_Start_Site",
}
STAGE = {
    "STAGE I": "I", "STAGE IA": "I", "STAGE IB": "I",
    "STAGE II": "II", "STAGE IIA": "II", "STAGE IIB": "II",
    "STAGE III": "III", "STAGE IIIA": "III", "STAGE IIIB": "III",
    "STAGE IV": "IV",
}


def table(path):
    return pd.read_csv(path, sep="\t", comment="#", dtype=str)


def require_unique(frame, column):
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f"Missing or duplicate {column}")


def run(source, out, reference):
    p = table(source / "data_clinical_patient.txt")
    s = table(source / "data_clinical_sample.txt")
    panel = table(source / "data_gene_panel_matrix.txt")
    timeline = table(source / "data_timeline_sample_acquisition.txt")
    variants = table(source / "mutations_four_gene_extract.tsv")
    for frame, key in [(p, "PATIENT_ID"), (s, "PATIENT_ID"),
                       (s, "SAMPLE_ID"), (panel, "SAMPLE_ID")]:
        require_unique(frame, key)
    seq_meta = dict(line.split(":", 1) for line in
                    (source / "case_lists/cases_sequenced.txt").read_text().splitlines()
                    if ":" in line)
    sequenced = set(seq_meta["case_list_ids"].split())
    acq = timeline.loc[timeline["EVENT_TYPE"].eq("Sample Acquisition"),
                       ["PATIENT_ID", "SAMPLE_ID", "START_DATE"]].copy()
    require_unique(acq, "SAMPLE_ID")
    linked = s[["PATIENT_ID", "SAMPLE_ID", "SAMPLE_TYPE"]].merge(
        p, on="PATIENT_ID", how="left", validate="one_to_one", indicator=True)
    if not linked["_merge"].eq("both").all():
        raise ValueError("A sample is missing patient metadata")
    linked = linked.drop(columns="_merge").merge(
        panel[["SAMPLE_ID", "mutations"]], on="SAMPLE_ID", how="left",
        validate="one_to_one")
    acq_check = acq.merge(s[["PATIENT_ID", "SAMPLE_ID"]], on="SAMPLE_ID",
                          suffixes=("_timeline", "_clinical"), validate="one_to_one")
    if not acq_check["PATIENT_ID_timeline"].eq(acq_check["PATIENT_ID_clinical"]).all():
        raise ValueError("Timeline and clinical patient IDs disagree")
    linked = linked.merge(acq.drop(columns="PATIENT_ID"), on="SAMPLE_ID",
                           how="left", validate="one_to_one")
    linked["age"] = pd.to_numeric(linked["AGE"], errors="coerce")
    linked["OS_MONTHS"] = pd.to_numeric(linked["OS_MONTHS"], errors="coerce")
    linked["OS_STATUS"] = linked["OS_STATUS"].map({"0:LIVING": 0, "1:DECEASED": 1})
    linked["sex"] = linked["SEX"].map({"Female": "F", "Male": "M"})
    linked["stage"] = linked["AJCC_PATHOLOGIC_TUMOR_STAGE"].str.upper().str.strip().map(STAGE)
    linked["ACQUISITION_START_DATE_REPORTED"] = pd.to_numeric(linked["START_DATE"], errors="coerce")
    complete = linked[["age", "sex", "stage", "OS_MONTHS", "OS_STATUS"]].notna().all(axis=1)
    complete &= np.isfinite(linked["age"]) & np.isfinite(linked["OS_MONTHS"]) & linked["OS_MONTHS"].gt(0)
    selected = linked.loc[complete & linked["ACQUISITION_START_DATE_REPORTED"].eq(0)].copy()
    if not selected["SAMPLE_ID"].isin(sequenced).all() or not selected["mutations"].eq("WXS").all():
        raise ValueError("Selected patients lack the required mutation profiling evidence")
    if not selected["age"].between(0, 120, inclusive="right").all():
        raise ValueError("Unexpected age")
    if not variants["Hugo_Symbol"].isin(GENES).all():
        raise ValueError("Extract contains a non-target gene")
    if not pd.to_numeric(variants["Entrez_Gene_Id"]).eq(variants["Hugo_Symbol"].map(ENTREZ)).all():
        raise ValueError("Gene/Entrez mismatch")
    key = ["Tumor_Sample_Barcode", "Hugo_Symbol", "Chromosome", "Start_Position",
           "End_Position", "Reference_Allele", "Tumor_Seq_Allele2"]
    if variants[key].isna().any().any() or variants.duplicated(key).any():
        raise ValueError("Missing or duplicate variant key")
    if not variants["Tumor_Sample_Barcode"].isin(s["SAMPLE_ID"]).all():
        raise ValueError("Unknown mutation sample")
    qualifying = variants.loc[variants["FILTER"].eq("PASS") & variants["Variant_Classification"].isin(CORE)]
    for gene in GENES:
        samples = qualifying.loc[qualifying["Hugo_Symbol"].eq(gene), "Tumor_Sample_Barcode"]
        selected[gene] = selected["SAMPLE_ID"].isin(samples).astype(int)
    selected = selected.sort_values("PATIENT_ID").reset_index(drop=True)
    columns = ["PATIENT_ID", "OS_MONTHS", "OS_STATUS", "age", "sex", "stage"] + GENES
    result = selected[columns].copy()
    result["OS_STATUS"] = result["OS_STATUS"].astype(int)
    ref = pd.read_csv(reference).sort_values("PATIENT_ID").reset_index(drop=True)
    pd.testing.assert_frame_equal(result, ref[columns], check_dtype=False, check_exact=True)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "tcga_luad_day0_reconstructed.csv", index=False)
    mapping = selected[["PATIENT_ID", "SAMPLE_ID", "SAMPLE_TYPE", "mutations",
                        "ACQUISITION_START_DATE_REPORTED"]].rename(columns={"mutations": "MUTATION_PROFILE"})
    mapping.to_csv(out / "cohort_sample_map.csv", index=False)
    audit = {
        "clinical_patients": len(p), "clinical_samples": len(s),
        "complete_positive_os_candidates": int(complete.sum()),
        "reported_day_zero_selected": len(result), "deaths": int(result["OS_STATUS"].sum()),
        "mutations": {g: int(result[g].sum()) for g in GENES},
        "exact_numeric_and_categorical_match_to_frozen_csv": True,
        "reference_csv_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "mutation_extract_rows": len(variants), "qualifying_variant_rows": len(qualifying),
        "scope": "Rebuilt from the cBioPortal study tables and the four-gene mutation extract.",
    }
    (out / "reconstruction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT / "data/reference_inputs")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/reconstructed")
    ap.add_argument("--reference", type=Path, default=ROOT / "tcga_luad_day0.csv")
    args = ap.parse_args()
    run(args.source, args.out, args.reference)
