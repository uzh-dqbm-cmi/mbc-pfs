from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    MARKER_LONG_WINDOW_DAYS,
    MARKER_SHORT_WINDOW_DAYS,
    MARKER_TAU_LONG,
    MARKER_TAU_SHORT,
    RADIOLOGY_REPORT_WINDOW_DAYS,
)
from design_matrix_pipeline.modalities.cancer_presence import (
    aggregate_cancer_presence_features,
)
from design_matrix_pipeline.modalities.pdl1_mmr import make_tri_state_from_last_prior
from design_matrix_pipeline.modalities.tumor_markers import (
    _impute_tumor_marker_features,
    aggregate_biomarkers,
)

from . import mapping as M
from . import paths

FrameAndMappings = Tuple[pd.DataFrame, List[M.FeatureMapping]]


@dataclass
class FeatureContext:
    labels: pd.DataFrame 
    cancer: pd.DataFrame
    patient: pd.DataFrame
    imaging: pd.DataFrame
    labtest: pd.DataFrame
    pathology: pd.DataFrame
    panel: pd.DataFrame
    mutations: pd.DataFrame
    cna: pd.DataFrame 
    groups: Dict[str, List[str]]

    @property
    def keys(self) -> pd.DataFrame:
        return self.labels[["PATIENT_ID", "LINE"]].copy()

    @property
    def lot_info(self) -> pd.DataFrame:
        return self.labels[["PATIENT_ID", "LINE", "LINE_START"]].copy()

    def cancer_per_line(self, cols: List[str]) -> pd.DataFrame:
        """Attach per-cancer ``cols`` to each line via (PATIENT_ID, CA_SEQ)."""
        right = self.cancer.rename(columns={"record_id": "PATIENT_ID", "ca_seq": "CA_SEQ"})
        right = right[["PATIENT_ID", "CA_SEQ", *cols]].copy()
        right["PATIENT_ID"] = right["PATIENT_ID"].astype(str)
        right["CA_SEQ"] = pd.to_numeric(right["CA_SEQ"], errors="coerce").astype("Int64")
        left = self.labels[["PATIENT_ID", "CA_SEQ", "LINE", "LINE_START"]].copy()
        return left.merge(right, on=["PATIENT_ID", "CA_SEQ"], how="left")


def build_clinical(ctx: FeatureContext) -> FrameAndMappings:
    frame = ctx.cancer_per_line(
        ["dob_ca_dx_days", "ca_bca_er", "ca_bca_pr", "ca_bca_her_summ"]
    )

    age = (pd.to_numeric(frame["dob_ca_dx_days"], errors="coerce")
           + frame["LINE_START"]) / 365.25
    frame["AGE"] = age.astype(float)

    er_pos = frame["ca_bca_er"].map(M.is_positive)
    pr_pos = frame["ca_bca_pr"].map(M.is_positive)
    frame["HR"] = (er_pos | pr_pos).astype("int8")
    frame["HER2"] = frame["ca_bca_her_summ"].map(M.is_positive).astype("int8")

    sex = ctx.patient.rename(columns={"record_id": "PATIENT_ID"})[
        ["PATIENT_ID", "naaccr_sex_code"]
    ].copy()
    sex["PATIENT_ID"] = sex["PATIENT_ID"].astype(str)
    sex["GENDER_IS_FEMALE"] = sex["naaccr_sex_code"].eq("Female").astype("int8")
    frame = frame.merge(
        sex[["PATIENT_ID", "GENDER_IS_FEMALE"]], on="PATIENT_ID", how="left"
    )
    frame["GENDER_IS_FEMALE"] = frame["GENDER_IS_FEMALE"].fillna(0).astype("int8")

    out = frame[["PATIENT_ID", "LINE", "GENDER_IS_FEMALE", "AGE", "HR", "HER2"]].copy()
    mappings = [
        M.FeatureMapping(
            "GENDER_IS_FEMALE", "CLINICAL", "patient_level_dataset.csv",
            "naaccr_sex_code", "1 if naaccr_sex_code == 'Female' else 0", "mapped",
        ),
        M.FeatureMapping(
            "AGE", "CLINICAL", "cancer_level_dataset_index.csv + regimen line start",
            "dob_ca_dx_days, dx_reg_start_int",
            "(dob_ca_dx_days + LINE_START) / 365.25 = age in years at line start "
            "(MSK uses age at sequencing instead)", "mapped",
        ),
        M.FeatureMapping(
            "HR", "CLINICAL", "cancer_level_dataset_index.csv", "ca_bca_er, ca_bca_pr",
            "1 if ER or PR result contains 'Positive' else 0", "mapped",
        ),
        M.FeatureMapping(
            "HER2", "CLINICAL", "cancer_level_dataset_index.csv", "ca_bca_her_summ",
            "1 if HER2 summary contains 'Positive' (amplified) else 0", "mapped",
        ),
    ]
    return out, mappings


def build_diagnosis(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["DIAGNOSIS"]
    frame = ctx.cancer_per_line(
        ["stage_dx_iv", "naaccr_clin_stage_cd", "naaccr_path_stage_cd",
         "naaccr_seer_sum_stage", "ca_histology", "ca_hist_brca", "ca_d_site"]
    )
    out = ctx.keys
    for col in cols:
        out[col] = 0
    out = out.astype({c: "int8" for c in cols})

    out["STAGE_CDM_DERIVED_IV"] = frame["stage_dx_iv"].eq("Stage IV").astype("int8").values

    def _set_onehot(prefix: str, categories: pd.Series) -> None:
        for idx, cat in categories.items():
            colname = f"{prefix}{cat}"
            if colname in out.columns:
                out.iat[idx, out.columns.get_loc(colname)] = 1

    _set_onehot("CLINICAL_GROUP_", frame["naaccr_clin_stage_cd"].map(M.collapse_stage_digit))
    _set_onehot("PATH_GROUP_", frame["naaccr_path_stage_cd"].map(M.collapse_stage_digit))
    _set_onehot("SUMMARY_", frame["naaccr_seer_sum_stage"].map(M.seer_summary_category))

    hist = [
        M.histologic_category(h, hb)
        for h, hb in zip(frame["ca_histology"], frame["ca_hist_brca"])
    ]
    for idx, colname in enumerate(hist):
        if colname in out.columns:
            out.iat[idx, out.columns.get_loc(colname)] = 1

    subsite = frame["ca_d_site"].map(M.site_subsite_category)
    for idx, colname in subsite.items():
        if colname in out.columns:
            out.iat[idx, out.columns.get_loc(colname)] = 1

    src = "cancer_level_dataset_index.csv"
    mappings = [
        M.FeatureMapping("STAGE_CDM_DERIVED_IV", "DIAGNOSIS", src, "stage_dx_iv",
                         "1 if derived stage == 'Stage IV'", "mapped"),
    ]
    mappings += [
        M.FeatureMapping(c, "DIAGNOSIS", src, "naaccr_clin_stage_cd",
                         "one-hot of first digit of NAACCR clinical group stage", "mapped")
        for c in cols if c.startswith("CLINICAL_GROUP_")
    ]
    mappings += [
        M.FeatureMapping(c, "DIAGNOSIS", src, "naaccr_path_stage_cd",
                         "one-hot of first digit of NAACCR pathologic group stage", "mapped")
        for c in cols if c.startswith("PATH_GROUP_")
    ]
    mappings += [
        M.FeatureMapping(c, "DIAGNOSIS", src, "naaccr_seer_sum_stage",
                         "one-hot SEER summary stage (0=in situ,1=localized,"
                         "2-5=regional,7=distant,else unknown)", "mapped")
        for c in cols if c.startswith("SUMMARY_")
    ]
    mappings += [
        M.FeatureMapping(c, "DIAGNOSIS", src, "ca_histology, ca_hist_brca",
                         "one-hot of duct/lobular keyword bucketing", "mapped")
        for c in cols if c.startswith("HISTOLOGIC_")
    ]
    mappings += [
        M.FeatureMapping(c, "DIAGNOSIS", src, "ca_d_site",
                         "one-hot ICD-O topography (C50.9->NOS, C50.8->overlapping, "
                         "C50.4->UOQ, else OTHER)", "mapped")
        for c in cols if c.startswith("CANCER_SITE_SUBSITE_")
    ]
    return out, mappings


def build_tumor_markers(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["TUMOR_MARKERS"]
    lab = ctx.labtest.copy()
    lab = lab[lab["TEST"].astype(str).str.strip() == "CA15-3"]
    markers = pd.DataFrame(
        {
            "PATIENT_ID": lab["PATIENT_ID"].astype(str),
            "START_DATE": pd.to_numeric(lab["START_DATE"], errors="coerce"),
            "TEST": "CA15-3",
            "RESULT": pd.to_numeric(lab["RESULT"], errors="coerce"),
        }
    ).dropna(subset=["PATIENT_ID", "START_DATE", "RESULT"])
    markers["START_DATE"] = markers["START_DATE"].astype(int)

    test_map = {"CA15_3": "CA15-3", "CEA": "__GENIE_HAS_NO_CEA__"}
    features = aggregate_biomarkers(
        tumor_markers=markers,
        patient_lot_info=ctx.lot_info,
        test_map=test_map,
        short_window=MARKER_SHORT_WINDOW_DAYS,
        long_window=MARKER_LONG_WINDOW_DAYS,
        tau_short=MARKER_TAU_SHORT,
        tau_long=MARKER_TAU_LONG,
    )
    features = _impute_tumor_marker_features(features)
    features = features.drop(columns=[c for c in features.columns if c.endswith("_COUNT")])
    out = ctx.keys.merge(features, on=["PATIENT_ID", "LINE"], how="left")
    out = out.reindex(columns=["PATIENT_ID", "LINE", *cols])

    mappings = []
    for c in cols:
        if c.startswith("CA15_3"):
            mappings.append(M.FeatureMapping(
                c, "TUMOR_MARKERS", "data_timeline_labtest.txt", "TEST=='CA15-3', RESULT, START_DATE",
                "MSK recency-weighted marker kinetics (short=60d/long=180d windows) "
                "computed on prior CA15-3 results; cutoff 30 U/mL", "mapped"))
        else:
            mappings.append(M.FeatureMapping(
                c, "TUMOR_MARKERS", "(none)", "(none)",
                "CEA is not measured in GENIE -> empty (MISSING=1, kinetics NaN)", "empty"))
    return out, mappings

def build_ecog(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["ECOG"]
    out = ctx.keys
    for c in cols:
        if c == "ECOG_MISSING":
            out[c] = np.int8(1)
        elif c == "ECOG_LAST_OBS_DAY":
            out[c] = np.nan
        else:
            out[c] = np.int8(0)
    mappings = [
        M.FeatureMapping(c, "ECOG", "(none)", "(none)",
                         "ECOG performance status not available in GENIE -> empty "
                         "(ECOG_MISSING=1)", "empty")
        for c in cols
    ]
    return out, mappings

def build_pdl1(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["PDL1"]
    pth = ctx.pathology.copy()
    tested = pth[pth["PDL1_TESTING"].astype(str).str.strip().eq("Yes")].copy()
    events = pd.DataFrame(
        {
            "PATIENT_ID": tested["PATIENT_ID"].astype(str),
            "START_DATE": pd.to_numeric(tested["START_DATE"], errors="coerce"),
            "PDL1_POSITIVE": tested["PDL1_POSITIVE_ANY"].map(M.is_positive).astype(int),
        }
    ).dropna(subset=["PATIENT_ID", "START_DATE"])
    tri = make_tri_state_from_last_prior(
        events_df=events,
        patient_lot_info=ctx.lot_info,
        value_col="PDL1_POSITIVE",
        pos_name="PDL1_POS",
        neg_name="PDL1_NEG",
        unk_name="PDL1_UNKNOWN",
    )
    out = ctx.keys.merge(tri, on=["PATIENT_ID", "LINE"], how="left")
    out = out.reindex(columns=["PATIENT_ID", "LINE", *cols])
    for c in cols:
        default = 1 if c == "PDL1_UNKNOWN" else 0
        out[c] = out[c].fillna(default).astype("int8")
    mappings = [
        M.FeatureMapping(c, "PDL1", "data_timeline_pathology.txt",
                         "PDL1_TESTING, PDL1_POSITIVE_ANY, START_DATE",
                         "tri-state from latest PD-L1 result before line start "
                         "(default UNKNOWN if untested)", "mapped")
        for c in cols
    ]
    return out, mappings


def build_mmr(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["MMR"]
    out = ctx.keys
    for c in cols:
        out[c] = np.int8(1 if c == "MMR_UNKNOWN" else 0)
    mappings = [
        M.FeatureMapping(c, "MMR", "(none)", "(none)",
                         "MMR/MSI status not available in GENIE -> empty "
                         "(MMR_UNKNOWN=1)", "empty")
        for c in cols
    ]
    return out, mappings

def build_treatment(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["TREATMENT"]
    planned_cols = {c for c in cols if c.startswith("PLANNED_AGENT_")}
    received_cols = {c for c in cols if c.startswith("RECEIVED_AGENT_")}

    out = ctx.keys
    for c in cols:
        out[c] = 0
    out = out.astype({c: "int8" for c in cols})
    for idx, drug_str in enumerate(ctx.labels["REGIMEN_DRUGS"].fillna("")):
        suffixes = M.line_agent_suffixes(drug_str) or set()
        for suffix in suffixes:
            p = f"PLANNED_AGENT_{suffix}"
            r = f"RECEIVED_AGENT_{suffix}"
            if p in planned_cols:
                out.iat[idx, out.columns.get_loc(p)] = 1
            if r in received_cols:
                out.iat[idx, out.columns.get_loc(r)] = 1

    if "IS_MLOT1" in cols:
        out["IS_MLOT1"] = ctx.labels["LINE"].eq(1).astype("int8").values

    mappings = []
    for c in cols:
        if c == "IS_MLOT1":
            mappings.append(M.FeatureMapping(
                c, "TREATMENT", "regimen line index", "LINE",
                "1 if this is the patient's first qualifying line", "mapped"))
        elif c.startswith(("PLANNED_AGENT_", "RECEIVED_AGENT_")):
            mappings.append(M.FeatureMapping(
                c, "TREATMENT", "regimen_cancer_level_dataset.csv", "regimen_drugs",
                "1 if a drug in this line maps to this MSK agent column via the "
                "GENIE_DRUG_TO_AGENT dictionary", "mapped"))
        else:
            mappings.append(M.FeatureMapping(c, "TREATMENT", "regimen_cancer_level_dataset.csv",
                                             "regimen_drugs", "treatment-class flag", "mapped"))
    return out, mappings

def build_local_treatment(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["LOCAL_TREATMENT"]
    out = ctx.keys
    for c in cols:
        out[c] = np.int8(0)
    mappings = [
        M.FeatureMapping(c, "LOCAL_TREATMENT", "(none)", "(none)",
                         "No per-line surgery/radiation timeline mapped from GENIE -> "
                         "0 (MSK 'no event in window' value)", "empty")
        for c in cols
    ]
    return out, mappings


def build_cancer_presence(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["CANCER_PRESENCE"]
    body_parts = M.MSK_BODY_PARTS
    img = ctx.imaging.copy()
    img["START_DATE"] = pd.to_numeric(img["START_DATE"], errors="coerce")
    img = img.dropna(subset=["PATIENT_ID", "START_DATE"])
    img["PATIENT_ID"] = img["PATIENT_ID"].astype(str)
    img["START_DATE"] = img["START_DATE"].astype(int)

    parts = img["SCAN_SITES"].map(M.scan_sites_to_parts)
    measurements = pd.DataFrame({
        "PATIENT_ID": img["PATIENT_ID"].values,
        "START_DATE": img["START_DATE"].values,
        "HAS_CANCER": img["CANCER_STATUS"].map(M.cancer_status_bucket).values,
    })
    for part in body_parts:
        measurements[part] = parts.map(lambda s, p=part: 1 if p in s else 0).astype(int).values
    # drop scans that imaged nothing recognizable
    measurements = measurements[measurements[body_parts].sum(axis=1) > 0].reset_index(drop=True)

    result = aggregate_cancer_presence_features(
        measurements=measurements,
        patient_lot_info=ctx.lot_info,
        body_parts=body_parts,
        window_days=RADIOLOGY_REPORT_WINDOW_DAYS,
    )
    for col in result.columns:
        if col.endswith("_STATUS"):
            result[col] = result[col].fillna(0).astype("int8")
        elif col.endswith("_MISSING") or col.endswith("_EVER"):
            result[col] = result[col].fillna(1).astype("int8")
    out = ctx.keys.merge(result, on=["PATIENT_ID", "LINE"], how="left")
    out = out.reindex(columns=["PATIENT_ID", "LINE", *cols])

    mappings = [
        M.FeatureMapping(c, "CANCER_PRESENCE", "data_timeline_imaging.txt",
                         "SCAN_SITES, CANCER_STATUS, START_DATE",
                         "latest pre-line scan status per body part within 90d "
                         "(scan-level cancer status attributed to each imaged region)",
                         "mapped")
        for c in cols
    ]
    return out, mappings

def build_tumor_sites(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["TUMOR_SITES"]
    flag_cols = list(M.DIST_METS_TO_TUMOR_SITE.keys())
    timing_cols = [f"dx_to_{g}_days" for g in flag_cols]
    needed = [c for c in (flag_cols + timing_cols + ["dx_to_dmets_days", "stage_dx_iv"])
              if c in ctx.cancer.columns]
    frame = ctx.cancer_per_line(needed)

    line_start = frame["LINE_START"].to_numpy(dtype=float)
    dmets_overall = pd.to_numeric(frame.get("dx_to_dmets_days"), errors="coerce").to_numpy(dtype=float)
    is_iv = frame.get("stage_dx_iv", pd.Series(index=frame.index)).eq("Stage IV").to_numpy()

    out = ctx.keys
    window = RADIOLOGY_REPORT_WINDOW_DAYS
    for site in M.MSK_TUMOR_SITES:
        contrib = [g for g in flag_cols if site in M.DIST_METS_TO_TUMOR_SITE[g]]
        present = np.zeros(len(frame), dtype=bool)
        onset = np.full(len(frame), np.nan)
        for g in contrib:
            if g not in frame.columns:
                continue
            flag = pd.to_numeric(frame[g], errors="coerce").to_numpy() == 1
            present |= flag
            tcol = f"dx_to_{g}_days"
            t = pd.to_numeric(frame.get(tcol), errors="coerce").to_numpy(dtype=float)
            cand = np.where(flag, t, np.nan)
            onset = np.fmin(onset, cand)  # nan-aware min
        # fallbacks for a present-but-undated site
        need_fallback = present & np.isnan(onset)
        onset = np.where(need_fallback, dmets_overall, onset)
        still = present & np.isnan(onset) & is_iv
        onset = np.where(still, 0.0, onset)

        ever = present & ~np.isnan(onset) & (onset <= line_start)
        in_window = ever & (onset >= (line_start - window))
        out[f"TUMOR_SITE_{site}_EVER"] = ever.astype("int8")
        out[f"TUMOR_SITE_{site}_IN_WINDOW"] = in_window.astype("int8")

    out = out.reindex(columns=["PATIENT_ID", "LINE", *cols])
    for c in cols:
        out[c] = out[c].fillna(0).astype("int8")
    mappings = [
        M.FeatureMapping(c, "TUMOR_SITES", "cancer_level_dataset_index.csv",
                         "dist_mets_*, dx_to_dist_mets_*_days",
                         "distant-met site flags mapped to MSK sites; EVER if onset "
                         "<= line start, IN_WINDOW if onset within 90d before line",
                         "mapped")
        for c in cols
    ]
    return out, mappings

def build_genomics(ctx: FeatureContext) -> FrameAndMappings:
    cols = ctx.groups["GENOMICS"]
    mut_genes = [c[len("GENOMICS_MUTATION_"):] for c in cols if c.startswith("GENOMICS_MUTATION_")]
    amp_genes = [c[len("GENOMICS_AMPLIFICATION_"):] for c in cols if c.startswith("GENOMICS_AMPLIFICATION_")]
    del_genes = [c[len("GENOMICS_DELETION_"):] for c in cols if c.startswith("GENOMICS_DELETION_")]

    # one representative panel sample per patient: earliest NGS report.
    panel = ctx.panel.copy()
    panel["record_id"] = panel["record_id"].astype(str)
    panel["dx_cpt_rep_days"] = pd.to_numeric(panel["dx_cpt_rep_days"], errors="coerce")
    panel = panel.sort_values(["record_id", "dx_cpt_rep_days"], kind="mergesort")
    rep = panel.drop_duplicates("record_id", keep="first")
    patient_to_sample = dict(zip(rep["record_id"], rep["cpt_genie_sample_id"].astype(str)))
    # NGS report day (days from diagnosis) of that sample, for no-lookahead gating
    patient_to_seqday = dict(zip(rep["record_id"], rep["dx_cpt_rep_days"]))

    mut = ctx.mutations
    mut_by_sample = mut.groupby("Tumor_Sample_Barcode")["Hugo_Symbol"].agg(set).to_dict()
    cna = ctx.cna  # index Hugo_Symbol, columns samples
    cna_samples = set(cna.columns)

    rows = []
    for pid, sample in patient_to_sample.items():
        row = {"PATIENT_ID": pid, "GENOMICS_MISSING": 0}
        muts = mut_by_sample.get(sample, set())
        for gene in mut_genes:
            row[f"GENOMICS_MUTATION_{gene}"] = int(gene in muts)
        in_cna = sample in cna_samples
        for gene in amp_genes:
            v = cna.at[gene, sample] if (in_cna and gene in cna.index) else np.nan
            row[f"GENOMICS_AMPLIFICATION_{gene}"] = int(pd.notna(v) and float(v) == 2.0)
        for gene in del_genes:
            v = cna.at[gene, sample] if (in_cna and gene in cna.index) else np.nan
            row[f"GENOMICS_DELETION_{gene}"] = int(pd.notna(v) and float(v) == -2.0)
        rows.append(row)
    genomics = pd.DataFrame(rows)

    out = ctx.keys.merge(genomics, on="PATIENT_ID", how="left")
    out = out.merge(
        ctx.labels[["PATIENT_ID", "LINE", "LINE_START"]], on=["PATIENT_ID", "LINE"], how="left"
    )
    seq_day = pd.to_numeric(out["PATIENT_ID"].map(patient_to_seqday), errors="coerce")
    available = seq_day.notna() & (seq_day <= out["LINE_START"])

    gene_cols = [c for c in cols if c != "GENOMICS_MISSING"]
    for c in gene_cols:
        out[c] = out[c].fillna(0)
    out.loc[~available, gene_cols] = 0
    out["GENOMICS_MISSING"] = (~available).astype("int8")
    for c in gene_cols:
        out[c] = out[c].astype("int8")
    out = out.reindex(columns=["PATIENT_ID", "LINE", *cols])

    mappings = []
    for c in cols:
        if c == "GENOMICS_MISSING":
            mappings.append(M.FeatureMapping(
                c, "GENOMICS", "cancer_panel_test_level_dataset.csv",
                "cpt_genie_sample_id, dx_cpt_rep_days",
                "1 if no sequenced panel, or the panel's NGS report day is after the "
                "line start (strict no-lookahead, matching MSK)", "mapped"))
        elif c.startswith("GENOMICS_MUTATION_"):
            mappings.append(M.FeatureMapping(
                c, "GENOMICS", "data_mutations_extended.txt", "Hugo_Symbol, Tumor_Sample_Barcode",
                "1 if gene is mutated in the patient's representative panel sample", "mapped"))
        else:
            kind = "amplified (CNA==2)" if c.startswith("GENOMICS_AMPLIFICATION_") else "deep-deleted (CNA==-2)"
            mappings.append(M.FeatureMapping(
                c, "GENOMICS", "data_CNA.txt", "Hugo_Symbol x sample matrix",
                f"1 if gene is {kind} in the representative panel sample", "mapped"))
    return out, mappings


BUILDERS = [
    build_cancer_presence,
    build_tumor_sites,
    build_tumor_markers,
    build_ecog,
    build_pdl1,
    build_mmr,
    build_treatment,
    build_local_treatment,
    build_clinical,
    build_diagnosis,
    build_genomics,
]
