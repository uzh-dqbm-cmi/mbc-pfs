from __future__ import annotations

from typing import List

import pandas as pd

from config import EVENT_COL, TIME_COL
from . import mapping, paths


def _records_frame(mappings: List[mapping.FeatureMapping]) -> pd.DataFrame:
    order = {g: i for i, g in enumerate(paths.load_msk_feature_groups())}
    feature_order = {c: i for i, c in enumerate(paths.msk_feature_columns())}
    df = pd.DataFrame([m.__dict__ for m in mappings])
    df["_g"] = df["group"].map(order)
    df["_f"] = df["feature"].map(feature_order)
    return df.sort_values(["_g", "_f"]).drop(columns=["_g", "_f"]).reset_index(drop=True)


def write_audit(
    mappings: List[mapping.FeatureMapping],
    design: pd.DataFrame,
    label_df: pd.DataFrame,
) -> None:
    df = _records_frame(mappings)
    df.to_csv(paths.AUDIT_CSV_OUT, index=False)

    n_total = len(df)
    n_mapped = int((df["status"] == "mapped").sum())
    n_empty = int((df["status"] == "empty").sum())

    lines = []
    lines.append("# GENIE -> MSK feature mapping audit\n")
    lines.append(
        "This report is generated directly from the builders in "
        "`genie_validation/features.py`. Each row is one MSK design-matrix "
        "feature column and how it was reproduced from raw GENIE data.\n"
    )

    lines.append("## Cohort and labels\n")
    lines.append(f"- Patients (lines): **{design['PATIENT_ID'].nunique()}** "
                 f"(**{len(design)}**)")
    ev = design[EVENT_COL].value_counts().to_dict()
    lines.append(f"- PFS-I events: progression/death={ev.get(1, 0)}, censored={ev.get(0, 0)}")
    lines.append(f"- Median PFS time (days): {design[TIME_COL].median():.1f}")
    lines.append(
        "- Label source: `regimen_cancer_level_dataset.csv` "
        "(`pfs_i_g_status` / `tt_pfs_i_g_days`), gated to patients with a non-null "
        "`PFS_I_ADV_STATUS` in `data_clinical_supp_survival.txt`. No advanced-disease "
        "anchoring: a non-null `tt_pfs_i_g_days` already denotes a metastatic line.")
    lines.append("- Outcome cleaning mirrors MSK: drop lines < 28 days, "
                 "administrative censoring at 730 days.")
    lines.append(
        "- **Drug filter:** every drug in a line must resolve to an MSK agent in the "
        "`GENIE_DRUG_TO_AGENT` dictionary (`genie_validation/mapping.py`), which maps "
        "each GENIE drug to its MSK agent column (investigational drugs map to MSK's own "
        "`OTHER_INVESTIGATIVE` agent). A line containing *any* still-unknown drug "
        "(generic 'OTHER' buckets, agents MSK never recorded) is dropped wholesale.\n")

    lines.append("## Key derivation notes\n")
    lines.append(
        "- **AGE** is *age at line start* = `(dob_ca_dx_days + dx_reg_start_int) / 365.25`. "
        "This differs from MSK, which uses age at sequencing; the user requested age at "
        "the start of the line.")
    lines.append(
        "- **Genomics** uses one representative panel per patient (earliest NGS report) with "
        "strict no-lookahead time-gating, mirroring MSK: if the panel's NGS report day is "
        "missing or after the line start, all gene flags are 0 and `GENOMICS_MISSING` is 1.")
    lines.append(
        "- **Treatment** uses the explicit `GENIE_DRUG_TO_AGENT` dictionary (derived from "
        "MSK's own agent/class data): each drug maps to one MSK agent column "
        "(`PLANNED_AGENT_*` and its `RECEIVED_AGENT_*` twin). Lines with any drug outside "
        "the dictionary were already dropped, so no speculative class mapping remains.\n")

    lines.append("## Coverage summary\n")
    lines.append(f"- Total MSK feature columns: **{n_total}**")
    lines.append(f"- Mapped from GENIE: **{n_mapped}**")
    lines.append(f"- Left empty (no GENIE source): **{n_empty}**\n")

    grp = df.groupby("group", sort=False)["status"].value_counts().unstack(fill_value=0)
    lines.append("| Modality | mapped | empty |")
    lines.append("| --- | --- | --- |")
    for g in paths.load_msk_feature_groups():
        if g in grp.index:
            lines.append(f"| {g} | {int(grp.loc[g].get('mapped', 0))} "
                         f"| {int(grp.loc[g].get('empty', 0))} |")
    lines.append("")

    empties = sorted(df.loc[df["status"] == "empty", "group"].unique())
    if empties:
        lines.append(f"**Empty modalities/features:** {', '.join(empties)} "
                     "(CEA, ECOG, MMR, local treatment have no GENIE equivalent).\n")

    lines.append("## Per-feature mapping\n")
    for g in paths.load_msk_feature_groups():
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        lines.append(f"### {g}\n")
        lines.append("| MSK feature | GENIE source | GENIE columns | Transform | Status |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, r in sub.iterrows():
            feat = str(r["feature"]).replace("|", "\\|")
            tr = str(r["transform"]).replace("|", "\\|")
            lines.append(
                f"| `{feat}` | {r['genie_source']} | `{r['genie_columns']}` | {tr} | {r['status']} |"
            )
        lines.append("")

    with open(paths.AUDIT_MD_OUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"  wrote {paths.AUDIT_CSV_OUT} ({n_total} rows) and {paths.AUDIT_MD_OUT}")
