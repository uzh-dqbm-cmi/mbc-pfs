# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

import build_pfs as build_pfs_module
import config
from build_design_matrix import build_design_matrix_pipeline
from build_pfs import (
    assign_patient_lines_and_pfs,
    first_metastasis_start_dates,
    measurement_days,
    prep_death_times,
    prep_diagnosis,
)
from config import (
    ADMIN_CENSOR_DAYS,
    EVENT_COL,
    PATIENT_ID_COL,
    RADIOLOGY_REPORT_WINDOW_DAYS,
    TIME_COL,
)
from design_matrix_pipeline.modalities import cancer_presence, tumor_sites

CANCER_TYPE = "BREAST"
SPLIT_OPTION = "A"
DAY_SPLIT = 365
AFTER_TREATMENT_GRACE_PERIOD = 28
DATA_DIR = Path("data/msk_chord_2024")

# %%
config.RADIOLOGY_REPORT_WINDOW_DAYS = RADIOLOGY_REPORT_WINDOW_DAYS
build_pfs_module.RADIOLOGY_REPORT_WINDOW_DAYS = RADIOLOGY_REPORT_WINDOW_DAYS
tumor_sites.RADIOLOGY_REPORT_WINDOW_DAYS = RADIOLOGY_REPORT_WINDOW_DAYS
cancer_presence.RADIOLOGY_REPORT_WINDOW_DAYS = RADIOLOGY_REPORT_WINDOW_DAYS

PFS_PATH = Path(f"data/{CANCER_TYPE}_pfs.csv")
TREATMENT_OUTPUT_PATH = Path(f"data/{CANCER_TYPE}_treatment.csv")


# Step 1: filter to breast-only primary diagnoses.
diag, TYPE_SPECIFIC_PATIENT_IDS, STAGE_IV_PATIENTS = prep_diagnosis()
first_metastasis_by_patient = first_metastasis_start_dates(TYPE_SPECIFIC_PATIENT_IDS)

print(f"Step 1 — Breast-only patients: {len(TYPE_SPECIFIC_PATIENT_IDS):,}")

# Step 2: load treatment/progression timelines, drop Bone treatment, and keep patients with both data types.
from build_pfs import _norm

treatment = _norm(
    pd.read_csv("data/msk_chord_2024/data_timeline_treatment.tsv", sep="\t")
)
treatment = treatment[treatment[PATIENT_ID_COL].isin(TYPE_SPECIFIC_PATIENT_IDS)]
treatment = treatment[treatment["SUBTYPE"] != "Bone Treatment"]
treatment = treatment[treatment["AGENT"] != "INVESTIGATIVE"]
treatment = treatment.drop(columns=["RX_INVESTIGATIVE", "FLAG_OROTOPICAL"])
treatment = treatment.sort_values([PATIENT_ID_COL, "START_DATE"], kind="mergesort")

progression = _norm(
    pd.read_csv(DATA_DIR / "data_timeline_progression.tsv", sep="\t", dtype=str)
)
progression = progression[progression[PATIENT_ID_COL].isin(TYPE_SPECIFIC_PATIENT_IDS)]
progression_as_rad_evidence = progression.copy()
progression = progression[progression["PROGRESSION"].isin(["Y", "N"])]
progression = progression.sort_values([PATIENT_ID_COL, "START_DATE"], kind="mergesort")

print(
    f"  Treatment records: {len(treatment):,} across {treatment['PATIENT_ID'].nunique():,} patients"
)
print(
    f"  Progression records: {len(progression):,} across {progression['PATIENT_ID'].nunique():,} patients"
)

# sanity checks before proceeding
assert not progression["START_DATE"].isnull().any()
assert not treatment["START_DATE"].isnull().any()

# by each patient that's not stage 4 by diagnosis, only consider progression and treatment entries after first_metastasis_by_patient
progression = progression[
    progression.apply(
        lambda row: (row[PATIENT_ID_COL] in STAGE_IV_PATIENTS)
        or (
            row["START_DATE"]
            >= first_metastasis_by_patient.get(row[PATIENT_ID_COL], 100_000_000)
        ),
        axis=1,
    )
].reset_index(drop=True)

treatment = treatment[
    treatment.apply(
        lambda row: (row[PATIENT_ID_COL] in STAGE_IV_PATIENTS)
        or (
            row["START_DATE"]
            >= first_metastasis_by_patient.get(row[PATIENT_ID_COL], 100_000_000)
        ),
        axis=1,
    )
].reset_index(drop=True)

treatment_patients = set(treatment[PATIENT_ID_COL])
progression_patients = set(progression[PATIENT_ID_COL])

print(
    f"  Filtered mBC treatment records: {len(treatment):,} across {len(treatment_patients):,} patients"
)
print(
    f"  Filtered mBC progression records: {len(progression):,} across {len(progression_patients):,} patients"
)

TYPE_SPECIFIC_PATIENT_IDS = treatment_patients.intersection(progression_patients)

progression = progression[
    progression[PATIENT_ID_COL].isin(TYPE_SPECIFIC_PATIENT_IDS)
].reset_index(drop=True)
treatment = treatment[
    treatment[PATIENT_ID_COL].isin(TYPE_SPECIFIC_PATIENT_IDS)
].reset_index(drop=True)


death_times = prep_death_times(type_specific_patient_ids=TYPE_SPECIFIC_PATIENT_IDS)
patient_radiology_days = measurement_days(
    treatment_patients, progression_df=progression
)


print(f"Step 2 — mBC patients with both: {len(TYPE_SPECIFIC_PATIENT_IDS):,}")

progression_groups = {
    pid: grp[["START_DATE", "PROGRESSION"]].reset_index(drop=True)
    for pid, grp in progression.groupby(PATIENT_ID_COL, observed=True)
}

pfs_rows: List[Dict[str, int]] = []
for pid, patient_treatments in treatment.groupby(PATIENT_ID_COL, observed=True):
    patient_events = progression_groups.get(
        pid, pd.DataFrame(columns=["START_DATE", "PROGRESSION"])
    )
    patient_lines, patient_pfs = assign_patient_lines_and_pfs(
        patient_treatments=patient_treatments,
        patient_progression=patient_events,
        day_split=DAY_SPLIT,
        grace_period=AFTER_TREATMENT_GRACE_PERIOD,
        death_time=death_times.get(pid),
    )

    pfs_rows.extend(patient_pfs)

pfs = (
    pd.DataFrame(pfs_rows).sort_values([PATIENT_ID_COL, "LINE"]).reset_index(drop=True)
)


# %%
# only labelled_by_progression_event and labelled_by_death_event should have PFS_EVENT = 1
# only labelled_by_non_progression should have PFS_EVENT = 0
# all others should have PFS_EVENT = -1
expected_event_labels = {
    "labelled_by_progression_event": 1,
    "labelled_by_death_event": 1,
    "labelled_by_non_progression": 0,
}
for line_source, expected_event in expected_event_labels.items():
    mask = pfs["LINE_SOURCE"] == line_source
    assert all(
        pfs.loc[mask, EVENT_COL] == expected_event
    ), f"LINE_SOURCE {line_source} does not have expected PFS_EVENT = {expected_event}."
assert all(
    pfs.loc[~pfs["LINE_SOURCE"].isin(expected_event_labels.keys()), EVENT_COL] == -1
), "Some other LINE_SOURCE events do not have PFS_EVENT = -1."

# set PFS_TIME_DAYS < 28 to another label
mask_too_short_pfs_effectiveness = (pfs[TIME_COL] < 28) & (pfs[EVENT_COL] != -1)
if mask_too_short_pfs_effectiveness.any():
    pfs.loc[mask_too_short_pfs_effectiveness, "LINE_SOURCE"] = (
        "line_too_short_for_effectiveness"
    )
    pfs.loc[mask_too_short_pfs_effectiveness, EVENT_COL] = -1
    print(
        f"  Adjusted {mask_too_short_pfs_effectiveness.sum()} lines with PFS_TIME_DAYS < 28 to 'line_too_short_for_effectiveness'"
    )

print("PFS LINE_SOURCE value counts:")
print(pfs["LINE_SOURCE"].value_counts())
print(f"Total patients before filtering: {len(pfs['PATIENT_ID'].unique())}")
print(f"Total lines before filtering: {len(pfs):,}")


# # Step 3: apply administrative censoring and persist PFS table.
def apply_censor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    longer_mask = (df[TIME_COL] > ADMIN_CENSOR_DAYS) & (df[EVENT_COL] != -1)
    df.loc[longer_mask, TIME_COL] = ADMIN_CENSOR_DAYS
    df.loc[longer_mask, EVENT_COL] = 0
    exact_event_mask = (df[TIME_COL] == ADMIN_CENSOR_DAYS) & (df[EVENT_COL] == 1)
    df.loc[exact_event_mask, TIME_COL] = ADMIN_CENSOR_DAYS - 0.001
    return df


# typecast pfs TIME_COL to Float32
pfs[TIME_COL] = pfs[TIME_COL].astype("Float32")
pfs = apply_censor(pfs)
# save dropped pfs lines for record-keeping
dropped_pfs = pfs[pfs[EVENT_COL] == -1].reset_index(drop=True)
dropped_pfs.to_csv(Path("data") / f"{CANCER_TYPE}_dropped_pfs_lines.csv", index=False)

# save only patients with EVENT_COL != -1
pfs = pfs[pfs[EVENT_COL] != -1].reset_index(drop=True)
print(f"Total patients after filtering: {len(pfs['PATIENT_ID'].unique())}")
print(f"Total lines after filtering: {len(pfs):,}")
pfs.to_csv(PFS_PATH, index=False)

# Step 4: assemble the design matrix and persist it via build_design_matrix_pipeline.
design_matrix = build_design_matrix_pipeline(data_suffix=None, option=SPLIT_OPTION)
print(f"Design matrix shape: {design_matrix.shape}")
print(f"unique patients in design matrix: {design_matrix[PATIENT_ID_COL].nunique():,}")
print(f"Step 5 — Final usable lines in design matrix: {design_matrix.shape}")
print(f"Design matrix event distribution: {design_matrix[EVENT_COL].value_counts()}")
