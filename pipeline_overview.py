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

import argparse
from typing import Dict, List, Tuple

import pandas as pd

from build_design_matrix import build_design_matrix_pipeline
from build_pfs import (
    _norm,
    after_metastasis,
    apply_censor,
    assign_patient_lines_and_pfs,
    first_metastasis_start_dates,
    flag_short_lines,
    measurement_days,
    prep_death_times,
    prep_diagnosis,
)
from config import (
    ADMIN_CENSOR_DAYS,
    CANCER_TYPE,
    DAY_SPLIT,
    EVENT_COL,
    LINE_START_WINDOW_DAYS,
    MSK_DATA_DIR,
    PATIENT_ID_COL,
    TIME_COL,
)
from data_paths import data_dir


def build_pfs_lines(window_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Step 1: filter to breast-only primary diagnoses.
    diag, patient_ids, stage_iv_patients = prep_diagnosis()
    first_metastasis = first_metastasis_start_dates(patient_ids)

    print(f"Step 1 — Breast-only patients: {len(patient_ids):,}")

    ref = pd.read_csv("data/brca_dx_1st_seq_OS.csv")
    assert patient_ids.issubset(ref[PATIENT_ID_COL].unique())

    # Step 2: load treatment/progression timelines, drop Bone treatment, and keep patients with both data types.
    treatment = _norm(pd.read_csv(MSK_DATA_DIR / "data_timeline_treatment.tsv", sep="\t"))
    treatment = treatment[treatment[PATIENT_ID_COL].isin(patient_ids)]
    treatment = treatment[treatment["SUBTYPE"] != "Bone Treatment"]
    treatment = treatment[treatment["AGENT"] != "INVESTIGATIVE"]
    treatment = treatment.drop(columns=["RX_INVESTIGATIVE", "FLAG_OROTOPICAL"])
    treatment = treatment.sort_values([PATIENT_ID_COL, "START_DATE"], kind="mergesort")

    progression = _norm(
        pd.read_csv(MSK_DATA_DIR / "data_timeline_progression.tsv", sep="\t", dtype=str)
    )
    progression = progression[progression[PATIENT_ID_COL].isin(patient_ids)]
    progression = progression[progression["PROGRESSION"].isin(["Y", "N"])]
    progression = progression.sort_values([PATIENT_ID_COL, "START_DATE"], kind="mergesort")

    print(
        f"  Treatment records: {len(treatment):,} across {treatment[PATIENT_ID_COL].nunique():,} patients"
    )
    print(
        f"  Progression records: {len(progression):,} across {progression[PATIENT_ID_COL].nunique():,} patients"
    )

    # sanity checks before proceeding
    assert not progression["START_DATE"].isnull().any()
    assert not treatment["START_DATE"].isnull().any()

    # by each patient that's not stage 4 by diagnosis, only consider progression and treatment entries after first_metastasis_by_patient
    progression = after_metastasis(progression, stage_iv_patients, first_metastasis)
    treatment = after_metastasis(treatment, stage_iv_patients, first_metastasis)

    treatment_patients = set(treatment[PATIENT_ID_COL])
    progression_patients = set(progression[PATIENT_ID_COL])

    print(
        f"  Filtered mBC treatment records: {len(treatment):,} across {len(treatment_patients):,} patients"
    )
    print(
        f"  Filtered mBC progression records: {len(progression):,} across {len(progression_patients):,} patients"
    )

    patient_ids = treatment_patients.intersection(progression_patients)
    progression = progression[progression[PATIENT_ID_COL].isin(patient_ids)].reset_index(
        drop=True
    )
    treatment = treatment[treatment[PATIENT_ID_COL].isin(patient_ids)].reset_index(
        drop=True
    )

    death_times = prep_death_times(type_specific_patient_ids=patient_ids)
    measurement_days(treatment_patients, progression_df=progression)

    print(f"Step 2 — mBC patients with both: {len(patient_ids):,}")

    progression_groups = {
        pid: grp[["START_DATE", "PROGRESSION"]].reset_index(drop=True)
        for pid, grp in progression.groupby(PATIENT_ID_COL, observed=True)
    }

    pfs_rows: List[Dict[str, int]] = []
    for pid, patient_treatments in treatment.groupby(PATIENT_ID_COL, observed=True):
        patient_events = progression_groups.get(
            pid, pd.DataFrame(columns=["START_DATE", "PROGRESSION"])
        )
        _, patient_pfs = assign_patient_lines_and_pfs(
            patient_treatments=patient_treatments,
            patient_progression=patient_events,
            day_split=DAY_SPLIT,
            grace_period=window_days,
            death_time=death_times.get(pid),
        )
        pfs_rows.extend(patient_pfs)

    pfs = pd.DataFrame(pfs_rows).sort_values([PATIENT_ID_COL, "LINE"]).reset_index(drop=True)
    return diag, pfs


def check_event_labels(pfs: pd.DataFrame) -> None:
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


def print_outcome_breakdown(df: pd.DataFrame, label: str) -> None:
    event_mask = df[EVENT_COL] == 1
    censor_mask = df[EVENT_COL] == 0
    progression_event_mask = event_mask & (
        df["LINE_SOURCE"] == "labelled_by_progression_event"
    )
    death_event_mask = event_mask & (df["LINE_SOURCE"] == "labelled_by_death_event")
    observed_non_progression_mask = censor_mask & (
        df["LINE_SOURCE"] == "labelled_by_non_progression"
    )
    admin_censor_mask = censor_mask & ~observed_non_progression_mask

    n_events = int(event_mask.sum())
    n_censored = int(censor_mask.sum())
    n_progression_events = int(progression_event_mask.sum())
    n_death_events = int(death_event_mask.sum())
    n_observed_non_progression = int(observed_non_progression_mask.sum())
    n_admin_censored = int(admin_censor_mask.sum())

    assert (
        n_progression_events + n_death_events == n_events
    ), "Final events should be fully explained by progression or death."
    assert (
        n_observed_non_progression + n_admin_censored == n_censored
    ), "Final censored lines should be fully explained by non-progression or admin censoring."

    print(f"{label} outcome breakdown across {len(df):,} lines:")
    print(
        "  Events:"
        f" {n_events:,} ({n_progression_events:,} labelled by progression,"
        f" {n_death_events:,} labelled by death)"
    )
    print(
        "  Censored:"
        f" {n_censored:,} ({n_observed_non_progression:,} labelled by non-progression,"
        f" {n_admin_censored:,} administratively censored at {ADMIN_CENSOR_DAYS:,} days)"
    )


def main(window_days: int = LINE_START_WINDOW_DAYS) -> None:
    out_dir = data_dir(window_days)
    out_dir.mkdir(parents=True, exist_ok=True)

    diag, pfs = build_pfs_lines(window_days)
    diag.to_csv(out_dir / "diagnosis_parsed.csv", index=False)
    check_event_labels(pfs)
    # set PFS_TIME_DAYS < 28 to another label
    pfs = flag_short_lines(pfs)
    print("PFS LINE_SOURCE value counts:")
    print(pfs["LINE_SOURCE"].value_counts())
    print(f"Total patients before filtering: {pfs[PATIENT_ID_COL].nunique()}")
    print(f"Total lines before filtering: {len(pfs):,}")

    # # Step 3: apply administrative censoring and persist PFS table.
    # typecast pfs TIME_COL to Float32
    pfs[TIME_COL] = pfs[TIME_COL].astype("Float32")
    pfs = apply_censor(pfs)
    # save dropped pfs lines for record-keeping
    pfs[pfs[EVENT_COL] == -1].reset_index(drop=True).to_csv(
        out_dir / f"{CANCER_TYPE}_dropped_pfs_lines.csv", index=False
    )
    # save only patients with EVENT_COL != -1
    pfs = pfs[pfs[EVENT_COL] != -1].reset_index(drop=True)
    print(f"Total patients after filtering: {pfs[PATIENT_ID_COL].nunique()}")
    print(f"Total lines after filtering: {len(pfs):,}")
    pfs.to_csv(out_dir / f"{CANCER_TYPE}_pfs.csv", index=False)

    # Step 4: assemble the design matrix and persist it via build_design_matrix_pipeline.
    design_matrix = build_design_matrix_pipeline(window_days)
    print(f"Final usable lines in design matrix: {design_matrix.shape}")
    print(f"unique patients in design matrix: {design_matrix[PATIENT_ID_COL].nunique():,}")
    print(f"Design matrix event distribution: {design_matrix[EVENT_COL].value_counts()}")
    print_outcome_breakdown(design_matrix, label="Design matrix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign lines of therapy, label PFS and build the design matrix."
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=LINE_START_WINDOW_DAYS,
        help="Treatment window after line start (default 28; other values write to data/window_<N>d/).",
    )
    main(parser.parse_args().window_days)
