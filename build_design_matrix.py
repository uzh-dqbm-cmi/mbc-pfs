"""Build the master design matrix by stitching modality-specific feature tables."""

# %%
from __future__ import annotations

import argparse
import json
import re
from typing import Optional

import numpy as np
import pandas as pd

from data_paths import data_csv, set_data_suffix
from design_matrix_pipeline import (
    assemble_design_matrix,
    build_all_modalities,
    create_context,
)
from qc import qc_from_received_flags


def adjust_age(design_matrix: pd.DataFrame) -> None:
    if "AGE" not in design_matrix.columns:
        return
    design_matrix["AGE"] = design_matrix["AGE"].astype(float) + np.floor_divide(
        design_matrix["LINE_START"], 365
    ).astype(int)


def order_columns(design_matrix: pd.DataFrame, features_dict: dict) -> pd.DataFrame:
    ordered: list[str] = [
        "PATIENT_ID",
        "LINE",
        "LINE_START",
        "PFS_TIME_DAYS",
        "PFS_EVENT",
    ]
    for modality in features_dict:
        ordered.extend(
            [col for col in features_dict[modality] if col in design_matrix.columns]
        )
    ordered_unique = [col for col in ordered if col in design_matrix.columns]
    remaining = [col for col in design_matrix.columns if col not in ordered_unique]
    return design_matrix[ordered_unique + remaining]


def _infer_option_from_suffix(data_suffix: Optional[str]) -> Optional[str]:
    if not data_suffix:
        return None
    match = re.search(r"option_([A-D])", data_suffix, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def build_design_matrix_pipeline(
    data_suffix: Optional[str] = None, option: Optional[str] = None
) -> pd.DataFrame:
    """Construct and persist the design matrix, returning the in-memory frame."""
    suffix = (data_suffix or "").strip()
    effective_option = (option or _infer_option_from_suffix(suffix) or "").upper()
    effective_option = effective_option or None
    set_data_suffix(suffix)
    suffix_arg: Optional[str] = suffix or None
    ctx = create_context(data_suffix=suffix_arg, option=effective_option)
    modality_frames = build_all_modalities(ctx)
    design_matrix, features_dict = assemble_design_matrix(ctx, modality_frames)

    adjust_age(design_matrix)
    pfs_event_before_qc = design_matrix["PFS_EVENT"].copy()
    design_matrix = qc_from_received_flags(design_matrix, verbose=True)
    pre_qc_ineligible = pfs_event_before_qc == -1
    qc_newly_ineligible = (pfs_event_before_qc != -1) & (
        design_matrix["PFS_EVENT"] == -1
    )
    if pre_qc_ineligible.any():
        print(
            "PFS_EVENT=-1 before QC:"
            f" {int(pre_qc_ineligible.sum()):,} LoTs across"
            f" {design_matrix.loc[pre_qc_ineligible, 'PATIENT_ID'].nunique():,} patients"
        )
    if qc_newly_ineligible.any():
        print(
            "QC marked additional PFS_EVENT=-1:"
            f" {int(qc_newly_ineligible.sum()):,} LoTs across"
            f" {design_matrix.loc[qc_newly_ineligible, 'PATIENT_ID'].nunique():,} patients"
        )
    design_matrix = order_columns(design_matrix, features_dict)

    design_matrix_path = data_csv("design_matrix", suffix_arg)

    print(
        "Design matrix before eligibility filters:"
        f" {design_matrix.shape} across {design_matrix['PATIENT_ID'].nunique():,} patients"
    )

    drop_mask = design_matrix["PFS_EVENT"] == -1
    dropped_lines = design_matrix.loc[drop_mask].copy().reset_index(drop=True)
    dropped_lines.to_csv("data/dropped_lines.csv", index=False)

    patients_before = set(design_matrix["PATIENT_ID"].unique())
    kept_patients = set(design_matrix.loc[~drop_mask, "PATIENT_ID"].unique())
    removed_patients = patients_before - kept_patients
    print(
        "Dropping"
        f" {len(dropped_lines):,} ineligible LoTs (PFS_EVENT=-1) across"
        f" {dropped_lines['PATIENT_ID'].nunique():,} patients"
        f" (removes {len(removed_patients):,} patients entirely)"
    )
    if not dropped_lines.empty and "QC_HARD_FAIL" in dropped_lines.columns:
        qc_mask = dropped_lines["QC_HARD_FAIL"].fillna(False).astype(bool)
        qc_lines = int(qc_mask.sum())
        qc_patients = dropped_lines.loc[qc_mask, "PATIENT_ID"].nunique()
        other_lines = int((~qc_mask).sum())
        other_patients = dropped_lines.loc[~qc_mask, "PATIENT_ID"].nunique()
        print(f"  QC hard-fail LoTs: {qc_lines:,} across {qc_patients:,} patients")
        print(
            f"  Other ineligible LoTs: {other_lines:,} across {other_patients:,} patients"
        )

    # DROP BEFORE DOWNSTREAM: LINES with PFS_EVENT = -1 and TREATMENT COLS
    design_matrix = (
        design_matrix[design_matrix["PFS_EVENT"] != -1].copy().reset_index(drop=True)
    )
    print(
        "Remaining after dropping PFS_EVENT=-1:"
        f" {design_matrix.shape} across {design_matrix['PATIENT_ID'].nunique():,} patients"
    )
    # Retain only agent-level features
    treatment_cols = [
        c
        for c in design_matrix.columns
        if c.startswith("PLANNED_TREATMENT_") or c.startswith("RECEIVED_TREATMENT_")
    ]
    design_matrix = design_matrix.drop(columns=treatment_cols)
    # also delete treatment_cols from features_dict
    if "TREATMENT" in features_dict:
        features_dict["TREATMENT"] = [
            c for c in features_dict["TREATMENT"] if c not in treatment_cols
        ]
    # save features dict directly
    with open("data/features_dict.json", "w", encoding="utf-8") as handle:
        json.dump(features_dict, handle, indent=2)
    bool_cols = [c for c in design_matrix.columns if design_matrix[c].dtype == "bool"]
    # typecast all boolean columns to Int8
    print(f"Converting {len(bool_cols)} boolean columns to Int8")
    for col in bool_cols:
        design_matrix[col] = design_matrix[col].astype("Int8")

    round_decimals = 6
    near_zero_atol = 0.5 * 10 ** (-round_decimals)
    float_cols = [
        c
        for c in design_matrix.columns
        if pd.api.types.is_float_dtype(design_matrix[c])
    ]
    replaced = 0
    for col in float_cols:
        values = design_matrix[col].astype("float64").to_numpy()
        mask = np.isclose(
            values,
            0.0,
            atol=near_zero_atol,
            rtol=0.0,
            equal_nan=False,
        )
        if mask.any():
            to_change = mask & ((values != 0.0) | np.signbit(values))
            replaced += int(to_change.sum())
            design_matrix[col] = design_matrix[col].mask(mask, 0)
    if replaced:
        print(
            f"Setting {replaced} near-zero float values (|x| <= {near_zero_atol:g}) to 0 before rounding"
        )

    print(
        f"Rounding {len(float_cols)} float columns to {round_decimals} decimal places"
    )
    for col in float_cols:
        design_matrix[col] = design_matrix[col].round(round_decimals)
    # # drop items with HAS_RADIOLOGY_REPORT_90_PRIOR
    # no_radiology_mask = design_matrix["HAS_RADIOLOGY_REPORT_90_PRIOR"] == 0
    # dropped_no_radiology = design_matrix.loc[no_radiology_mask].copy()
    # if not dropped_no_radiology.empty:
    #     dropped_no_radiology.to_csv(
    #         "data/dropped_lines_no_radiology_prior.csv", index=False
    #     )
    # patients_before_radiology = set(design_matrix["PATIENT_ID"].unique())
    # design_matrix = design_matrix.loc[~no_radiology_mask].reset_index(drop=True)
    # patients_after_radiology = set(design_matrix["PATIENT_ID"].unique())
    # removed_patients_radiology = patients_before_radiology - patients_after_radiology
    # print(
    #     "Dropping"
    #     f" {int(no_radiology_mask.sum()):,} LoTs with no radiology report within 90 days prior"
    #     f" (affects {dropped_no_radiology['PATIENT_ID'].nunique():,} patients;"
    #     f" removes {len(removed_patients_radiology):,} patients entirely)"
    # )
    # design_matrix.drop(columns=["HAS_RADIOLOGY_REPORT_90_PRIOR"], inplace=True)
    # print(
    #     "Remaining after radiology lookback filter:"
    #     f" {design_matrix.shape} across {design_matrix['PATIENT_ID'].nunique():,} patients"
    # )
    # just in case not ignored
    design_matrix.drop(
        columns=["ORIGINAL_PFS_TIME_DAYS", "ORIGINAL_PFS_EVENT"],
        inplace=True,
        errors="ignore",
    )
    design_matrix.to_csv(design_matrix_path, index=False)
    return design_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble the design matrix, optionally using a suffixed artifact set."
    )
    parser.add_argument(
        "--data-suffix",
        default="",
        help="Suffix appended to BREAST_treatment/pfs and design_matrix outputs.",
    )
    parser.add_argument(
        "--option",
        choices=["A", "B", "C", "D"],
        help="Option letter (used for option-specific column handling).",
    )
    args = parser.parse_args()
    build_design_matrix_pipeline(
        data_suffix=args.data_suffix or "", option=args.option or None
    )
