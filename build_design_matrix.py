from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import LINE_START_WINDOW_DAYS
from data_paths import data_dir
from design_matrix_pipeline import (
    assemble_design_matrix,
    build_all_modalities,
    create_context,
)
from qc import qc_from_received_flags


def adjust_age(design_matrix: pd.DataFrame) -> None:
    if "AGE" not in design_matrix.columns:
        raise ValueError("Expected 'AGE' column in design matrix for age adjustment")
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


def build_design_matrix_pipeline(window_days: int = LINE_START_WINDOW_DAYS) -> pd.DataFrame:
    out_dir = data_dir(window_days)
    ctx = create_context(window_days)
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

    print(
        "Design matrix before eligibility filters:"
        f" {design_matrix.shape} across {design_matrix['PATIENT_ID'].nunique():,} patients"
    )

    drop_mask = design_matrix["PFS_EVENT"] == -1
    dropped_lines = design_matrix.loc[drop_mask].copy().reset_index(drop=True)
    dropped_lines.to_csv(out_dir / "dropped_lines.csv", index=False)

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

    design_matrix = (
        design_matrix[design_matrix["PFS_EVENT"] != -1].copy().reset_index(drop=True)
    )
    print(
        "Remaining after dropping PFS_EVENT=-1:"
        f" {design_matrix.shape} across {design_matrix['PATIENT_ID'].nunique():,} patients"
    )
    treatment_cols = [
        c
        for c in design_matrix.columns
        if c.startswith("PLANNED_TREATMENT_") or c.startswith("RECEIVED_TREATMENT_")
    ]
    design_matrix = design_matrix.drop(columns=treatment_cols)
    if "TREATMENT" in features_dict:
        features_dict["TREATMENT"] = [
            c for c in features_dict["TREATMENT"] if c not in treatment_cols
        ]
    bool_cols = [c for c in design_matrix.columns if design_matrix[c].dtype == "bool"]
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
    design_matrix.drop(
        columns=["ORIGINAL_PFS_TIME_DAYS", "ORIGINAL_PFS_EVENT"],
        inplace=True,
        errors="ignore",
    )
    design_matrix["IS_MLOT1"] = design_matrix["LINE"] == 1
    features_dict.setdefault("TREATMENT", [])
    if "IS_MLOT1" not in features_dict["TREATMENT"]:
        features_dict["TREATMENT"].append("IS_MLOT1")
    with open(out_dir / "features_dict.json", "w", encoding="utf-8") as handle:
        json.dump(features_dict, handle, indent=2)
    design_matrix.to_csv(out_dir / "design_matrix.csv", index=False)
    return design_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild the design matrix from BREAST_pfs.csv written by pipeline_overview.py."
    )
    parser.add_argument("--window-days", type=int, default=LINE_START_WINDOW_DAYS)
    build_design_matrix_pipeline(parser.parse_args().window_days)
