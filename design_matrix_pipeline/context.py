from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data_paths import data_csv, get_data_suffix
from config import (
    MARKER_LONG_WINDOW_DAYS,
    MARKER_SHORT_WINDOW_DAYS,
    MARKER_TAU_LONG,
    MARKER_TAU_SHORT,
    ECOG_LOOKBACK_DAYS,
    ECOG_EVER_GE2_WINDOW,
)


@dataclass
class DesignMatrixContext:
    cancer_type: str
    data_dir: Path
    treatment_path: Path
    outcomes_path: Path
    diag_path: Path
    type_specific_patients: List[str]
    patient_lot_info: pd.DataFrame
    treatment_long: pd.DataFrame
    option: Optional[str] = None

    @property
    def config(self) -> dict:
        return {
            "MARKER_LONG_WINDOW_DAYS": MARKER_LONG_WINDOW_DAYS,
            "MARKER_SHORT_WINDOW_DAYS": MARKER_SHORT_WINDOW_DAYS,
            "MARKER_TAU_LONG": MARKER_TAU_LONG,
            "MARKER_TAU_SHORT": MARKER_TAU_SHORT,
            "ECOG_LOOKBACK_DAYS": ECOG_LOOKBACK_DAYS,
            "ECOG_EVER_GE2_WINDOW": ECOG_EVER_GE2_WINDOW,
        }


def build_patient_lot_info(treatment_df: pd.DataFrame) -> pd.DataFrame:
    """Build a dataframe with PATIENT_ID, LINE, LINE_START information."""
    info = (
        treatment_df.groupby(["PATIENT_ID", "LINE"], observed=True)["START_DATE"]
        .min()
        .rename("LINE_START")
        .reset_index()
    )
    info = info.sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)
    return info


def inflate_treatment_rows(
    treatment_df: pd.DataFrame, patient_lot_info: pd.DataFrame
) -> pd.DataFrame:
    dup_candidates = treatment_df.merge(
        patient_lot_info.copy().rename(columns={"LINE": "LINE_NEXT"}),
        on="PATIENT_ID",
        how="left",
    )
    mask = (
        dup_candidates["STOP_DATE"].notna()
        & (dup_candidates["LINE_NEXT"] > dup_candidates["LINE"]).fillna(False)
        & (dup_candidates["LINE_START"] < dup_candidates["STOP_DATE"]).fillna(False)
    )
    duplicates = dup_candidates.loc[
        mask, [c for c in treatment_df.columns if c != "LINE"] + ["LINE_NEXT"]
    ].copy()
    duplicates = duplicates.rename(columns={"LINE_NEXT": "LINE"})
    if not duplicates.empty:
        duplicates["LINE"] = duplicates["LINE"].astype(treatment_df["LINE"].dtype)
        treatment_df = pd.concat([treatment_df, duplicates], ignore_index=True)
    return treatment_df


def create_context(
    cancer_type: str = "BREAST",
    data_suffix: Optional[str] = None,
    option: Optional[str] = None,
) -> DesignMatrixContext:
    data_dir = Path("data/msk_chord_2024")
    resolved_suffix = data_suffix if data_suffix is not None else get_data_suffix()
    treatment_path = data_csv(f"{cancer_type}_treatment", resolved_suffix)
    outcomes_path = data_csv(f"{cancer_type}_pfs", resolved_suffix)
    diag_path = Path("data/diagnosis_parsed.csv")

    # treatment_raw = pd.read_csv(treatment_path)
    outcomes_raw = pd.read_csv(outcomes_path)

    # assert set(treatment_raw["PATIENT_ID"]) == set(outcomes_raw["PATIENT_ID"])

    # treatment_prepped = filter_and_sort_input(treatment_raw)
    # treatment_prepped["TREATMENT"] = treatment_prepped["SUBTYPE"].str.upper()
    # treatment_prepped = treatment_prepped.drop(columns=["SUBTYPE"])
    # treatment_prepped["LINE"] = _to_int(treatment_prepped["LINE"])

    # patient_lot_info = build_patient_lot_info(treatment_prepped)
    patient_lot_info = outcomes_raw[
        ["PATIENT_ID", "LINE", "LINE_START", "EVENT_DAY"]
    ].copy()
    # Prefer LINE_START (and EVENT_DAY) from PFS to keep anchors consistent when
    # treatments are back-filled across lines.
    # if {"LINE_START", "EVENT_DAY"} <= set(outcomes_raw.columns):
    #     pfs_line_starts = outcomes_raw[
    #         ["PATIENT_ID", "LINE", "LINE_START", "EVENT_DAY"]
    #     ].copy()
    #     pfs_line_starts["LINE_START"] = pfs_line_starts["LINE_START"].astype(int)
    #     pfs_line_starts["EVENT_DAY"] = pd.to_numeric(
    #         pfs_line_starts["EVENT_DAY"], errors="coerce"
    #     ).astype("Int64")
    #     patient_lot_info = patient_lot_info.merge(
    #         pfs_line_starts,
    #         on=["PATIENT_ID", "LINE"],
    #         how="left",
    #         suffixes=("", "_PFS"),
    #         validate="one_to_one",
    #     )
    #     patient_lot_info["LINE_START"] = (
    #         patient_lot_info["LINE_START_PFS"]
    #         .fillna(patient_lot_info["LINE_START"])
    #         .astype(int)
    #     )
    #     patient_lot_info = patient_lot_info.drop(columns=["LINE_START_PFS"])
    #     patient_lot_info = patient_lot_info.sort_values(
    #         ["PATIENT_ID", "LINE"]
    #     ).reset_index(drop=True)

    # treatment_long = treatment_prepped.sort_values(
    #     ["PATIENT_ID", "LINE", "START_DATE"]
    # ).reset_index(drop=True)

    return DesignMatrixContext(
        cancer_type=cancer_type,
        data_dir=data_dir,
        treatment_path=treatment_path,
        outcomes_path=outcomes_path,
        diag_path=diag_path,
        type_specific_patients=set(outcomes_raw["PATIENT_ID"]),
        patient_lot_info=patient_lot_info,
        treatment_long=pd.DataFrame(),  # stale
        option=option,
    )
