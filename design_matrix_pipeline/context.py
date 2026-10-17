from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from config import CANCER_TYPE, LINE_START_WINDOW_DAYS, MSK_DATA_DIR
from data_paths import data_dir


@dataclass
class DesignMatrixContext:
    data_dir: Path
    outcomes_path: Path
    diag_path: Path
    type_specific_patients: Set[str]
    patient_lot_info: pd.DataFrame
    window_days: int = LINE_START_WINDOW_DAYS
    top_agents: Optional[Dict[str, Set[str]]] = None
    diagnosis_categories: Optional[Dict[str, pd.Index]] = None


def create_context(window_days: int = LINE_START_WINDOW_DAYS) -> DesignMatrixContext:
    out_dir = data_dir(window_days)
    outcomes_path = out_dir / f"{CANCER_TYPE}_pfs.csv"
    outcomes_raw = pd.read_csv(outcomes_path)
    # treatment_raw = pd.read_csv(treatment_path)

    # assert set(treatment_raw["PATIENT_ID"]) == set(outcomes_raw["PATIENT_ID"])

    # treatment_prepped = filter_and_sort_input(treatment_raw)
    # treatment_prepped["TREATMENT"] = treatment_prepped["SUBTYPE"].str.upper()
    # treatment_prepped = treatment_prepped.drop(columns=["SUBTYPE"])
    # treatment_prepped["LINE"] = _to_int(treatment_prepped["LINE"])

    # patient_lot_info = build_patient_lot_info(treatment_prepped)
    # Prefer LINE_START (and EVENT_DAY) from PFS to keep anchors consistent when
    # treatments are back-filled across lines.
    patient_lot_info = outcomes_raw[
        ["PATIENT_ID", "LINE", "LINE_START", "EVENT_DAY"]
    ].copy()
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
        data_dir=MSK_DATA_DIR,
        outcomes_path=outcomes_path,
        diag_path=out_dir / "diagnosis_parsed.csv",
        type_specific_patients=set(outcomes_raw["PATIENT_ID"]),
        patient_lot_info=patient_lot_info,
        window_days=window_days,
    )
