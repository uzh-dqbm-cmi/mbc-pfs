import pandas as pd
from ..utils import filter_and_sort_input, iter_measurements_before_line
from typing import Iterable, List, Dict
from .base import ModalityFrame
from config import (
    PATIENT_ID_COL,
    LINE_COL,
    START_DATE_COL,
    RADIOLOGY_REPORT_WINDOW_DAYS,
)

CANCER_PRESENCE_FILE = "data_timeline_cancer_presence.tsv"


def aggregate_cancer_presence_features(
    measurements: pd.DataFrame,
    patient_lot_info: pd.DataFrame,
    body_parts: Iterable[str],
    window_days: int,
) -> pd.DataFrame:
    """Five-state body-part features: Y/N/Indet/not-imaged + ever-positive."""

    rows: List[Dict[str, float]] = []

    for pid, line_id, line_start, history in iter_measurements_before_line(
        measurements,
        patient_lot_info,
    ):
        row: Dict[str, float] = {PATIENT_ID_COL: pid, LINE_COL: line_id}

        for part in body_parts:
            base = f"CANCER_{part}"
            row[f"{base}_IMAGED_Y_STATUS"] = 0
            row[f"{base}_IMAGED_N_STATUS"] = 0
            row[f"{base}_IMAGED_INDET_STATUS"] = 0
            row[f"{base}_MISSING"] = 1
            row[f"{base}_EVER"] = 0

        if history.empty:
            rows.append(row)
            continue

        history = history.copy()
        history["DAYS_AGO"] = line_start - history[START_DATE_COL].astype(int)

        for part in body_parts:
            base = f"CANCER_{part}"
            part_mask = history[part] == 1

            ever_positive = (part_mask & history["HAS_CANCER"].eq("Y")).any()
            row[f"{base}_EVER"] = int(ever_positive)

            recent = history.loc[part_mask & history["DAYS_AGO"].le(window_days)].copy()
            if recent.empty:
                # keep MISSING=1, others already 0
                continue

            row[f"{base}_MISSING"] = 0
            latest = recent.sort_values(START_DATE_COL).iloc[-1]
            status = latest["HAS_CANCER"]
            if status == "Y":
                row[f"{base}_IMAGED_Y_STATUS"] = 1
            elif status == "N":
                row[f"{base}_IMAGED_N_STATUS"] = 1
            else:
                row[f"{base}_IMAGED_INDET_STATUS"] = 1
        rows.append(row)

    out = pd.DataFrame(rows)
    keys = patient_lot_info[[PATIENT_ID_COL, LINE_COL]].drop_duplicates()
    out = keys.merge(
        out, on=[PATIENT_ID_COL, LINE_COL], how="left", validate="one_to_one"
    )
    return out


def build_cancer_presence_features(ctx) -> ModalityFrame:
    raw = pd.read_csv(ctx.data_dir / CANCER_PRESENCE_FILE, sep="\t")
    presence = filter_and_sort_input(raw, ctx.type_specific_patients)
    body_parts = ["CHEST", "ABDOMEN", "PELVIS", "HEAD", "OTHER"]

    presence = presence[[PATIENT_ID_COL, START_DATE_COL, "HAS_CANCER"] + body_parts]
    presence[START_DATE_COL] = presence[START_DATE_COL].astype(int)
    for part in body_parts:
        presence[part] = presence[part].map(
            {
                "TRUE": 1,
                "FALSE": 0,
                True: 1,
                False: 0,
            }
        )
    presence["HAS_CANCER"] = presence["HAS_CANCER"].map(
        {"Y": "Y", "N": "N", "Indeterminate": "INDETERMINATE"}
    )
    window_days = RADIOLOGY_REPORT_WINDOW_DAYS

    result = aggregate_cancer_presence_features(
        measurements=presence,
        patient_lot_info=ctx.patient_lot_info,
        body_parts=body_parts,
        window_days=window_days,
    )

    for col in result.columns:
        if col.endswith("_STATUS"):
            result[col] = result[col].fillna(0).astype("int8")
        elif col.endswith("_MISSING") or col.endswith("_EVER"):
            result[col] = result[col].fillna(1).astype("int8")
    result = result.sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)
    return ModalityFrame(name="CANCER_PRESENCE", frame=result, temporal=True)
