from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ModalityFrame
from ..utils import filter_and_sort_input

LINE_START_WINDOW_DAYS = 28
SURGERY_RECEIVED_IN_LINE_COL = "RECEIVED_SURGERY"
RADIATION_RECEIVED_IN_LINE_COL = "RECEIVED_RADIATION_THERAPY"
SURGERY_PLANNED_IN_LINE_COL = "PLANNED_SURGERY"
RADIATION_PLANNED_IN_LINE_COL = "PLANNED_RADIATION_THERAPY"


def _normalize_token(series: pd.Series) -> pd.Series:
    tokens = series.fillna("").astype(str).str.strip().str.upper()
    tokens = tokens.replace("", pd.NA)
    return tokens


def _patient_date_lookup(events: pd.DataFrame) -> dict[str, list[int]]:
    lookup: dict[str, list[int]] = {}
    if events.empty:
        return lookup
    grouped = events.groupby("PATIENT_ID", observed=True)["START_DATE"]
    for pid, dates in grouped:
        cleaned = pd.to_numeric(dates, errors="coerce").dropna().astype(int)
        if not cleaned.empty:
            lookup[pid] = sorted(set(cleaned.tolist()))
    return lookup


def _event_in_interval(
    dates: list[int], interval_start: int, interval_end: int | None
) -> int:
    if not dates:
        return 0
    arr = np.asarray(dates, dtype=int)
    start_idx = int(np.searchsorted(arr, interval_start, side="left"))
    if interval_end is None:
        return 1 if start_idx < arr.size else 0
    end_idx = int(np.searchsorted(arr, interval_end, side="left"))
    return 1 if start_idx < end_idx else 0


def build_local_treatment_features(ctx) -> ModalityFrame:
    lot_info = ctx.patient_lot_info[
        ["PATIENT_ID", "LINE", "LINE_START", "EVENT_DAY"]
    ].copy()
    lot_info = lot_info.sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)
    lot_info["EVENT_DAY"] = lot_info["EVENT_DAY"].astype("Int64")
    patient_ids = lot_info["PATIENT_ID"].unique()

    surgery = pd.read_csv(
        ctx.data_dir / "data_timeline_surgery.tsv", sep="\t", dtype=str
    )
    surgery = filter_and_sort_input(surgery, patient_ids=patient_ids)
    surgery["SUBTYPE"] = _normalize_token(surgery["SUBTYPE"])
    surgery = surgery[surgery["SUBTYPE"] == "PROCEDURE"]
    surgery = surgery.dropna(subset=["START_DATE"])
    surgery_dates = _patient_date_lookup(surgery)

    radiation = pd.read_csv(
        ctx.data_dir / "data_timeline_radiation.tsv", sep="\t", dtype=str
    )
    radiation = filter_and_sort_input(radiation, patient_ids=patient_ids)
    radiation["EVENT_TYPE"] = _normalize_token(radiation["EVENT_TYPE"])
    radiation = radiation[radiation["EVENT_TYPE"] == "TREATMENT"]
    radiation = radiation.dropna(subset=["START_DATE"])
    radiation_dates = _patient_date_lookup(radiation)

    records = []
    for pid, patient_lines in lot_info.groupby("PATIENT_ID", sort=False):
        s_dates = surgery_dates.get(pid, [])
        r_dates = radiation_dates.get(pid, [])
        for _, line_row in patient_lines.iterrows():
            line_start = int(line_row["LINE_START"])
            planned_end = line_start + LINE_START_WINDOW_DAYS + 1
            line_end = line_row.get("EVENT_DAY")
            line_end_int = int(line_end) if pd.notna(line_end) else None
            records.append(
                {
                    "PATIENT_ID": pid,
                    "LINE": line_row["LINE"],
                    SURGERY_PLANNED_IN_LINE_COL: _event_in_interval(
                        s_dates, line_start, planned_end
                    ),
                    RADIATION_PLANNED_IN_LINE_COL: _event_in_interval(
                        r_dates, line_start, planned_end
                    ),
                    SURGERY_RECEIVED_IN_LINE_COL: _event_in_interval(
                        s_dates, line_start, line_end_int
                    ),
                    RADIATION_RECEIVED_IN_LINE_COL: _event_in_interval(
                        r_dates, line_start, line_end_int
                    ),
                }
            )

    desired_cols = [
        "PATIENT_ID",
        "LINE",
        SURGERY_PLANNED_IN_LINE_COL,
        RADIATION_PLANNED_IN_LINE_COL,
        SURGERY_RECEIVED_IN_LINE_COL,
        RADIATION_RECEIVED_IN_LINE_COL,
    ]

    history_frame = pd.DataFrame.from_records(records, columns=desired_cols)

    history_frame[SURGERY_PLANNED_IN_LINE_COL] = (
        history_frame[SURGERY_PLANNED_IN_LINE_COL].fillna(0).astype("int8")
    )
    history_frame[RADIATION_PLANNED_IN_LINE_COL] = (
        history_frame[RADIATION_PLANNED_IN_LINE_COL].fillna(0).astype("int8")
    )
    history_frame[SURGERY_RECEIVED_IN_LINE_COL] = (
        history_frame[SURGERY_RECEIVED_IN_LINE_COL].fillna(0).astype("int8")
    )
    history_frame[RADIATION_RECEIVED_IN_LINE_COL] = (
        history_frame[RADIATION_RECEIVED_IN_LINE_COL].fillna(0).astype("int8")
    )

    history_frame = history_frame.sort_values(["PATIENT_ID", "LINE"]).reset_index(
        drop=True
    )
    return ModalityFrame(name="LOCAL_TREATMENT", frame=history_frame, temporal=True)
