from __future__ import annotations

import pandas as pd
import re

from .base import ModalityFrame
from ..utils import filter_and_sort_input

LOOKBACK_DAYS = 180


def _normalize_token(series: pd.Series) -> pd.Series:
    tokens = series.fillna("").astype(str).str.strip().str.upper()
    tokens = tokens.replace("", pd.NA)
    return tokens


def _exposure_summary(
    events: list[tuple[int, int]], window_start: int, line_start: int
) -> tuple[int, int]:
    """
    Determine whether any event stops within [window_start, line_start) and the
    days since its stop. If an event is ongoing at line_start, LAST_STOP_DAY = 0.
    If no prior event exists, LAST_STOP_DAY = LOOKBACK_DAYS + 1 (181).
    """
    if not events:
        return 0, LOOKBACK_DAYS + 1
    exposed = False
    last_stop: int | None = None
    for start, stop in events:
        if stop >= window_start and start < line_start:
            exposed = True
            last_stop = stop if last_stop is None else max(last_stop, stop)
    if not exposed:
        return 0, LOOKBACK_DAYS + 1
    if last_stop is not None and last_stop >= line_start:
        return 1, 0
    assert last_stop is not None
    return 1, int(line_start - last_stop)


def _patient_event_lookup(
    events: pd.DataFrame, label_col: str
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """
    Build a mapping patient -> label -> sorted list of (start, stop) days.
    STOP_DATE defaults to START_DATE when missing.
    """
    lookup: dict[str, dict[str, list[tuple[int, int]]]] = {}
    if events.empty:
        return lookup
    grouped = events.groupby(["PATIENT_ID", label_col], observed=True)
    for (pid, label), grp in grouped:
        starts = pd.to_numeric(grp["START_DATE"], errors="coerce")
        stops = pd.to_numeric(grp.get("STOP_DATE"), errors="coerce")
        stops = stops.fillna(starts)
        pairs = [
            (int(s), int(max(s, e)))
            for s, e in zip(starts.dropna().astype(int), stops.dropna().astype(int))
        ]
        if pairs:
            lookup.setdefault(pid, {}).setdefault(label, []).extend(sorted(pairs))
    return lookup


def build_history_features(ctx) -> ModalityFrame:
    lot_info = ctx.patient_lot_info[["PATIENT_ID", "LINE", "LINE_START"]].copy()
    patient_ids = lot_info["PATIENT_ID"].unique()

    # Surgery/radiation events from timelines
    surgery = pd.read_csv(
        ctx.data_dir / "data_timeline_surgery.tsv", sep="\t", dtype=str
    )
    surgery = filter_and_sort_input(surgery, patient_ids=patient_ids)
    surgery["SUBTYPE"] = _normalize_token(surgery["SUBTYPE"])
    surgery = surgery[surgery["SUBTYPE"] == "PROCEDURE"]
    surgery = surgery.dropna(subset=["START_DATE"])
    surgery_map = _patient_event_lookup(surgery, "SUBTYPE")

    radiation = pd.read_csv(
        ctx.data_dir / "data_timeline_radiation.tsv", sep="\t", dtype=str
    )
    radiation = filter_and_sort_input(radiation, patient_ids=patient_ids)
    radiation["EVENT_TYPE"] = _normalize_token(radiation["EVENT_TYPE"])
    radiation = radiation[radiation["EVENT_TYPE"] == "TREATMENT"]
    radiation = radiation.dropna(subset=["START_DATE"])
    radiation_map = _patient_event_lookup(radiation, "EVENT_TYPE")

    treatment_history = ctx.treatment_long[
        ["PATIENT_ID", "START_DATE", "STOP_DATE", "TREATMENT"]
    ].copy()
    treatment_history["TREATMENT"] = _normalize_token(treatment_history["TREATMENT"])
    treatment_history = treatment_history.dropna(subset=["START_DATE", "TREATMENT"])
    treatment_history["START_DATE"] = pd.to_numeric(
        treatment_history["START_DATE"], errors="coerce"
    )
    treatment_history["STOP_DATE"] = pd.to_numeric(
        treatment_history["STOP_DATE"], errors="coerce"
    )
    treatment_history["STOP_DATE"] = treatment_history["STOP_DATE"].fillna(
        treatment_history["START_DATE"]
    )
    treatment_history = treatment_history.dropna(subset=["START_DATE"]).copy()

    def _safe_label(text: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "_", text).strip("_")

    treatment_history["TREATMENT"] = treatment_history["TREATMENT"].apply(_safe_label)
    treatment_history = treatment_history.drop_duplicates(
        subset=["PATIENT_ID", "START_DATE", "STOP_DATE", "TREATMENT"]
    )

    treatment_values = sorted(treatment_history["TREATMENT"].unique().tolist())

    treatment_event_map = _patient_event_lookup(treatment_history, "TREATMENT")

    records = []
    for pid, patient_lines in lot_info.groupby("PATIENT_ID", sort=False):
        t_events = treatment_event_map.get(pid, {})
        s_events = surgery_map.get(pid, {})
        r_events = radiation_map.get(pid, {})
        for _, line_row in patient_lines.iterrows():
            line_start = int(line_row["LINE_START"])
            window_start = line_start - LOOKBACK_DAYS
            record = {"PATIENT_ID": pid, "LINE": line_row["LINE"]}

            # Surgery/radiation exposure in past 180d
            s_flag, s_last = _exposure_summary(
                s_events.get("PROCEDURE", []), window_start, line_start
            )
            r_flag, r_last = _exposure_summary(
                r_events.get("TREATMENT", []), window_start, line_start
            )
            record["HISTORY_SURGERY_180D_EXPOSED"] = s_flag
            record["HISTORY_SURGERY_LAST_STOP_DAY"] = s_last
            record["HISTORY_RADIATION_THERAPY_180D_EXPOSED"] = r_flag
            record["HISTORY_RADIATION_THERAPY_LAST_STOP_DAY"] = r_last

            for treatment in treatment_values:
                exposed, last_stop = _exposure_summary(
                    t_events.get(treatment, []), window_start, line_start
                )
                record[f"HISTORY_TREATMENT_{treatment}_180D_EXPOSED"] = exposed
                record[f"HISTORY_TREATMENT_{treatment}_LAST_STOP_DAY"] = last_stop

            records.append(record)

    history_frame = pd.DataFrame.from_records(records)
    desired_cols = ["PATIENT_ID", "LINE"]
    desired_cols += [
        "HISTORY_SURGERY_180D_EXPOSED",
        "HISTORY_SURGERY_LAST_STOP_DAY",
        "HISTORY_RADIATION_THERAPY_180D_EXPOSED",
        "HISTORY_RADIATION_THERAPY_LAST_STOP_DAY",
    ]
    desired_cols += [f"HISTORY_TREATMENT_{t}_180D_EXPOSED" for t in treatment_values]
    desired_cols += [f"HISTORY_TREATMENT_{t}_LAST_STOP_DAY" for t in treatment_values]

    for col in desired_cols:
        if col not in history_frame.columns:
            # defaults: 0 for flags, LOOKBACK_DAYS+1 for last-day
            if col.endswith("_180D_EXPOSED"):
                history_frame[col] = 0
            elif col.endswith("_LAST_STOP_DAY"):
                history_frame[col] = LOOKBACK_DAYS + 1
            else:
                history_frame[col] = pd.NA

    history_frame = history_frame[desired_cols]

    flag_cols = [c for c in history_frame.columns if c.endswith("_180D_EXPOSED")]
    last_cols = [c for c in history_frame.columns if c.endswith("_LAST_STOP_DAY")]
    history_frame[flag_cols] = history_frame[flag_cols].fillna(0).astype("int8")
    history_frame[last_cols] = (
        history_frame[last_cols].fillna(LOOKBACK_DAYS + 1).astype("int32")
    )

    history_frame = history_frame.sort_values(["PATIENT_ID", "LINE"]).reset_index(
        drop=True
    )
    return ModalityFrame(name="HISTORY", frame=history_frame, temporal=True)
