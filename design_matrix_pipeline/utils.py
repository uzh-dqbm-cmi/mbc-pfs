from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from utils import _to_float, _to_int


def filter_and_sort_input(
    df: pd.DataFrame,
    patient_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Normalize column names, filter to the provided patient ids, and coerce dates.

    The returned dataframe is sorted by ``PATIENT_ID`` and, when available,
    ``START_DATE``.
    """
    out = df.copy()
    out.columns = [c.strip().upper().replace(" ", "_") for c in out.columns]
    if patient_ids is not None:
        out = out[out["PATIENT_ID"].isin(patient_ids)].copy()
    sort_cols: List[str] = ["PATIENT_ID"]
    if "START_DATE" in out.columns:
        out.loc[:, "START_DATE"] = _to_int(out["START_DATE"])
        sort_cols.append("START_DATE")
    if "STOP_DATE" in out.columns:
        out.loc[:, "STOP_DATE"] = _to_float(out["STOP_DATE"])
        sort_cols.append("STOP_DATE")
    if "_ROW_ID" in out.columns:
        out.loc[:, "_ROW_ID"] = _to_int(out["_ROW_ID"])
        sort_cols.append("_ROW_ID")
    # Use a stable sort so ties (e.g., multiple measurements on the same START_DATE)
    # preserve deterministic input order.
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out


def iter_measurements_before_line(
    measurements: pd.DataFrame,
    patient_lot_info: pd.DataFrame,
    patient_id_col: str = "PATIENT_ID",
    line_col: str = "LINE",
    line_start_col: str = "LINE_START",
    start_date_col: str = "START_DATE",
    include_same_day_measurement: bool = False,
) -> Iterable[tuple[str, int, float, pd.DataFrame]]:
    """Yield measurement history for each patient-lot combination.

    For each patient, lot combination, this generator yields the patient id, lot
    number, lot start, and the slice of ``measurements`` containing all
    observations recorded on or before the lot start.
    """

    side = "right" if include_same_day_measurement else "left"

    patient_lot_info = patient_lot_info.sort_values(
        [patient_id_col, line_start_col]
    ).reset_index(drop=True)

    values = measurements.dropna(subset=[patient_id_col, start_date_col]).copy()
    values[start_date_col] = values[start_date_col].astype(int)
    values = values.sort_values([patient_id_col, start_date_col], kind="mergesort")

    grouped_values = values.groupby(patient_id_col, sort=False)
    empty_slice = values.iloc[0:0].copy()

    for pid, patient_lots in patient_lot_info.groupby(patient_id_col, sort=False):
        try:
            patient_values = grouped_values.get_group(pid)
        except KeyError:
            for _, line_row in patient_lots.iterrows():
                yield pid, line_row[line_col], int(
                    line_row[line_start_col]
                ), empty_slice.copy()
            continue

        measurement_dates = patient_values[start_date_col].to_numpy(dtype=int)
        for _, line_row in patient_lots.iterrows():
            current_line_start = int(line_row[line_start_col])
            pos = np.searchsorted(measurement_dates, current_line_start, side=side)
            history = patient_values.iloc[:pos].copy()
            # make sure the history is sorted by start_date_col
            if not history[start_date_col].is_monotonic_increasing:
                raise ValueError(
                    f"History for patient {pid}, line {line_row[line_col]} is not sorted by {start_date_col}"
                )
            yield pid, line_row[line_col], current_line_start, history
