from __future__ import annotations

import numpy as np
import pandas as pd

from config import LINE_COL, PATIENT_ID_COL, START_DATE_COL
from ..utils import filter_and_sort_input, iter_measurements_before_line
from .base import ModalityFrame

PROGRESSION_FILE = "data_timeline_progression.tsv"
PROGRESSION_LOOKBACK_DAYS = 90


def _count_recent_progressions(
    measurements: pd.DataFrame, patient_lot_info: pd.DataFrame, window_days: int
) -> pd.DataFrame:
    """Count positive progression reports within a fixed lookback window."""

    rows: list[dict[str, int]] = []
    for pid, line_id, line_start, history in iter_measurements_before_line(
        measurements,
        patient_lot_info,
        include_same_day_measurement=True,
    ):
        count = 0
        if not history.empty:
            days_ago = line_start - history[START_DATE_COL].to_numpy(dtype=int)
            progression_vals = history["PROGRESSION"].to_numpy(dtype=object)
            positive_mask = progression_vals == "Y"
            in_window = (days_ago >= 0) & (days_ago <= window_days)
            count = int(np.count_nonzero(positive_mask & in_window))

        rows.append(
            {
                PATIENT_ID_COL: pid,
                LINE_COL: line_id,
                "PROGRESSION_POS_90D_COUNT": count,
            }
        )

    out = pd.DataFrame(rows)
    keys = patient_lot_info[[PATIENT_ID_COL, LINE_COL]].drop_duplicates()
    out = keys.merge(out, on=[PATIENT_ID_COL, LINE_COL], how="left")
    out["PROGRESSION_POS_90D_COUNT"] = (
        out["PROGRESSION_POS_90D_COUNT"].fillna(0).astype("int8")
    )
    return out.sort_values([PATIENT_ID_COL, LINE_COL]).reset_index(drop=True)


def build_progression_features(ctx) -> ModalityFrame:
    raw = pd.read_csv(ctx.data_dir / PROGRESSION_FILE, sep="\t")
    progression = filter_and_sort_input(raw, ctx.type_specific_patients)
    progression = progression.dropna(subset=[START_DATE_COL, "PROGRESSION"])
    progression[START_DATE_COL] = progression[START_DATE_COL].astype(int)
    progression["PROGRESSION"] = progression["PROGRESSION"].astype(str).str.upper()

    measurements = progression[[PATIENT_ID_COL, START_DATE_COL, "PROGRESSION"]].copy()

    result = _count_recent_progressions(
        measurements=measurements,
        patient_lot_info=ctx.patient_lot_info,
        window_days=PROGRESSION_LOOKBACK_DAYS,
    )
    return ModalityFrame(name="PROGRESSION", frame=result, temporal=True)
