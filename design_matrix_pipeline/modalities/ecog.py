from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import filter_and_sort_input
from .base import ModalityFrame
from ..utils import iter_measurements_before_line
from config import (
    START_DATE_COL,
    ECOG_LOOKBACK_DAYS,
    ECOG_EVER_GE2_WINDOW,
)

ECOG_FILE = "data_timeline_performance_status.tsv"


def build_ecog_features(ctx) -> ModalityFrame:
    raw = pd.read_csv(ctx.data_dir / ECOG_FILE, sep="\t")
    ecog = filter_and_sort_input(raw, ctx.type_specific_patients)
    ecog = ecog[["PATIENT_ID", "START_DATE", "ECOG"]].copy()
    ecog["ECOG"] = pd.to_numeric(ecog["ECOG"], errors="coerce")
    ecog = ecog.dropna(subset=["ECOG"])
    ecog = ecog.sort_values(["PATIENT_ID", "START_DATE"]).reset_index(drop=True)

    base = ctx.patient_lot_info[["PATIENT_ID", "LINE"]].copy()
    records: list[dict] = []

    for pid, line, line_start, history in iter_measurements_before_line(
        ecog, ctx.patient_lot_info
    ):
        record = {"PATIENT_ID": pid, "LINE": line}
        record["ECOG_LAST_OBS"] = np.nan
        record["ECOG_LAST_OBS_DAY"] = np.nan
        record["EVER_GE2_180D"] = np.nan
        record["ECOG_MISSING"] = 1

        if not history.empty:
            dates = history[START_DATE_COL].to_numpy(dtype=int)
            values = history["ECOG"].to_numpy(dtype=int)
            days_before = line_start - dates
            record["EVER_GE2_180D"] = bool(
                np.any((values >= 2) & (days_before <= ECOG_EVER_GE2_WINDOW))
            )
            mask_window = days_before <= ECOG_LOOKBACK_DAYS
            if not mask_window.any():
                continue
            record["ECOG_MISSING"] = 0
            scoped_values = values[mask_window]
            scoped_days = days_before[mask_window]
            record["ECOG_LAST_OBS"] = scoped_values[-1]
            record["ECOG_LAST_OBS_DAY"] = scoped_days[-1]

        records.append(record)

    features = pd.DataFrame(records)
    out = base.merge(features, on=["PATIENT_ID", "LINE"], how="left")
    out["EVER_GE2_180D"] = (
        out["EVER_GE2_180D"].astype("boolean").fillna(False).astype(int)
    )
    out["ECOG_MISSING"] = out["ECOG_LAST_OBS"].isna().astype(int)
    out["ECOG_LAST_OBS"] = out["ECOG_LAST_OBS"].fillna(0).astype(int)
    return ModalityFrame(name="ECOG", frame=out, temporal=True)
