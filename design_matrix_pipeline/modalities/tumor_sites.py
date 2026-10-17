from __future__ import annotations
from typing import Iterable, List, Dict

import pandas as pd

from ..utils import filter_and_sort_input, iter_measurements_before_line
from .base import ModalityFrame
import numpy as np

from config import (
    LINE_COL,
    PATIENT_ID_COL,
    START_DATE_COL,
    RADIOLOGY_REPORT_WINDOW_DAYS,
)


TUMOR_SITE_FILE = "data_timeline_tumor_sites.tsv"


def aggregate_tumor_site_features(
    measurements: pd.DataFrame,
    patient_lot_info: pd.DataFrame,
    value_cols: Iterable[str],
    window_days: int,
) -> pd.DataFrame:
    """Summarize tumor-site indicators with 90d window + ever-positive flags."""

    rows: List[Dict[str, float]] = []

    for pid, line_id, line_start, history in iter_measurements_before_line(
        measurements,
        patient_lot_info,
    ):
        row: Dict[str, float] = {PATIENT_ID_COL: pid, LINE_COL: line_id}

        for col in value_cols:
            base = f"TUMOR_SITE_{col}"
            row[f"{base}_IN_WINDOW"] = 0
            row[f"{base}_EVER"] = 0

        if history.empty:
            rows.append(row)
            continue

        measurement_days = line_start - history[START_DATE_COL].to_numpy(dtype=int)
        values_all = history.loc[:, value_cols].to_numpy(dtype=float)

        for idx, col in enumerate(value_cols):
            base = f"TUMOR_SITE_{col}"
            col_values = values_all[:, idx]
            positive_mask = col_values > 0
            if not positive_mask.any():
                continue

            observed_days = measurement_days[positive_mask]
            row[f"{base}_EVER"] = 1
            row[f"{base}_IN_WINDOW"] = int((observed_days <= window_days).any())

        rows.append(row)

    out = pd.DataFrame(rows)
    keys = patient_lot_info[[PATIENT_ID_COL, LINE_COL]].drop_duplicates()
    out = keys.merge(out, on=[PATIENT_ID_COL, LINE_COL], how="left")
    return out


def _impute_tumor_site_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_IN_WINDOW") or col.endswith("_EVER"):
            out[col] = out[col].fillna(0).astype("int8")
    return out


def build_tumor_site_features(ctx) -> ModalityFrame:
    raw = pd.read_csv(ctx.data_dir / TUMOR_SITE_FILE, sep="	")
    sites = filter_and_sort_input(raw, ctx.type_specific_patients)

    subsites = sorted(sites["TUMOR_SITE"].dropna().astype(str).unique().tolist())
    for sub in subsites:
        # Rows only exist when a site is present; leave others missing.
        sites[sub] = np.where(sites["TUMOR_SITE"] == sub, 1, np.nan)
    measurements = sites[["PATIENT_ID", "START_DATE"] + subsites].copy()

    window_days = RADIOLOGY_REPORT_WINDOW_DAYS
    result = aggregate_tumor_site_features(
        measurements=measurements,
        patient_lot_info=ctx.patient_lot_info,
        value_cols=subsites,
        window_days=window_days,
    )

    result = _impute_tumor_site_features(result)
    result = result.sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)

    return ModalityFrame(name="TUMOR_SITES", frame=result, temporal=True)
