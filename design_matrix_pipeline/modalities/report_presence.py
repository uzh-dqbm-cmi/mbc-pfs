from __future__ import annotations

import numpy as np
import pandas as pd

from config import LINE_COL, PATIENT_ID_COL, START_DATE_COL
from ..utils import filter_and_sort_input, iter_measurements_before_line
from .base import ModalityFrame

PROGRESSION_FILE = "data_timeline_progression.tsv"
from .tumor_sites import TUMOR_SITE_FILE
from .cancer_presence import CANCER_PRESENCE_FILE
from config import RADIOLOGY_REPORT_WINDOW_DAYS


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
        has_report = 0
        if not history.empty:
            days_ago = line_start - history[START_DATE_COL].to_numpy(dtype=int)
            in_window = (days_ago >= 0) & (days_ago <= window_days)
            if np.any(in_window):
                has_report = 1

        rows.append(
            {
                PATIENT_ID_COL: pid,
                LINE_COL: line_id,
                f"HAS_RADIOLOGY_REPORT_{window_days}_PRIOR": has_report,
            }
        )

    out = pd.DataFrame(rows)
    keys = patient_lot_info[[PATIENT_ID_COL, LINE_COL]].drop_duplicates()
    out = keys.merge(
        out, on=[PATIENT_ID_COL, LINE_COL], how="left", validate="one_to_one"
    )
    return out.sort_values([PATIENT_ID_COL, LINE_COL]).reset_index(drop=True)


def build_report_presence_features(ctx) -> ModalityFrame:
    progression = pd.read_csv(ctx.data_dir / PROGRESSION_FILE, sep="\t")[
        [PATIENT_ID_COL, START_DATE_COL]
    ]
    progression = filter_and_sort_input(progression, ctx.type_specific_patients)
    tumor_sites = pd.read_csv(ctx.data_dir / TUMOR_SITE_FILE, sep="\t")[
        [PATIENT_ID_COL, START_DATE_COL]
    ]
    tumor_sites = filter_and_sort_input(tumor_sites, ctx.type_specific_patients)
    cancer_presence = pd.read_csv(ctx.data_dir / CANCER_PRESENCE_FILE, sep="\t")[
        [PATIENT_ID_COL, START_DATE_COL]
    ]
    cancer_presence = filter_and_sort_input(cancer_presence, ctx.type_specific_patients)

    # merge the above dataframes on PATIENT_ID_COL and START_DATE_COL
    measurements = pd.merge(
        progression, tumor_sites, on=[PATIENT_ID_COL, START_DATE_COL], how="outer"
    )
    measurements = pd.merge(
        measurements, cancer_presence, on=[PATIENT_ID_COL, START_DATE_COL], how="outer"
    )
    result = _count_recent_progressions(
        measurements=measurements,
        patient_lot_info=ctx.patient_lot_info,
        window_days=RADIOLOGY_REPORT_WINDOW_DAYS,
    )

    result[f"HAS_RADIOLOGY_REPORT_{RADIOLOGY_REPORT_WINDOW_DAYS}_PRIOR"] = (
        result[f"HAS_RADIOLOGY_REPORT_{RADIOLOGY_REPORT_WINDOW_DAYS}_PRIOR"]
        .fillna(0)
        .astype(int)
    )
    return ModalityFrame(name="REPORT_PRESENCE", frame=result, temporal=True)
