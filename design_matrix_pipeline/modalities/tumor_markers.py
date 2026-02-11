from __future__ import annotations

import pandas as pd

from ..utils import filter_and_sort_input, iter_measurements_before_line
from .base import ModalityFrame
import numpy as np
from typing import Dict, List

from config import PATIENT_ID_COL, LINE_COL, START_DATE_COL

TUMOR_MARKER_FILE = "data_timeline_tumor_markers.tsv"
TEST_COL = "TEST"
RESULT_COL = "RESULT"


def _impute_tumor_marker_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_MISSING"):
            out[col] = out[col].fillna(1).astype(int)
        elif col.endswith("_COUNT"):
            out[col] = out[col].fillna(0).astype(int)
        elif col.endswith("_ABOVE_CUTOFF") or col.endswith("_RISE_GT20PCT"):
            out[col] = out[col].fillna(0).astype(int)
    return out


def recency_weights(days_before: np.ndarray, tau: float) -> np.ndarray:
    """Return normalized exponential-decay weights for recency aggregation."""

    days_before = np.asarray(days_before, dtype=float)
    weights = np.exp(-days_before / float(tau))
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid weights computed for recency aggregation.")
    return weights / total


def aggregate_biomarkers(
    tumor_markers: pd.DataFrame,
    patient_lot_info: pd.DataFrame,
    test_map: Dict[str, str],
    short_window: int,
    long_window: int,
    tau_short: float,
    tau_long: float,
) -> pd.DataFrame:
    """Aggregate long-form biomarker measurements into per-line log summaries."""

    cutoff_map = {"CA15_3": 30.0, "CEA": 5.0}
    rows: List[Dict[str, float]] = []

    for pid, line_id, line_start, history in iter_measurements_before_line(
        tumor_markers,
        patient_lot_info,
    ):
        row: Dict[str, float] = {PATIENT_ID_COL: pid, LINE_COL: line_id}

        for out_key, test_name in test_map.items():
            test_history = history.loc[history["TEST"] == test_name].copy()
            for scope in ("SHORT", "LONG"):
                prefix = f"{out_key}_{scope}"
                row[f"{prefix}_RW_MEAN"] = np.nan
                row[f"{prefix}_MAX"] = np.nan
                row[f"{prefix}_MIN"] = np.nan
                row[f"{prefix}_MISSING"] = 1.0
                row[f"{prefix}_COUNT"] = 0
                row[f"{prefix}_SLOPE"] = np.nan
                row[f"{prefix}_DELTA_LAST"] = np.nan
            row[f"{out_key}_LAST_OBS_LOG"] = np.nan
            row[f"{out_key}_LAST_OBS_DAY"] = np.nan
            row[f"{out_key}_SHORT_OVER_LONG"] = np.nan
            row[f"{out_key}_SHORT_MINUS_LONG"] = np.nan
            row[f"{out_key}_ABOVE_CUTOFF"] = 0
            row[f"{out_key}_RISE_GT20PCT"] = 0

            if test_history.empty:
                continue
            # sort by start date again just in case
            test_history = test_history.sort_values(START_DATE_COL)
            # log-transform and avoid issues with 0
            raw_values = test_history["RESULT"].to_numpy(dtype=float)
            values = np.log1p(raw_values)
            days_before = line_start - test_history[START_DATE_COL].to_numpy(dtype=int)

            last_obs_log = np.nan
            last_obs_day = np.nan

            for scope, (window, tau) in (
                ("LONG", (long_window, tau_long)),
                ("SHORT", (short_window, tau_short)),
            ):
                prefix = f"{out_key}_{scope}"
                mask_window = days_before <= window
                if not np.any(mask_window):
                    continue
                scoped_values = values[mask_window]
                scoped_days = days_before[mask_window]
                row[f"{prefix}_COUNT"] = int(mask_window.sum())
                weights = recency_weights(scoped_days, tau)
                row[f"{prefix}_RW_MEAN"] = float(np.dot(weights, scoped_values))
                row[f"{prefix}_MAX"] = float(scoped_values.max())
                row[f"{prefix}_MIN"] = float(scoped_values.min())
                row[f"{prefix}_MISSING"] = 0.0
                if scoped_values.size >= 2:
                    unique_days = np.unique(scoped_days)
                    if unique_days.size >= 2:
                        slope, _ = np.polyfit(
                            scoped_days - scoped_days.mean(), scoped_values, 1
                        )
                        row[f"{prefix}_SLOPE"] = float(slope)
                        row[f"{prefix}_DELTA_LAST"] = float(
                            scoped_values[-1] - scoped_values[-2]
                        )
                    else:
                        # all samples same day → skip slope/delta
                        row[f"{prefix}_SLOPE"] = np.nan
                        row[f"{prefix}_DELTA_LAST"] = np.nan
                last_obs_day = float(scoped_days[-1])
                last_obs_log = float(scoped_values[-1])
            row[f"{out_key}_LAST_OBS_LOG"] = last_obs_log
            row[f"{out_key}_LAST_OBS_DAY"] = last_obs_day
            # short vs long contrasts if both available
            short_mean = row[f"{out_key}_SHORT_RW_MEAN"]
            long_mean = row[f"{out_key}_LONG_RW_MEAN"]
            if not np.isnan(short_mean) and not np.isnan(long_mean) and long_mean != 0:
                row[f"{out_key}_SHORT_OVER_LONG"] = float(short_mean / long_mean)
                row[f"{out_key}_SHORT_MINUS_LONG"] = float(short_mean - long_mean)
            # threshold flags using raw values
            last_raw = raw_values[-1]
            cutoff = cutoff_map.get(out_key)
            if cutoff is not None and last_raw > cutoff:
                row[f"{out_key}_ABOVE_CUTOFF"] = 1
            if raw_values.size >= 2:
                prev_raw = raw_values[-2]
                if prev_raw > 0 and (last_raw - prev_raw) / prev_raw > 0.20:
                    row[f"{out_key}_RISE_GT20PCT"] = 1

        rows.append(row)

    out = pd.DataFrame(rows)
    keys = patient_lot_info[[PATIENT_ID_COL, LINE_COL]].drop_duplicates()
    out = keys.merge(out, on=[PATIENT_ID_COL, LINE_COL], how="left")
    return out


def build_tumor_marker_features(ctx) -> ModalityFrame:
    tumor_markers = pd.read_csv(ctx.data_dir / TUMOR_MARKER_FILE, sep="\t")
    # Preserve original row order so same-day duplicates can be deterministically resolved
    # by selecting the last recorded measurement for that day.
    tumor_markers["_ROW_ID"] = np.arange(tumor_markers.shape[0], dtype=int)
    tumor_markers = filter_and_sort_input(tumor_markers, ctx.type_specific_patients)

    short_window = int(ctx.config["MARKER_SHORT_WINDOW_DAYS"])
    long_window = int(ctx.config["MARKER_LONG_WINDOW_DAYS"])

    # filter out only CEA and CA15-3 results
    cols = [PATIENT_ID_COL, START_DATE_COL, TEST_COL, RESULT_COL, "_ROW_ID"]
    # if there is NA in any of these columns, create a warning, then drop
    if tumor_markers.loc[:, cols].isna().any().any():
        print("Warning: NA values found in tumor marker measurements")
    tumor_markers = tumor_markers.loc[:, cols].dropna()
    test_map = {"CA15_3": "CA15-3 (U/mL)", "CEA": "CEA (ng/mL)"}
    tumor_markers = tumor_markers[
        tumor_markers[TEST_COL].isin(set(test_map.values()))
    ].copy()
    tumor_markers[START_DATE_COL] = tumor_markers[START_DATE_COL].astype(int)
    tumor_markers[RESULT_COL] = tumor_markers[RESULT_COL].astype(float)

    # Aggregate per day: keep the last recorded measurement for each (patient, date, test).
    # This avoids unstable behavior when there are multiple measurements on the same day
    # (previously, *_DELTA_LAST could flip sign depending on tie-order).
    tumor_markers = tumor_markers.sort_values(
        [PATIENT_ID_COL, START_DATE_COL, TEST_COL, "_ROW_ID"],
        kind="mergesort",
    )
    tumor_markers = tumor_markers.drop_duplicates(
        subset=[PATIENT_ID_COL, START_DATE_COL, TEST_COL],
        keep="last",
    ).drop(columns=["_ROW_ID"])

    features = aggregate_biomarkers(
        tumor_markers=tumor_markers,
        patient_lot_info=ctx.patient_lot_info,
        test_map=test_map,
        short_window=short_window,
        long_window=long_window,
        tau_short=float(ctx.config["MARKER_TAU_SHORT"]),
        tau_long=float(ctx.config["MARKER_TAU_LONG"]),
    )

    features = _impute_tumor_marker_features(features)

    # drop monitoring patterns, i.e. the _COUNT columns
    drop_cols = [col for col in features.columns if col.endswith("_COUNT")]
    features = features.drop(columns=drop_cols)

    features = features.sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)
    return ModalityFrame(name="TUMOR_MARKERS", frame=features, temporal=True)
