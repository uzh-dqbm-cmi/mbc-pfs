import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import (
    ADMIN_CENSOR_DAYS,
    CANCER_TYPE,
    EVENT_COL,
    MIN_PFS_DAYS,
    MONTH_DAYS,
    MSK_DATA_DIR,
    PATIENT_ID_COL,
    RADIOLOGY_REPORT_WINDOW_DAYS,
    TIME_COL,
)
from utils import _to_float, _to_int

PRIOR_MEASUREMENT_FILES = [
    "data_timeline_tumor_sites.tsv",
    "data_timeline_cancer_presence.tsv",
    "data_timeline_progression.tsv",
]

prior_measurement_map: Optional[Dict[str, np.ndarray]] = None


def _next_treatment_index(
    start_dates: np.ndarray,
    current_idx: int,
    target_day: int,
    inclusive: bool,
    agent_labels: np.ndarray,
    require_new_agent: bool,
) -> int:
    side = "left" if inclusive else "right"
    idx = int(np.searchsorted(start_dates, target_day, side=side))
    min_allowed = current_idx + 1
    if idx < min_allowed:
        idx = min_allowed
    if idx > len(start_dates):
        idx = len(start_dates)
    if require_new_agent and idx < len(start_dates) and agent_labels is not None:
        last_idx = max(min(idx - 1, len(agent_labels) - 1), current_idx)
        last_agent = agent_labels[last_idx]
        while idx < len(start_dates) and agent_labels[idx] == last_agent:
            idx += 1
    return idx


def has_prior_measurement(
    measurement_map: Optional[Dict[str, np.ndarray]],
    pid: str,
    line_start: int,
    window_days: int = RADIOLOGY_REPORT_WINDOW_DAYS,
) -> bool:
    if not measurement_map:
        return True
    days = measurement_map.get(pid)
    if days is None or days.size == 0:
        return False
    window_start = line_start - window_days
    left = int(np.searchsorted(days, window_start, side="left"))
    right = int(np.searchsorted(days, line_start, side="right"))
    return right > left


def assign_patient_lines_and_pfs(
    patient_treatments: pd.DataFrame,
    patient_progression: pd.DataFrame,
    day_split: int,
    grace_period: int,
    death_time: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    patient_treatments = patient_treatments.copy()
    patient_treatments = patient_treatments.sort_values(
        "START_DATE", kind="mergesort"
    ).reset_index(drop=True)
    start_dates = patient_treatments["START_DATE"].to_numpy(dtype=int)
    n_treat = len(start_dates)
    agent_labels = patient_treatments.get("AGENT")

    events = patient_progression.copy()
    event_dates = (
        events["START_DATE"].to_numpy(dtype=int)
        if not events.empty
        else np.array([], dtype=int)
    )
    event_types = (
        events["PROGRESSION"].to_numpy(dtype="U1")
        if not events.empty
        else np.array([], dtype="U1")
    )
    n_events = len(event_dates)
    lines = np.empty(n_treat, dtype=np.int32)
    pfs_rows: List[Dict[str, Any]] = []
    pid = patient_treatments["PATIENT_ID"].iat[0]

    event_ptr = 0
    idx = 0
    current_line = 1
    while idx < n_treat:
        line_start_idx = idx
        line_start_date = int(start_dates[line_start_idx])
        new_lot_start_after_day = line_start_date + day_split
        line_source = "undefined"
        next_treatment_idx_day_split_away = _next_treatment_index(
            start_dates=start_dates,
            current_idx=line_start_idx,
            target_day=new_lot_start_after_day,
            inclusive=True,
            agent_labels=agent_labels,
            require_new_agent=True,
        )
        boundary_day = (
            int(start_dates[next_treatment_idx_day_split_away])
            if next_treatment_idx_day_split_away < n_treat
            else None
        )

        search_ptr = event_ptr
        while search_ptr < n_events and event_dates[search_ptr] <= line_start_date:
            search_ptr += 1

        ptr_after_line = search_ptr
        last_n = None
        pfs_event = None
        event_day = None
        original_pfs_time_days = None
        original_pfs_event = None

        while ptr_after_line < n_events and (
            boundary_day is None or event_dates[ptr_after_line] <= boundary_day
        ):
            evt_day = int(event_dates[ptr_after_line])
            evt_type = event_types[ptr_after_line]
            if evt_type == "N":
                last_n = evt_day
                ptr_after_line += 1
                continue
            elif evt_type == "Y":
                if evt_day >= line_start_date + grace_period:
                    event_day = evt_day
                    pfs_event = 1
                    line_source = "labelled_by_progression_event"
                    ptr_after_line += 1
                    break
                ptr_after_line += 1
            else:
                raise ValueError(f"Unexpected PROGRESSION value: {evt_type}")

        event_ptr = ptr_after_line

        if pfs_event == 1:
            event_day_int = int(event_day)
            progression_next_idx = _next_treatment_index(
                start_dates,
                line_start_idx,
                event_day_int,
                inclusive=True,
                agent_labels=agent_labels,
                require_new_agent=True,
            )
            next_idx = progression_next_idx

            if progression_next_idx > line_start_idx:
                window_start = event_day_int - grace_period
                window_mask = (
                    start_dates[line_start_idx:progression_next_idx] > window_start
                )
                if window_mask.any():
                    first_in_window = int(np.argmax(window_mask)) + line_start_idx
                    next_idx = first_in_window

            pfs_time = event_day_int - line_start_date
        else:
            if last_n is not None:
                next_idx = next_treatment_idx_day_split_away
                event_day = int(last_n)
                pfs_event = 0
                pfs_time = event_day - line_start_date
                line_source = "labelled_by_non_progression"
            else:
                next_idx = next_treatment_idx_day_split_away
                event_day = line_start_date
                pfs_time = 0
                pfs_event = -1
                line_source = "no_radiology_follow_up"

        lines[line_start_idx:next_idx] = current_line
        if next_idx == n_treat and pfs_event <= 0:
            if death_time is not None:
                if death_time < line_start_date:
                    raise ValueError("Death time before line start date.")
                event_day = int(death_time)
                pfs_event = 1
                pfs_time = event_day - line_start_date
                line_source = "labelled_by_death_event"

        if not has_prior_measurement(prior_measurement_map, pid, line_start_date):
            original_pfs_event = pfs_event
            original_pfs_time_days = pfs_time
            pfs_event = -1
            line_source = f"no_radiology_within_{RADIOLOGY_REPORT_WINDOW_DAYS}_prior"

        pfs_rows.append(
            {
                "PATIENT_ID": pid,
                "LINE": current_line,
                "LINE_START": line_start_date,
                "EVENT_DAY": int(event_day),
                "PFS_TIME_DAYS": int(max(0, pfs_time)),
                "PFS_EVENT": int(pfs_event),
                "LINE_SOURCE": line_source,
                "ORIGINAL_PFS_TIME_DAYS": original_pfs_time_days,
                "ORIGINAL_PFS_EVENT": original_pfs_event,
            }
        )

        if next_idx >= n_treat:
            break

        idx = next_idx
        current_line += 1

    patient_treatments["LINE"] = lines

    if "STOP_DATE" in patient_treatments.columns and pfs_rows:
        line_starts = pd.DataFrame(pfs_rows)[["LINE", "LINE_START"]]
        inflated_rows: List[pd.Series] = []
        for _, row in patient_treatments.iterrows():
            stop_day = row.get("STOP_DATE")
            if pd.isna(stop_day):
                continue
            current_line = int(row["LINE"])
            later_lines = line_starts[line_starts["LINE"] > current_line]
            if later_lines.empty:
                continue
            overlapping_lines = later_lines[later_lines["LINE_START"] < stop_day][
                "LINE"
            ]
            for new_line in overlapping_lines:
                dup = row.copy()
                dup["LINE"] = int(new_line)
                inflated_rows.append(dup)
        if inflated_rows:
            patient_treatments = pd.concat(
                [patient_treatments, pd.DataFrame(inflated_rows)], ignore_index=True
            )
            patient_treatments["LINE"] = patient_treatments["LINE"].astype(lines.dtype)
            patient_treatments = patient_treatments.sort_values(
                ["START_DATE", "LINE"]
            ).reset_index(drop=True)

    return patient_treatments, pfs_rows


def measurement_days(
    patient_ids: Set[str], progression_df: Optional[pd.DataFrame] = None
) -> Dict[str, np.ndarray]:
    if not patient_ids:
        return {}

    def _load_measurements(filename: str) -> pd.DataFrame:
        if filename == "data_timeline_progression.tsv" and progression_df is not None:
            df = progression_df.copy()
        else:
            path = MSK_DATA_DIR / filename
            df = pd.read_csv(
                path, sep="\t", dtype=str, usecols=["PATIENT_ID", "START_DATE"]
            )
        df = df[df["PATIENT_ID"].isin(patient_ids)].copy()
        df["START_DATE"] = pd.to_numeric(df["START_DATE"], errors="coerce")
        return df.dropna(subset=["START_DATE"])

    def _append_measurements(df: pd.DataFrame, bucket: Dict[str, List[int]]) -> None:
        if df.empty:
            return
        grouped = df.groupby("PATIENT_ID", observed=True)["START_DATE"]
        for pid, dates in grouped:
            bucket[pid].extend(dates.to_numpy(dtype=np.int64).tolist())

    radiology_map: Dict[str, List[int]] = defaultdict(list)
    for filename in PRIOR_MEASUREMENT_FILES:
        df = _load_measurements(filename)
        _append_measurements(df, radiology_map)

    global prior_measurement_map
    prior_measurement_map = {
        pid: np.sort(np.asarray(days, dtype=np.int64))
        for pid, days in radiology_map.items()
        if days
    }
    return prior_measurement_map


def after_metastasis(
    df: pd.DataFrame, stage_iv_patients: set, first_metastasis: Dict[str, int]
) -> pd.DataFrame:
    return df[
        df.apply(
            lambda row: (row[PATIENT_ID_COL] in stage_iv_patients)
            or (
                row["START_DATE"]
                >= first_metastasis.get(row[PATIENT_ID_COL], 100_000_000)
            ),
            axis=1,
        )
    ].reset_index(drop=True)


def first_metastasis_start_dates(
    patient_ids: Set[str],
) -> Dict[str, int]:
    if not patient_ids:
        return {}

    df = _norm(
        pd.read_csv(MSK_DATA_DIR / "data_timeline_tumor_sites.tsv", sep="\t", dtype=str)
    )
    df = df[df["PATIENT_ID"].isin(patient_ids)]
    df = df.dropna(subset=["START_DATE", "TUMOR_SITE"])
    df["TUMOR_SITE"] = df["TUMOR_SITE"].str.strip().str.upper()
    df = df[~df["TUMOR_SITE"].isin({"OTHER", "LYMPH NODES"})]
    if df.empty:
        return {}

    first_dates = (
        df.groupby("PATIENT_ID", observed=True)["START_DATE"].min().dropna().astype(int)
    )
    return first_dates.to_dict()


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    assert "PATIENT_ID" in df.columns
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    if "START_DATE" in df.columns:
        df.loc[:, "START_DATE"] = _to_int(df["START_DATE"])
    if "STOP_DATE" in df.columns:
        df.loc[:, "STOP_DATE"] = _to_float(df["STOP_DATE"])
    return df


def prep_diagnosis() -> Tuple[pd.DataFrame, set[str], set[str]]:
    diag = _norm(
        pd.read_csv(MSK_DATA_DIR / "data_timeline_diagnosis.tsv", sep="\t", dtype=str)
    )
    pattern = re.compile(
        r"""
        ^\s*
        (?P<histologic>[^|]+?)                  # text up to first '|'
        \s*\|\s*
        (?P<site>[^,|()]+?)                     # site up to first comma or '('
        (?:\s*,\s*(?P<subset>.*?))?             # subset = EVERYTHING after first comma up to '(M####/..'
        \s*(?=\(\s*M\d{4}/)                     # lookahead to the ICD-O morph block '(M####/..'
        \(\s*
        (?P<icdo_morph>M\d{4}/[012369](?:[0-4]|9)?)  # ICD-O morphology: mandatory 'M'
        \s*\|\s*
        (?P<icdo_topo>C\d{3})                   # ICD-O topography: exactly Cxxx
        \s*\)\s*$
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    out = diag["DX_DESCRIPTION"].str.extract(pattern)

    out["HISTOLOGIC"] = out["histologic"].str.strip().str.upper()
    out["SITE"] = out["site"].str.strip().str.upper()
    out["CANCER_SITE_SUBSITE"] = out["subset"].str.strip().str.upper()
    out["ICDO_MORPH"] = out["icdo_morph"].str.upper()
    out["ICDO_TOPO"] = out["icdo_topo"].str.upper().str.replace(r"\.", "", regex=True)

    diag = pd.concat(
        [
            diag,
            out[
                ["HISTOLOGIC", "SITE", "CANCER_SITE_SUBSITE", "ICDO_MORPH", "ICDO_TOPO"]
            ],
        ],
        axis=1,
    )
    diag["SUMMARY"] = diag["SUMMARY"].str.strip()
    diag["SUMMARY"] = diag["SUMMARY"].replace(
        to_replace=["N/A", "Unknown/Unstaged", "Unstaged unknown", "Unstaged  unknown"],
        value="NA",
        regex=True,
    )
    diagnosed_patient_ids = set(diag[diag["SITE"] == CANCER_TYPE]["PATIENT_ID"])
    other_site_patient_ids = set(diag[diag["SITE"] != CANCER_TYPE]["PATIENT_ID"])
    multi_cancer_diagnosis_patients = diagnosed_patient_ids & other_site_patient_ids
    diag = diag[
        diag["PATIENT_ID"].isin(diagnosed_patient_ids - multi_cancer_diagnosis_patients)
    ]
    assert diag["SITE"].isna().sum() == 0
    TYPE_SPECIFIC_PATIENT_IDS = set(diag["PATIENT_ID"].unique())
    STAGE_IV_PATIENTS = set(
        diag[diag["STAGE_CDM_DERIVED"] == "Stage 4"]["PATIENT_ID"].unique()
    )
    return diag, TYPE_SPECIFIC_PATIENT_IDS, STAGE_IV_PATIENTS


def prep_progression(
    type_specific_patient_ids: set,
) -> pd.DataFrame:
    progression = _norm(
        pd.read_csv(MSK_DATA_DIR / "data_timeline_progression.tsv", sep="\t", dtype=str)
    )
    progression = progression[progression["PATIENT_ID"].isin(type_specific_patient_ids)]
    progression["PROGRESSION"] = progression["PROGRESSION"].str.strip().str.upper()
    progression = progression[progression["PROGRESSION"].isin(["Y", "N"])]
    assert not progression["START_DATE"].isnull().any()
    progression = progression.sort_values(
        ["PATIENT_ID", "START_DATE"], kind="mergesort"
    )
    return progression


def prep_death_times(
    type_specific_patient_ids: set,
) -> Dict[str, int]:
    clinical = pd.read_csv(
        MSK_DATA_DIR / "data_clinical_patient.tsv", sep="\t", dtype=str, skiprows=4
    )[["PATIENT_ID", "CURRENT_AGE_DEID", "OS_MONTHS", "OS_STATUS"]]
    clinical = clinical[clinical["PATIENT_ID"].isin(type_specific_patient_ids)]
    clinical["OS_DAYS"] = clinical["OS_MONTHS"].astype(float) * MONTH_DAYS
    clinical["OS_DAYS"] = clinical["OS_DAYS"].astype(int)
    return (
        clinical[clinical["OS_STATUS"] == "1:DECEASED"]
        .set_index("PATIENT_ID")["OS_DAYS"]
        .to_dict()
    )


def flag_short_lines(pfs: pd.DataFrame) -> pd.DataFrame:
    pfs = pfs.copy()
    too_short = (pfs[TIME_COL] < MIN_PFS_DAYS) & (pfs[EVENT_COL] != -1)
    pfs.loc[too_short, "LINE_SOURCE"] = "line_too_short_for_effectiveness"
    pfs.loc[too_short, EVENT_COL] = -1
    return pfs


def apply_censor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    longer_mask = (df[TIME_COL] > ADMIN_CENSOR_DAYS) & (df[EVENT_COL] != -1)
    df.loc[longer_mask, TIME_COL] = ADMIN_CENSOR_DAYS
    df.loc[longer_mask, EVENT_COL] = 0
    exact_event_mask = (df[TIME_COL] == ADMIN_CENSOR_DAYS) & (df[EVENT_COL] == 1)
    df.loc[exact_event_mask, TIME_COL] = ADMIN_CENSOR_DAYS - 0.001
    return df
