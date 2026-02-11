# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---
"""Unified PFS construction logic with selectable LoT split policy.

Option A (default): only documented progressions (Y) or censored Ns drive LoT changes.
Forced day-split transitions without follow-up are excluded from PFS (PFS_EVENT=-1).

Option B: forced day-split transitions are treated as observed progression events
(`PFS_EVENT=1`) even when no assessments exist.
"""
import re
from collections import defaultdict
from pathlib import Path

# %%
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import RADIOLOGY_REPORT_WINDOW_DAYS
from utils import _to_float, _to_int

PRIOR_MEASUREMENT_FILES = [
    "data_timeline_tumor_sites.tsv",
    "data_timeline_cancer_presence.tsv",
    "data_timeline_progression.tsv",
]

# Populated by the pipeline to gate LoTs without recent radiology evidence.
prior_measurement_map: Optional[Dict[str, np.ndarray]] = None
investigative_treatment_cache: Optional[pd.DataFrame] = None


def _next_treatment_index(
    start_dates: np.ndarray,
    current_idx: int,
    target_day: int,
    inclusive: bool,
    agent_labels: np.ndarray,
    require_new_agent: bool,
) -> int:
    """
    Return the index of the first treatment on/after (or strictly after) target_day.

    If require_new_agent is True, skip treatments that share the same AGENT as the
    most recent treatment in the current LoT (i.e., the latest treatment before the
    candidate) and return the first index with a different AGENT.
    """
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


def _has_prior_measurement(
    measurement_map: Optional[Dict[str, np.ndarray]],
    pid: str,
    line_start: int,
) -> bool:
    """Check whether radiology data exists within 90 days prior to line start."""
    if not measurement_map:
        return True
    days = measurement_map.get(pid)
    if days is None or days.size == 0:
        return False
    window_start = line_start - RADIOLOGY_REPORT_WINDOW_DAYS
    # index which, if present, points to the first measurement on/after window_start
    # because window_start <= days[i]
    left = int(np.searchsorted(days, window_start, side="left"))
    # index which, if present, points to the first measurement strictly after line_start
    # because line_start < days[j]
    right = int(np.searchsorted(days, line_start, side="right"))
    # if j > i, at least one measurement exists in [window_start, line_start]
    return right > left


def assign_patient_lines_and_pfs(
    patient_treatments: pd.DataFrame,
    patient_progression: pd.DataFrame,
    day_split: int,
    grace_period: int,
    death_time: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Assign LoTs and PFS for a single patient by scanning treatments and progression events once.

    Rules:
      * Use the first qualifying 'Y' (≥ line_start + grace) as a PFS event; the next treatment
        on/after that date starts the following LoT.
      * If only 'N' assessments are observed, censor at the last 'N' and advance the LoT with the
        next treatment strictly after that date.
      * If there are no 'N' observations at all, force a new LoT only when a later treatment
        begins ≥ day_split days after the current LoT start; otherwise fall back to death or
        same-day censoring for the terminal line. The post-threshold LoT starts at the first
        treatment whose SUBTYPE differs from the most recent treatment in the current line.
    """

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
        # find index of the immediate next treatment day_split days away
        next_treatment_idx_day_split_away = _next_treatment_index(
            start_dates=start_dates,
            current_idx=line_start_idx,
            target_day=new_lot_start_after_day,
            inclusive=True,
            agent_labels=agent_labels,
            require_new_agent=True,
        )
        # day of the next treatment, this is the upper bound to find a progression
        # event in
        boundary_day = (
            int(start_dates[next_treatment_idx_day_split_away])
            if next_treatment_idx_day_split_away < n_treat
            else None
        )

        # Skip events that occur before the current LoT start.
        search_ptr = event_ptr
        while search_ptr < n_events and event_dates[search_ptr] <= line_start_date:
            search_ptr += 1

        # now the search_ptr is at the first event on/after line_start_date
        ptr_after_line = search_ptr
        last_n = None
        pfs_event = None
        event_day = None
        original_pfs_time_days = None
        original_pfs_event = None

        # when subsequent observations are available
        # Traverse consecutive 'N' assessments until a qualifying 'Y' is seen or data ends.
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

            # progression_next_idx is the index of next treatment immediately after progression
            if progression_next_idx > line_start_idx:
                window_start = event_day_int - grace_period
                window_mask = (
                    start_dates[line_start_idx:progression_next_idx] > window_start
                )
                if window_mask.any():
                    first_in_window = int(np.argmax(window_mask)) + line_start_idx
                    # Progression occurred within the grace window of at least one recent treatment;
                    # start the earliest such treatment as the next LoT.
                    next_idx = first_in_window

            pfs_time = event_day_int - line_start_date
        else:
            if last_n is not None:  # observed documented non-progression
                next_idx = next_treatment_idx_day_split_away
                event_day = int(last_n)
                pfs_event = 0
                pfs_time = event_day - line_start_date
                line_source = "labelled_by_non_progression"
            else:  # no progress reports in current LoT regime
                next_idx = next_treatment_idx_day_split_away
                event_day = line_start_date
                pfs_time = 0
                pfs_event = -1
                line_source = "no_radiology_follow_up"

        lines[line_start_idx:next_idx] = current_line
        # if this is already the last line
        if next_idx == n_treat and pfs_event <= 0:
            # check again if a death time exists, to convert censoring to event
            if death_time is not None:
                if death_time < line_start_date:
                    raise ValueError("Death time before line start date.")
                event_day = int(death_time)
                pfs_event = 1
                pfs_time = event_day - line_start_date
                line_source = "labelled_by_death_event"

        has_prior_measurement = _has_prior_measurement(
            prior_measurement_map,
            pid,
            line_start_date,
        )
        if not has_prior_measurement:
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

    # Inflate treatment rows so therapies that continue past a later line start
    # are visible in that subsequent line as well.
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
    """
    Build ordered measurement histories for each patient based on tumor-site,
    cancer-presence, and progression timelines. Any entry in these sources counts as
    radiographic evidence for prior-imaging checks.
    """

    if not patient_ids:
        return {}

    def _load_measurements(filename: str) -> pd.DataFrame:
        if filename == "data_timeline_progression.tsv" and progression_df is not None:
            df = progression_df.copy()
        else:
            path = DATA_DIR / filename
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


def first_metastasis_start_dates(
    patient_ids: Set[str],
) -> Dict[str, int]:
    """
    Return the earliest tumor-site start date per patient, excluding nonspecific
    sites ("OTHER", "LYMPH NODES").
    """
    if not patient_ids:
        return {}

    df = _norm(
        pd.read_csv(DATA_DIR / "data_timeline_tumor_sites.tsv", sep="\t", dtype=str)
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


def censor_pre_metastasis_lines(
    pfs: pd.DataFrame,
    metastasis_start_map: Dict[str, int],
    return_mapping: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Set PFS_EVENT=-1 and LINE_SOURCE='prior_to_metastasis' for lines starting
    before the first metastasis date for the patient.
    """
    pfs = pfs.copy()
    if not metastasis_start_map:
        return (pfs, None) if return_mapping else pfs

    pfs["ORIGINAL_LINE"] = pfs["LINE"]
    meta_start = pfs["PATIENT_ID"].map(metastasis_start_map)
    mask = meta_start.notna() & (pfs["LINE_START"] < meta_start)
    mapping: Optional[pd.DataFrame] = None
    if mask.any():
        pfs.loc[mask, "PFS_EVENT"] = -1
        pfs.loc[mask, "LINE_SOURCE"] = "prior_to_metastasis"
        pfs = pfs[pfs["LINE_SOURCE"] != "prior_to_metastasis"]
        pfs = pfs.reset_index(drop=True)
        pfs["LINE"] = pfs.groupby("PATIENT_ID", observed=True).cumcount() + 1
        if return_mapping:
            mapping = pfs[["PATIENT_ID", "ORIGINAL_LINE", "LINE"]].rename(
                columns={"ORIGINAL_LINE": "OLD_LINE", "LINE": "NEW_LINE"}
            )
    pfs = pfs.drop(columns=["ORIGINAL_LINE"])
    if return_mapping:
        return pfs, mapping
    return pfs


def remap_treatment_lines(
    treatment: pd.DataFrame, line_map: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Filter/renumber treatment lines to mirror metastasis censoring applied to PFS.

    Parameters
    ----------
    treatment : pd.DataFrame
        Treatment dataframe with columns including PATIENT_ID, LINE, START_DATE.
    line_map : Optional[pd.DataFrame]
        Mapping of PATIENT_ID, OLD_LINE -> NEW_LINE for retained lines.
    """
    if line_map is None or line_map.empty:
        return treatment

    aligned = treatment.merge(
        line_map,
        how="inner",
        left_on=["PATIENT_ID", "LINE"],
        right_on=["PATIENT_ID", "OLD_LINE"],
    )
    aligned = aligned.drop(columns=["LINE", "OLD_LINE"])
    aligned = aligned.rename(columns={"NEW_LINE": "LINE"})
    aligned["LINE"] = aligned["LINE"].astype(int)
    if "START_DATE" in aligned.columns:
        aligned = aligned.sort_values(["PATIENT_ID", "LINE", "START_DATE"])
    else:
        aligned = aligned.sort_values(["PATIENT_ID", "LINE"])
    return aligned.reset_index(drop=True)


# %%
def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize index column names and types."""
    assert "PATIENT_ID" in df.columns
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    if "START_DATE" in df.columns:
        df.loc[:, "START_DATE"] = _to_int(df["START_DATE"])
    if "STOP_DATE" in df.columns:
        df.loc[:, "STOP_DATE"] = _to_float(df["STOP_DATE"])
    return df


DATA_DIR = Path("data/msk_chord_2024")


def prep_diagnosis() -> Tuple[pd.DataFrame, set[str]]:
    diag = _norm(
        pd.read_csv(DATA_DIR / "data_timeline_diagnosis.tsv", sep="\t", dtype=str)
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

    # Clean/normalize
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
    # in "SUMMARY", set N/A, Unknown/Unstaged, and Unstaged  unknown  to NA
    diag["SUMMARY"] = diag["SUMMARY"].replace(
        to_replace=["N/A", "Unknown/Unstaged", "Unstaged unknown", "Unstaged  unknown"],
        value="NA",
        regex=True,
    )
    CANCER_TYPE = "BREAST"
    # First, get patients who are diagnosed with Breast cancer
    diagnosed_patient_ids = set(diag[diag["SITE"] == CANCER_TYPE]["PATIENT_ID"])
    # filter out patients who were also diagnosed with other cancers
    other_site_patient_ids = set(diag[diag["SITE"] != CANCER_TYPE]["PATIENT_ID"])
    multi_cancer_diagnosis_patients = diagnosed_patient_ids & other_site_patient_ids
    diag = diag[
        diag["PATIENT_ID"].isin(diagnosed_patient_ids - multi_cancer_diagnosis_patients)
    ]
    # # if STAGE_CDM_DERIVED is Stage 1-3, the SUMMARY should not contain "distant"
    # stage_mask = diag["STAGE_CDM_DERIVED"] == "Stage 1-3"
    # distant_mask = diag["SUMMARY"].str.contains("distant", case=False, na=False)
    # inconsistent_mask = stage_mask & distant_mask
    # print(
    #     f"Removing {inconsistent_mask.sum()} inconsistent records where STAGE_CDM_DERIVED is 'Stage 1-3' but SUMMARY contains 'distant'."
    # )
    # diag = diag[~inconsistent_mask]
    assert diag["SITE"].isna().sum() == 0
    TYPE_SPECIFIC_PATIENT_IDS = set(diag["PATIENT_ID"].unique())
    STAGE_IV_PATIENTS = set(
        diag[diag["STAGE_CDM_DERIVED"] == "Stage 4"]["PATIENT_ID"].unique()
    )
    diag.to_csv("data/diagnosis_parsed.csv", index=False)
    return diag, TYPE_SPECIFIC_PATIENT_IDS, STAGE_IV_PATIENTS


def prep_progression(
    type_specific_patient_ids: set,
) -> pd.DataFrame:
    progression = _norm(
        pd.read_csv(DATA_DIR / "data_timeline_progression.tsv", sep="\t", dtype=str)
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
        "data/msk_chord_2024/data_clinical_patient.tsv", sep="\t", dtype=str, skiprows=4
    )[["PATIENT_ID", "CURRENT_AGE_DEID", "OS_MONTHS", "OS_STATUS"]]
    clinical = clinical[clinical["PATIENT_ID"].isin(type_specific_patient_ids)]
    # transform OS_MONTHS to days:
    clinical["OS_DAYS"] = clinical["OS_MONTHS"].astype(float) * 30.437
    # round down to the nearest integer
    clinical["OS_DAYS"] = clinical["OS_DAYS"].astype(int)
    return (
        clinical[clinical["OS_STATUS"] == "1:DECEASED"]
        .set_index("PATIENT_ID")["OS_DAYS"]
        .to_dict()
    )
