from __future__ import annotations

import numpy as np
import pandas as pd

from genie_validation.labels import candidate_regimens, clean_outcomes
from genie_validation.paths import GENIE_DIR as GENIE

PD = "Progressing/Worsening/Enlarging"
MIXED = "Mixed"
NON_PD = ("Stable/No change", "Improving/Responding")

LHRH_AGONISTS = frozenset({"Goserlin Acetate", "Histrelin Acetate", "Leuprolide", "Triptorelin"})
N_DRUG_SLOTS = 5


def short_name(label: str) -> str:
    return label.split("(")[0].strip()


def load_regimens() -> pd.DataFrame:
    reg = pd.read_csv(GENIE / "regimen_cancer_level_dataset.csv", low_memory=False)
    ca = pd.read_csv(GENIE / "cancer_level_dataset_index.csv", low_memory=False)
    reg = reg.merge(ca[["record_id", "ca_seq", "ca_type", "bca_subtype"]], on=["record_id", "ca_seq"], how="left")
    patients = pd.read_csv(GENIE / "patient_level_dataset.csv",
                           usecols=["record_id", "dob_lastalive_int", "hybrid_death_int"])
    reg = reg.merge(patients, on="record_id", how="left")
    reg["start"] = reg[[f"drugs_startdt_int_{k}" for k in range(1, N_DRUG_SLOTS + 1)]].min(axis=1)
    names = reg[[f"drugs_drug_{k}" for k in range(1, N_DRUG_SLOTS + 1)]].to_numpy(object)
    reg["is_lhrh"] = [any(short_name(d) in LHRH_AGONISTS for d in row if isinstance(d, str)) for row in names]
    reg["pfs_eligible"] = reg["pfs_i_g_status"].notna()
    return reg


def add_next_start(reg: pd.DataFrame) -> pd.DataFrame:
    uniq = reg.drop_duplicates(["record_id", "regimen_number"])
    later = {pid: g.sort_values("regimen_number")[["regimen_number", "start"]].to_numpy(float)
             for pid, g in uniq[~uniq["is_lhrh"]].groupby("record_id")}

    def nxt(row):
        arr = later.get(row.record_id)
        if arr is None:
            return np.nan
        after = arr[arr[:, 0] > row.regimen_number, 1]
        return after.min() if len(after) else np.nan

    reg = reg.copy()
    reg["next_start"] = reg.apply(nxt, axis=1)
    return reg


def load_imaging() -> pd.DataFrame:
    img = pd.read_csv(GENIE / "imaging_level_dataset.csv", low_memory=False)
    return img.rename(columns={"image_scan_int": "day", "image_overall": "status"})[["record_id", "day", "status"]]


def days_by_patient(reports: pd.DataFrame, statuses) -> dict[str, np.ndarray]:
    sel = reports[reports["status"].isin(statuses)]
    return {pid: np.sort(g["day"].to_numpy(float)) for pid, g in sel.groupby("record_id")}


EMPTY = np.array([], float)

def derive(start, next_start, death, last_alive, img_days):
    hi = np.inf if np.isnan(next_start) else next_start
    days = img_days[(img_days > start) & (img_days < hi)]
    ev = days.min() if len(days) else np.nan
    candidates = [t for t in (ev, death if death <= hi else np.nan) if not np.isnan(t)]
    if candidates:
        return 1, min(candidates) - start
    return 0, min(hi, last_alive) - start


def mixed_endpoints(reg, reports=None, columns=("mixed_event", "mixed_time")):
    rows = reg[reg.pfs_eligible].copy()
    days = days_by_patient(load_imaging() if reports is None else reports, [PD, MIXED])
    rows[list(columns)] = [derive(r.start, np.nan if r.is_lhrh else r.next_start, r.hybrid_death_int,
                                  r.dob_lastalive_int, days.get(r.record_id, EMPTY)) for r in rows.itertuples()]
    return rows


def validation_labels(lines, rows):
    keys = ["PATIENT_ID", "CA_SEQ", "regimen_number"]
    out = lines.merge(candidate_regimens()[["PATIENT_ID", "LINE", "CA_SEQ", "regimen_number"]],
                      on=["PATIENT_ID", "LINE"], how="left", validate="one_to_one")
    out = out.merge(rows.rename(columns={"record_id": "PATIENT_ID", "ca_seq": "CA_SEQ"})[
        keys + ["mixed_time", "mixed_event"]], on=keys, how="left", validate="one_to_one")
    if out.mixed_event.isna().any():
        raise ValueError("A validation line has no released regimen row.")
    return clean_outcomes(out.assign(PFS_TIME_DAYS=out.mixed_time.astype(float),
                                     PFS_EVENT=out.mixed_event.astype(int)))
