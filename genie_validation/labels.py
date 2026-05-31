from __future__ import annotations

import pandas as pd

from config import ADMIN_CENSOR_DAYS
from . import mapping as M
from . import paths

MIN_LINE_DAYS = 28
LINE_SOURCE = "genie_pfs_i"


def _read_cohort_patients() -> pd.Index:
    surv = pd.read_csv(paths.SURVIVAL_FILE, sep="\t", comment="#")
    cohort = surv.loc[surv["PFS_I_ADV_STATUS"].notna(), "PATIENT_ID"].unique()
    return pd.Index(cohort, name="PATIENT_ID")


def build_labels(apply_drug_filter: bool = True) -> pd.DataFrame:
    cohort_patients = _read_cohort_patients()

    reg = pd.read_csv(paths.REGIMEN_FILE, low_memory=False)
    cancer = pd.read_csv(paths.CANCER_FILE, low_memory=False)

    cancer_keys = cancer[["record_id", "ca_seq", "ca_type"]].copy()
    merged = reg.merge(cancer_keys, on=["record_id", "ca_seq"], how="left")

    merged = merged[merged["record_id"].isin(cohort_patients)]
    merged = merged[merged["ca_type"].eq("Breast Cancer")]

    line_start = pd.to_numeric(merged["dx_reg_start_int"], errors="coerce")
    status = pd.to_numeric(merged["pfs_i_g_status"], errors="coerce")
    time_days = pd.to_numeric(merged["tt_pfs_i_g_days"], errors="coerce")

    keep = line_start.notna() & status.isin([0.0, 1.0]) & time_days.notna()
    merged = merged.loc[keep].copy()

    out = pd.DataFrame(
        {
            "PATIENT_ID": merged["record_id"].astype(str),
            "CA_SEQ": pd.to_numeric(merged["ca_seq"], errors="coerce").astype("Int64"),
            "LINE_START": line_start.loc[merged.index].astype(int),
            "PFS_TIME_DAYS": time_days.loc[merged.index].astype(float),
            "PFS_EVENT": status.loc[merged.index].astype(int),
            "REGIMEN_DRUGS": merged["regimen_drugs"].astype("string"),
            "LINE_SOURCE": LINE_SOURCE,
        }
    )

    out = out.sort_values(["PATIENT_ID", "LINE_START"], kind="mergesort").reset_index(
        drop=True
    )
    out["LINE"] = out.groupby("PATIENT_ID").cumcount() + 1

    out = out[out["PFS_TIME_DAYS"] >= MIN_LINE_DAYS].copy()
    longer = out["PFS_TIME_DAYS"] > ADMIN_CENSOR_DAYS
    out.loc[longer, "PFS_TIME_DAYS"] = float(ADMIN_CENSOR_DAYS)
    out.loc[longer, "PFS_EVENT"] = 0
    exact = (out["PFS_TIME_DAYS"] == ADMIN_CENSOR_DAYS) & (out["PFS_EVENT"] == 1)
    out.loc[exact, "PFS_TIME_DAYS"] = ADMIN_CENSOR_DAYS - 0.001

    if apply_drug_filter:
        keep_line = out["REGIMEN_DRUGS"].map(
            lambda d: M.line_agent_suffixes(d) is not None
        )
        n_before = len(out)
        out = out[keep_line].reset_index(drop=True)
        print(
            f"  drug filter: kept {len(out)}/{n_before} lines whose drugs are all in "
            f"GENIE_DRUG_TO_AGENT ({n_before - len(out)} dropped for an unknown drug)"
        )

    out["EVENT_DAY"] = out["LINE_START"] + out["PFS_TIME_DAYS"]
    out = out[
        ["PATIENT_ID", "CA_SEQ", "LINE", "LINE_START", "PFS_TIME_DAYS", "PFS_EVENT",
         "EVENT_DAY", "REGIMEN_DRUGS", "LINE_SOURCE"]
    ].reset_index(drop=True)
    return out


def write_labels(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = build_labels()
    paths.PFS_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.PFS_OUT, index=False)
    return df
