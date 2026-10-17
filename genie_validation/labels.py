from __future__ import annotations

import pandas as pd

from build_pfs import apply_censor
from config import MIN_PFS_DAYS
from . import mapping as M
from . import paths

LINE_SOURCE = "genie_pfs_i"


def _read_cohort_patients() -> pd.Index:
    surv = pd.read_csv(paths.SURVIVAL_FILE, sep="\t", comment="#")
    cohort = surv.loc[surv["PFS_I_ADV_STATUS"].notna(), "PATIENT_ID"].unique()
    return pd.Index(cohort, name="PATIENT_ID")


def candidate_regimens() -> pd.DataFrame:
    reg = pd.read_csv(paths.REGIMEN_FILE, low_memory=False).rename(
        columns={"record_id": "PATIENT_ID", "ca_seq": "CA_SEQ",
                 "dx_reg_start_int": "LINE_START", "regimen_drugs": "REGIMEN_DRUGS"}
    )
    cancer = pd.read_csv(paths.CANCER_FILE, usecols=["record_id", "ca_seq", "ca_type"]).rename(
        columns={"record_id": "PATIENT_ID", "ca_seq": "CA_SEQ"}
    )
    merged = reg.merge(cancer, on=["PATIENT_ID", "CA_SEQ"], how="left")

    keep = (
        merged["PATIENT_ID"].isin(_read_cohort_patients())
        & merged["ca_type"].eq("Breast Cancer")
        & merged["LINE_START"].notna()
        & merged["pfs_i_g_status"].isin([0.0, 1.0])
        & merged["tt_pfs_i_g_days"].notna()
    )

    merged = merged.loc[keep].sort_values(["PATIENT_ID", "LINE_START"], kind="mergesort")
    merged["LINE"] = merged.groupby("PATIENT_ID").cumcount() + 1
    return merged


def clean_outcomes(out: pd.DataFrame) -> pd.DataFrame:
    return apply_censor(out[out["PFS_TIME_DAYS"] >= MIN_PFS_DAYS])


def build_labels(apply_drug_filter: bool = True) -> pd.DataFrame:
    merged = candidate_regimens()
    out = pd.DataFrame(
        {
            "PATIENT_ID": merged["PATIENT_ID"].astype(str),
            "CA_SEQ": merged["CA_SEQ"].astype("Int64"),
            "LINE_START": merged["LINE_START"].astype(int),
            "PFS_TIME_DAYS": merged["tt_pfs_i_g_days"].astype(float),
            "PFS_EVENT": merged["pfs_i_g_status"].astype(int),
            "REGIMEN_DRUGS": merged["REGIMEN_DRUGS"].astype("string"),
            "LINE_SOURCE": LINE_SOURCE,
            "LINE": merged["LINE"],
        }
    )

    out = clean_outcomes(out)

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
