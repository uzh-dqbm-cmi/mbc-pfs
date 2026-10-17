from __future__ import annotations

from typing import Dict, FrozenSet, Iterable

import numpy as np
import pandas as pd
from genie_validation.mapping import canonical_regimen
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from config import MSK_DATA_DIR as MSK_DIR


SUBTYPES = ["HR+/HER2-", "HR-/HER2+", "TNBC", "HR+/HER2+"]

GNRH_AGENTS = frozenset({"LEUPROLIDE", "GOSERELIN", "TRIPTORELIN", "HISTRELIN",
                         "DEGARELIX", "RELUGOLIX"})
COMBO_PRODUCTS: Dict[str, FrozenSet[str]] = {
    "HYALURONIDASE-TRASTUZUMAB": frozenset({"TRASTUZUMAB"}),
    "TRASTUZUMAB/HYALURONIDASE-OYSK": frozenset({"TRASTUZUMAB"}),
    "HYALURONIDASE/PERTUZUMAB/TRASTUZUMAB": frozenset({"PERTUZUMAB", "TRASTUZUMAB"}),
    "PERTUZUMAB-TRASTUZUMAB-HYALURONIDASE-ZZXF": frozenset({"PERTUZUMAB", "TRASTUZUMAB"}),
}
AI = frozenset({"LETROZOLE", "ANASTROZOLE", "EXEMESTANE"})
CDK46 = frozenset({"PALBOCICLIB", "RIBOCICLIB", "ABEMACICLIB"})
SERD_SERM = frozenset({"FULVESTRANT", "TAMOXIFEN", "TOREMIFENE", "RALOXIFENE", "ELACESTRANT", "MEGESTROL"})
ENDOCRINE = AI | SERD_SERM


def canonical_agents(names: Iterable[object]) -> FrozenSet[str]:
    out = set()
    for a in canonical_regimen(names):
        out |= COMBO_PRODUCTS.get(a, frozenset({a}))
    return frozenset(out)


def strip_gnrh(agents: FrozenSet[str]) -> FrozenSet[str]:
    return frozenset(a for a in agents if a not in GNRH_AGENTS)


def subtype_label(hr: pd.Series, her2: pd.Series) -> pd.Series:
    hr = pd.to_numeric(hr).astype(int)
    her2 = pd.to_numeric(her2).astype(int)
    out = np.where(hr.eq(1) & her2.eq(0), "HR+/HER2-",
          np.where(hr.eq(1) & her2.eq(1), "HR+/HER2+",
          np.where(hr.eq(0) & her2.eq(1), "HR-/HER2+", "TNBC")))
    return pd.Series(out, index=hr.index)


BCA_SUBTYPE_MAP = {"HR+, HER2-": "HR+/HER2-", "HR+, HER2+": "HR+/HER2+",
                   "HR-, HER2+": "HR-/HER2+", "Triple Negative": "TNBC"}


def load_design_matrix() -> pd.DataFrame:
    dm = pd.read_csv("data/design_matrix.csv",
                     usecols=["PATIENT_ID", "LINE", "LINE_START", "PFS_TIME_DAYS", "PFS_EVENT", "HR", "HER2"])
    dm["SUBTYPE"] = subtype_label(dm.HR, dm.HER2)
    return dm


def load_breast_pfs() -> pd.DataFrame:
    return pd.read_csv("data/BREAST_pfs.csv")


def load_dropped_pfs_lines() -> pd.DataFrame:
    return pd.read_csv("data/BREAST_dropped_pfs_lines.csv")


def load_msk_treatment(patient_ids: Iterable[str] | None = None) -> pd.DataFrame:
    tx = pd.read_csv(MSK_DIR / "data_timeline_treatment.tsv", sep="\t", dtype=str)
    if patient_ids is not None:
        tx = tx[tx.PATIENT_ID.isin(set(patient_ids))]
    tx = tx[tx.AGENT.notna()].copy()
    tx["START"] = tx.START_DATE.astype(int)
    tx["STOP"] = pd.to_numeric(tx.STOP_DATE, errors="coerce").fillna(tx.START + 1).astype(int)
    tx["AGENTS"] = tx.AGENT.map(lambda a: canonical_agents([a]))
    return tx[["PATIENT_ID", "START", "STOP", "SUBTYPE", "AGENTS"]].reset_index(drop=True)


def load_msk_progression(patient_ids: Iterable[str] | None = None) -> pd.DataFrame:
    pr = pd.read_csv(MSK_DIR / "data_timeline_progression.tsv", sep="\t", dtype=str)
    if patient_ids is not None:
        pr = pr[pr.PATIENT_ID.isin(set(patient_ids))]
    pr = pr.assign(DAY=pr.START_DATE.astype(int),
                   PROGRESSION=pr.PROGRESSION.str.strip(),
                   PROB=pd.to_numeric(pr.NLP_PROGRESSION_PROBABILITY, errors="coerce"))
    return pr[["PATIENT_ID", "DAY", "PROGRESSION", "PROB", "PROCEDURE_TYPE"]].sort_values(
        ["PATIENT_ID", "DAY"], kind="mergesort").reset_index(drop=True)


def km(time, event) -> dict:
    t = pd.Series(time).astype(float).to_numpy()
    e = pd.Series(event).astype(int).to_numpy()
    if len(t) == 0:
        return {"n": 0, "events": 0, "median": np.nan, "lo": np.nan, "hi": np.nan}
    fit = KaplanMeierFitter().fit(t, e)
    ci = median_survival_times(fit.confidence_interval_)
    return {"n": len(t), "events": int(e.sum()), "median": float(fit.median_survival_time_),
            "lo": float(ci.iloc[0, 0]), "hi": float(ci.iloc[0, 1])}
