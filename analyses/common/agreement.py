from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import ADMIN_CENSOR_DAYS


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.nan
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    return 2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def kappa(x: np.ndarray, y: np.ndarray) -> float:
    return cohen_kappa_score(x, y)


def censor_730(t: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, e = np.asarray(t, float).copy(), np.asarray(e, int).copy()
    late = t > ADMIN_CENSOR_DAYS
    t[late], e[late] = ADMIN_CENSOR_DAYS, 0
    return t, e


def reference_regimens(regimens: pd.DataFrame) -> pd.DataFrame:
    ref = regimens[regimens["LABELLED"] & ~regimens["IS_LHRH"]].copy()
    ref = ref.sort_values(["PATIENT_ID", "START", "regimen_number"]).reset_index(drop=True)
    ref["B_ID"] = np.arange(len(ref))
    return ref


def nearest_abs(a: pd.DataFrame, a_col: str, b: pd.DataFrame, b_col: str) -> np.ndarray:
    by = {pid: np.sort(grp[b_col].to_numpy(float)) for pid, grp in b.groupby("PATIENT_ID")}
    out = np.full(len(a), np.inf)
    for i, (pid, t) in enumerate(zip(a["PATIENT_ID"], a[a_col])):
        arr = by.get(pid)
        if arr is not None and len(arr):
            out[i] = np.abs(arr - t).min()
    return out


def greedy_match(a: pd.DataFrame, b: pd.DataFrame, tol: int) -> pd.DataFrame:
    out = []
    b_by = {pid: grp for pid, grp in b.groupby("PATIENT_ID")}
    for pid, ga in a.groupby("PATIENT_ID"):
        gb = b_by.get(pid)
        if gb is None:
            continue
        cand = sorted((abs(ta - tb), ta, ia, ib, tb - ta)
                      for ia, ta in zip(ga["A_ID"], ga["A_START"])
                      for ib, tb in zip(gb["B_ID"], gb["B_START"]) if abs(ta - tb) <= tol)
        used_a, used_b = set(), set()
        for _, _, ia, ib, delta in cand:
            if ia in used_a or ib in used_b:
                continue
            used_a.add(ia), used_b.add(ib)
            out.append({"PATIENT_ID": pid, "A_ID": ia, "B_ID": ib, "DELTA_B_MINUS_A": delta})
    return pd.DataFrame(out, columns=["PATIENT_ID", "A_ID", "B_ID", "DELTA_B_MINUS_A"])
