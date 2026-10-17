from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from genie_validation import mapping as M
from analyses.common.bpc_data import GENIE, LHRH_AGONISTS, MIXED, N_DRUG_SLOTS, NON_PD, PD

BREAST = "Breast Cancer"
NED_LABEL = "No, the Impression states or implies there is no evidence of cancer"
LHRH_PATTERN = re.compile("|".join(sorted(name.split()[0].lower() for name in LHRH_AGONISTS)), re.I)
INVESTIGATIONAL = "Investigational Drug"
INVESTIGATIVE_AGENT = "INVESTIGATIVE"

EXTRA_CLASS: Dict[str, str] = {
    "TRASTUZUMAB/HYALURONIDASE-OYSK": "BIOLOGIC",
    "PERTUZUMAB-TRASTUZUMAB-HYALURONIDASE-ZZXF": "BIOLOGIC",
    "OTHER HORMONE": "HORMONE",
    "OTHER ANTINEOPLASTIC": "OTHER",
    "OTHER NOS": "OTHER",
    "OLARATUMAB": "BIOLOGIC",
    "YTTRIUM Y90 IBRITUMOMAB TIUXETAN": "OTHER",
    "TASELISIB": "TARGETED",
    "BCG SOLUTION": "OTHER",
    "PACLITAXEL LOADED POLYMERIC MICELLE": "CHEMO",
    "APATINIB": "TARGETED",
    "TEGAFURGIMERACILOTERACIL POTASSIUM": "CHEMO",
    "INVESTIGATIONAL DRUG": "INVESTIGATIONAL",
}
EXTRA_AGENT: Dict[str, str] = {
    "TRASTUZUMAB/HYALURONIDASE-OYSK": "TRASTUZUMAB",
    "PERTUZUMAB-TRASTUZUMAB-HYALURONIDASE-ZZXF": "HYALURONIDASE/PERTUZUMAB/TRASTUZUMAB",
}


def base_drug_name(raw: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\(.*$", "", str(raw)).strip()).upper()


def drug_class(base: str) -> str:
    if base in EXTRA_CLASS:
        return EXTRA_CLASS[base]
    mapped = M.GENIE_DRUG_TO_AGENT.get(base)
    return "OTHER" if mapped is None else mapped.split("_")[0]


def canonical_agent(base: str) -> str:
    return EXTRA_AGENT.get(base, M.canonical_agent(base))


@dataclass(frozen=True)
class Genie:
    regimens_raw: pd.DataFrame
    index_cancers: pd.DataFrame
    non_index_cancers: pd.DataFrame
    imaging_raw: pd.DataFrame
    patients: pd.DataFrame
    anchor: pd.Series


def load() -> Genie:
    reg = pd.read_csv(GENIE / "regimen_cancer_level_dataset.csv", low_memory=False)
    idx = pd.read_csv(GENIE / "cancer_level_dataset_index.csv", low_memory=False)
    non = pd.read_csv(GENIE / "cancer_level_dataset_non_index.csv", low_memory=False)
    img = pd.read_csv(GENIE / "imaging_level_dataset.csv", low_memory=False)
    pat = pd.read_csv(GENIE / "patient_level_dataset.csv", low_memory=False)
    anchor = idx.sort_values("dob_ca_dx_days").groupby("record_id")["dob_ca_dx_days"].first()
    return Genie(reg, idx, non, img, pat, anchor)


def build_meta(g: Genie) -> pd.DataFrame:
    can = g.index_cancers[g.index_cancers["ca_type"].eq(BREAST)].copy()
    can["DX_DAY"] = can["dob_ca_dx_days"] - can["record_id"].map(g.anchor)
    rows = []
    for pid, grp in can.groupby("record_id"):
        grp = grp.sort_values("dob_ca_dx_days")
        first = grp.iloc[0]
        stage4 = grp["stage_dx_iv"].eq("Stage IV")
        dm = grp[grp["dmets_post_dx"].eq(1) & grp["dx_to_dmets_days"].notna()]
        met_post = float((dm["DX_DAY"] + dm["dx_to_dmets_days"]).min()) if len(dm) else np.nan
        met_stage4 = float(grp.loc[stage4, "DX_DAY"].min()) if stage4.any() else np.nan
        hr = M.is_positive(first["ca_bca_er"]) or M.is_positive(first["ca_bca_pr"])
        rows.append({
            "PATIENT_ID": pid, "SITE": first["institution"], "STAGE_IV": bool(stage4.any()),
            "MET_ONSET": met_stage4 if stage4.any() else met_post,
            "HR": int(hr),
        })
    meta = pd.DataFrame(rows)
    death = g.patients.set_index("record_id")["hybrid_death_int"]
    meta["DEATH_DAY"] = meta["PATIENT_ID"].map(death) - meta["PATIENT_ID"].map(g.anchor)
    return meta


def _regimen_rows_with_clock(g: Genie) -> pd.DataFrame:
    cancers = pd.concat(
        [g.index_cancers[["record_id", "ca_seq", "dob_ca_dx_days", "ca_type"]],
         g.non_index_cancers[["record_id", "ca_seq", "dob_ca_dx_days", "ca_type"]]]
    )
    reg = g.regimens_raw.merge(cancers, on=["record_id", "ca_seq"], how="left")
    reg["ANCHOR"] = reg["record_id"].map(g.anchor)
    reg["ASSOC_OFFSET"] = reg["dob_ca_dx_days"] - reg["ANCHOR"]
    start_cols = [f"drugs_startdt_int_{k}" for k in range(1, N_DRUG_SLOTS + 1)]
    reg["START"] = reg[start_cols].min(axis=1) - reg["ANCHOR"]
    reg["IS_LHRH"] = reg["regimen_drugs"].fillna("").str.contains(LHRH_PATTERN)
    reg["IS_INVESTIGATIONAL"] = reg["regimen_drugs"].fillna("").str.contains(INVESTIGATIONAL)
    return reg


def _breast_regimens(reg: pd.DataFrame) -> pd.DataFrame:
    return (reg[reg["ca_type"].eq(BREAST)]
            .sort_values(["record_id", "regimen_number", "pfs_i_g_status"], na_position="last")
            .drop_duplicates(["record_id", "regimen_number"]))


def build_regimens(g: Genie) -> pd.DataFrame:
    reg = _regimen_rows_with_clock(g)
    all_nonlhrh = {pid: np.sort(grp["START"].to_numpy(float))
                   for pid, grp in reg[~reg["IS_LHRH"]].groupby("record_id")}
    breast = _breast_regimens(reg).rename(columns={"record_id": "PATIENT_ID", "institution": "SITE"})
    breast["LABELLED"] = breast["pfs_i_g_status"].notna()

    def next_start(pid: str, s: float) -> float:
        arr = all_nonlhrh.get(pid, np.array([]))
        arr = arr[arr > s]
        return float(arr[0]) if len(arr) else np.nan

    breast["NEXT_NONLHRH_START"] = [next_start(p, s) for p, s in zip(breast["PATIENT_ID"], breast["START"])]
    keep = ["PATIENT_ID", "SITE", "ca_seq", "regimen_number", "regimen_drugs", "START",
            "IS_LHRH", "IS_INVESTIGATIONAL", "LABELLED", "NEXT_NONLHRH_START"]
    return breast[keep].sort_values(["PATIENT_ID", "START", "regimen_number"]).reset_index(drop=True)


def build_drug_courses(g: Genie) -> pd.DataFrame:
    reg = _breast_regimens(_regimen_rows_with_clock(g))
    parts = []
    for k in range(1, N_DRUG_SLOTS + 1):
        x = reg[reg[f"drugs_drug_{k}"].notna()]
        parts.append(pd.DataFrame({
            "PATIENT_ID": x["record_id"].to_numpy(),
            "START_DATE": (x[f"drugs_startdt_int_{k}"] - x["ANCHOR"]).to_numpy(),
            "STOP_DATE": (x[f"dx_drug_end_or_lastadm_int_{k}"] + x["ASSOC_OFFSET"]).to_numpy(),
            "DRUG_RAW": x[f"drugs_drug_{k}"].to_numpy(),
            "REGIMEN_NUMBER": x["regimen_number"].to_numpy(),
            "DRUG_K": k,
        }))
    tx = pd.concat(parts, ignore_index=True).dropna(subset=["START_DATE"])
    tx["BASE"] = tx["DRUG_RAW"].map(base_drug_name)
    tx["AGENT"] = tx["BASE"].map(canonical_agent)
    tx["SUBTYPE"] = tx["BASE"].map(drug_class)
    tx["START_DATE"] = tx["START_DATE"].astype(int)
    return tx.sort_values(["PATIENT_ID", "START_DATE", "REGIMEN_NUMBER", "DRUG_K"],
                          kind="mergesort").reset_index(drop=True)


def build_imaging(g: Genie) -> pd.DataFrame:
    im = g.imaging_raw.copy()
    im["PATIENT_ID"] = im["record_id"]
    im["START_DATE"] = (im["image_scan_int"] - im["record_id"].map(g.anchor)).astype(int)
    im["LABEL"] = np.select(
        [im["image_overall"].eq(PD), im["image_overall"].isin(NON_PD),
         im["image_overall"].eq(MIXED), im["image_ca"].eq(NED_LABEL)],
        ["Progressing", "Stable/Improving", "Mixed", "NED"], default="Indeterminate")
    return im[["PATIENT_ID", "START_DATE", "LABEL"]]


def build_progression(g: Genie) -> pd.DataFrame:
    im = build_imaging(g)
    label = pd.Series(pd.NA, index=im.index, dtype="object")
    label[im["LABEL"].isin(["Progressing", "Mixed"])] = "Y"
    label[im["LABEL"].eq("Stable/Improving")] = "N"
    out = im.assign(PROGRESSION=label).dropna(subset=["PROGRESSION"])
    return (out[["PATIENT_ID", "START_DATE", "PROGRESSION"]]
            .sort_values(["PATIENT_ID", "START_DATE"], kind="mergesort").reset_index(drop=True))


def imaging_days(g: Genie) -> Dict[str, np.ndarray]:
    im = build_imaging(g)
    return {pid: np.sort(grp["START_DATE"].to_numpy(np.int64)) for pid, grp in im.groupby("PATIENT_ID")}
