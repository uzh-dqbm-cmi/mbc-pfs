from __future__ import annotations

import pandas as pd

from ..utils import filter_and_sort_input
from .base import ModalityFrame


DIAG_COLUMNS = [
    "PATIENT_ID",
    "STAGE_CDM_DERIVED_IV",
    "CLINICAL_GROUP",
    "PATH_GROUP",
    "SUMMARY",
    "HISTOLOGIC",
    "CANCER_SITE_SUBSITE",
]
TOPK_DIAG = 3
SUMMARY_UNKNOWN_CATEGORY = "UNKNOWN"
SUMMARY_CATEGORY_RULES = [
    ("IN_SITU", ("in situ",)),
    ("LOCALIZED", ("localized",)),
    ("DISTANT", ("distant",)),
    ("REGIONAL", ("regional",)),
]


def _limit_top_categories(df: pd.DataFrame, column: str, topk: int) -> None:
    series = df[column].fillna("OTHER")
    top_values = series.value_counts().nlargest(topk).index
    df[column] = series.where(series.isin(top_values), "OTHER")


def _bucket_summary_categories(series: pd.Series) -> pd.Series:
    def categorize(value: object) -> str:
        if pd.isna(value):
            return SUMMARY_UNKNOWN_CATEGORY
        value_str = str(value).strip()
        if not value_str or value_str.upper() == "NA":
            return SUMMARY_UNKNOWN_CATEGORY
        lowered = value_str.lower()
        for category, keywords in SUMMARY_CATEGORY_RULES:
            if any(keyword in lowered for keyword in keywords):
                return category
        return SUMMARY_UNKNOWN_CATEGORY

    return series.map(categorize)


def _collapse_group_substages(series: pd.Series) -> pd.Series:
    def collapse(value: object) -> object:
        if pd.isna(value):
            return value
        value_str = str(value).strip().upper()
        for ch in value_str:
            if ch.isdigit():
                return ch
        return value_str

    return series.map(collapse)


def build_diagnosis_features(ctx) -> ModalityFrame:
    diag_raw = pd.read_csv(ctx.diag_path)
    diag = filter_and_sort_input(diag_raw, ctx.type_specific_patients)
    diag["STAGE_CDM_DERIVED_IV"] = (
        diag["STAGE_CDM_DERIVED"].eq("Stage 4").astype("Int8")
    )
    diag = diag.drop(columns=["STAGE_CDM_DERIVED"])
    diag = diag[DIAG_COLUMNS].copy()
    diag["SUMMARY"] = _bucket_summary_categories(diag["SUMMARY"])
    diag["CLINICAL_GROUP"] = _collapse_group_substages(diag["CLINICAL_GROUP"])
    diag["PATH_GROUP"] = _collapse_group_substages(diag["PATH_GROUP"])

    for column in ["HISTOLOGIC", "CANCER_SITE_SUBSITE"]:
        _limit_top_categories(diag, column, TOPK_DIAG)

    diag_encoded = pd.get_dummies(
        diag,
        columns=[
            "CLINICAL_GROUP",
            "PATH_GROUP",
            "HISTOLOGIC",
            "SUMMARY",
            "CANCER_SITE_SUBSITE",
        ],
        prefix=[
            "CLINICAL_GROUP",
            "PATH_GROUP",
            "HISTOLOGIC",
            "SUMMARY",
            "CANCER_SITE_SUBSITE",
        ],
        drop_first=False,
    )

    or_cols = [
        c
        for c in diag_encoded.columns
        if c.startswith("CANCER_SITE_SUBSITE_") or c.startswith("ICDO_TOPO_")
    ]
    agg_dict = {
        col: ("max" if col in or_cols else "last")
        for col in diag_encoded.columns
        if col != "PATIENT_ID"
    }

    diag_final = (
        diag_encoded.groupby("PATIENT_ID", sort=True, as_index=False)
        .agg(agg_dict)
        .copy()
    )

    return ModalityFrame(name="DIAGNOSIS", frame=diag_final, temporal=False)
