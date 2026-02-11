from __future__ import annotations

import pandas as pd

from ..utils import filter_and_sort_input
from .base import ModalityFrame

CLINICAL_FILE = "data_clinical_patient.tsv"


def build_clinical_features(ctx) -> ModalityFrame:
    raw = pd.read_csv(ctx.data_dir / CLINICAL_FILE, sep="\t", dtype=str, skiprows=4)
    clin = filter_and_sort_input(raw, ctx.type_specific_patients)

    rename = {"CURRENT_AGE_DEID": "AGE"}
    clin = clin.rename(columns=rename)
    clin["GENDER_IS_FEMALE"] = (clin["GENDER"] == "FEMALE").astype("Int8")
    clin["HR"] = clin["HR"].str.upper().eq("YES").astype("Int8")
    clin["HER2"] = clin["HER2"].str.upper().eq("YES").astype("Int8")
    clin["AGE"] = pd.to_numeric(clin["AGE"]).astype("Float32")

    cols = ["PATIENT_ID", "GENDER_IS_FEMALE", "AGE", "HR", "HER2"]
    output = clin[cols].copy()
    return ModalityFrame(name="CLINICAL", frame=output, temporal=False)
