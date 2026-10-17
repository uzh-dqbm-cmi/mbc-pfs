from __future__ import annotations

import pandas as pd

from ..utils import filter_and_sort_input
from .base import ModalityFrame

CLINICAL_FILE = "data_clinical_patient.tsv"
MSK_BRCA_COHORT_FILE = "brca_dx_1st_seq_OS.csv"


def build_clinical_features(ctx) -> ModalityFrame:
    raw = pd.read_csv(ctx.data_dir / CLINICAL_FILE, sep="\t", dtype=str, skiprows=4)
    msk_brca = pd.read_csv(ctx.data_dir / MSK_BRCA_COHORT_FILE, dtype=str)
    clin = filter_and_sort_input(raw, ctx.type_specific_patients)
    msk_brca_clin = filter_and_sort_input(msk_brca, ctx.type_specific_patients)

    # check if the two DFs have the same patients in the same order, and if not, raise an error
    if not clin["PATIENT_ID"].equals(msk_brca_clin["PATIENT_ID"]):
        raise ValueError(
            "Patient IDs in clinical and MSK-BRCA data do not match or are not in the same order"
        )

    # Use AGE from https://github.com/clinical-data-mining/msk-chord-figures-public/blob/main/data/brca_dx_1st_seq_OS.csv, where the original MSK-CHORD paper performs survival analysis using the age at time 0 (time of sequencing)
    clin["AGE"] = msk_brca_clin["AGE"].astype("Float32")

    clin["GENDER_IS_FEMALE"] = (clin["GENDER"] == "FEMALE").astype("Int8")
    clin["HR"] = clin["HR"].str.upper().eq("YES").astype("Int8")
    clin["HER2"] = clin["HER2"].str.upper().eq("YES").astype("Int8")

    cols = ["PATIENT_ID", "GENDER_IS_FEMALE", "AGE", "HR", "HER2"]
    output = clin[cols].copy()
    return ModalityFrame(name="CLINICAL", frame=output, temporal=False)
