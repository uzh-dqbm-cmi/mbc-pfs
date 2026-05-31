"""Centralised file paths and the MSK feature manifest used as source of truth.
Ensures GENIE columns contain exactly the same features as MSK.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

# --- MSK reference artifacts (trusted) ---
MSK_DESIGN_MATRIX = Path("data/design_matrix.csv")
MSK_FEATURES_DICT = Path("data/features_dict.json")
MSK_DATA_DIR = Path("data/msk_chord_2024")
MSK_TREATMENT_TIMELINE = MSK_DATA_DIR / "data_timeline_treatment.tsv"

# --- raw GENIE source files ---
GENIE_DIR = Path("data/genie")
REGIMEN_FILE = GENIE_DIR / "regimen_cancer_level_dataset.csv"
CANCER_FILE = GENIE_DIR / "cancer_level_dataset_index.csv"
PATIENT_FILE = GENIE_DIR / "patient_level_dataset.csv"
SURVIVAL_FILE = GENIE_DIR / "data_clinical_supp_survival.txt"
IMAGING_FILE = GENIE_DIR / "data_timeline_imaging.txt"
LABTEST_FILE = GENIE_DIR / "data_timeline_labtest.txt"
PATHOLOGY_FILE = GENIE_DIR / "data_timeline_pathology.txt"
PANEL_FILE = GENIE_DIR / "cancer_panel_test_level_dataset.csv"
MUTATIONS_FILE = GENIE_DIR / "data_mutations_extended.txt"
CNA_FILE = GENIE_DIR / "data_CNA.txt"
SV_FILE = GENIE_DIR / "data_sv.txt"

# --- outputs (kept inside data/genie/ so they never collide with the
# distrusted data/genie_design_matrix.csv at the data/ root) ---
PFS_OUT = GENIE_DIR / "genie_pfs.csv"
DESIGN_MATRIX_OUT = GENIE_DIR / "genie_design_matrix.csv"
FEATURES_DICT_OUT = GENIE_DIR / "genie_features_dict.json"

# Identity / label columns carried alongside the feature columns.
KEY_COLS: List[str] = [
    "PATIENT_ID",
    "CA_SEQ",
    "LINE",
    "LINE_START",
    "PFS_TIME_DAYS",
    "PFS_EVENT",
    "LINE_SOURCE",
]


def load_msk_feature_groups() -> Dict[str, List[str]]:
    with open(MSK_FEATURES_DICT, encoding="utf-8") as handle:
        return json.load(handle)


def msk_feature_columns() -> List[str]:
    header = list(pd.read_csv(MSK_DESIGN_MATRIX, nrows=0).columns)
    members = set()
    for cols in load_msk_feature_groups().values():
        members.update(cols)
    ordered = [c for c in header if c in members]
    missing = members.difference(ordered)
    if missing:
        raise ValueError(
            f"features_dict columns absent from design_matrix header: {sorted(missing)}"
        )
    return ordered
