import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

RESULTS_PATH = Path("results")
FIGURES_PATH = Path("manuscript/figures")
PATIENCE = 50

DEVICE = "cpu"
if torch is not None:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
print(f"Using device: {DEVICE}", file=sys.stderr)

MODALITY_GROUPS: Dict[str, List[str]] = {
    "Treatment": ["TREATMENT", "LOCAL_TREATMENT"],
    "Genomics": [
        "GENOMICS",
        "PDL1",
        "MMR",
    ],
    "Clinicopathologic": ["DIAGNOSIS", "CLINICAL", "ECOG"],
    "Tumor Markers": ["TUMOR_MARKERS"],
    "Radiology": ["TUMOR_SITES", "CANCER_PRESENCE"],
}

DESIGN_MATRIX_PATH = Path("data") / "design_matrix.csv"
FEATURES_DICT_PATH = Path("data") / "features_dict.json"
MSK_DATA_DIR = Path("data/msk_chord_2024")
CANCER_TYPE = "BREAST"

# additionally, read features_dict.json to group survival-pattern features
# this can only be used after features_dict.json is generated
features_dict = (
    json.load(open(FEATURES_DICT_PATH)) if FEATURES_DICT_PATH.exists() else {}
)
surv_pattern_suffixes = (
    "_MISSING",
    "_DAY",
    "_IMAGED_Y_STATUS",
    "_IMAGED_N_STATUS",
    "_IMAGED_INDET_STATUS",
)
surv_patterns: list[str] = []
for cols in features_dict.values():
    for name in cols:
        if name.endswith(surv_pattern_suffixes):
            surv_patterns.append(name)
surv_patterns = list(dict.fromkeys(surv_patterns))

MODALITY_GROUPS["Surveillance Patterns"] = surv_patterns

RANDOM_STATE = 42
ADMIN_CENSOR_DAYS = 365 * 2
LINE_START_WINDOW_DAYS = 28
DAY_SPLIT = 365
MONTH_DAYS = 30.437
CALIBRATION_HORIZONS = (180, 365, 730)
MIN_PFS_DAYS = 28
EVAL_HORIZONS = list(range(90, 2 * 365 + 1, 30))
EVAL_HORIZONS += [365, 730]


# Explainability / evaluation constants
SHAP_GBSA_BACKGROUND = 200

# Aggregation windows (for building the design matrix)
# Tumor markers (numeric):
MARKER_SHORT_WINDOW_DAYS = 60
MARKER_LONG_WINDOW_DAYS = 180
MARKER_TAU_SHORT = 30.0
MARKER_TAU_LONG = 90.0

# Radiology report window:
RADIOLOGY_REPORT_WINDOW_DAYS = 90

# ECOG parameters
ECOG_LOOKBACK_DAYS = 90
ECOG_EVER_GE2_WINDOW = 180

# Constants for column names
PATIENT_ID_COL = "PATIENT_ID"
LINE_COL = "LINE"
START_DATE_COL = "START_DATE"

EVENT_COL = "PFS_EVENT"
TIME_COL = "PFS_TIME_DAYS"
IGNORE_COLS = [
    "PATIENT_ID",
    "PFS_TIME_DAYS",
    "PFS_EVENT",
    "LINE",
    "LINE_START",
    "LINE_SOURCE",
    "NO_RADIOLOGY_REPORT_WITHIN",
    "PRIOR_TO_FIRST_RADIOLOGY",
    "ORIGINAL_PFS_TIME_DAYS",
    "ORIGINAL_PFS_EVENT",
    "QC_HARD_FAIL",
    "QC_REASONS",
]
IGNORE_PREFIXES = [
    "RECEIVED_",
]

C_INDEX = "C"
AUC = "mean_auc"
IBS = "IBS"

MUTATION_GENES = [
    "AKT1",
    "ARID1A",
    "ATM",
    "BRCA1",
    "BRCA2",
    "CBFB",
    "CDH1",
    "CDKN1B",
    "ESR1",
    "FOXA1",
    "GATA3",
    "KMT2C",
    "KMT2D",
    "MAP2K4",
    "MAP3K1",
    "NCOR1",
    "NF1",
    "PIK3CA",
    "PIK3R1",
    "PTEN",
    "RB1",
    "RUNX1",
    "SF3B1",
    "TBX3",
    "TP53",
]

AMPLIFICATION_GENES = [
    "CCND1",
    "CCNE1",
    "EGFR",
    "ERBB2",
    "ERLIN2",
    "FGFR1",
    "FGFR2",
    "FOXA1",
    "GRB7",
    "IGF1R",
    "MCL1",
    "MYC",
    "PAK1",
    "PIK3CA",
    "RPS6KB1",
    "ZNF217",
    "ZNF703",
]

DELETION_GENES = [
    "ATM",
    "BRCA1",
    "BRCA2",
    "CDKN2A",
    "CDKN2B",
    "MAP2K4",
    "PTEN",
    "RB1",
]
