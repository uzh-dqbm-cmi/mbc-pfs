from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Set

import pandas as pd


@dataclass(frozen=True)
class FeatureMapping:
    """One row of the feature-mapping audit."""

    feature: str  # MSK design-matrix column name
    group: str  # MSK modality (features_dict key)
    genie_source: str  # GENIE file(s) the value comes from
    genie_columns: str  # GENIE column(s) used
    transform: str  # short human description of the derivation
    status: str  # "mapped" | "empty" | "constant"

MARKER_TO_GENIE_TEST: Dict[str, Optional[str]] = {
    "CA15_3": "CA15-3",
    "CEA": None,  # not measured in GENIE BPC BrCa
}
MARKER_CUTOFF = {"CA15_3": 30.0, "CEA": 5.0}

MSK_BODY_PARTS = ["CHEST", "ABDOMEN", "PELVIS", "HEAD", "OTHER"]
SCAN_SITE_TO_BODY_PART: Dict[str, Set[str]] = {
    "BRAIN/HEAD": {"HEAD"},
    "CHEST": {"CHEST"},
    "ABDOMEN": {"ABDOMEN"},
    "PELVIS": {"PELVIS"},
    "SPINE": {"OTHER"},
    "NECK": {"OTHER"},
    "EXTREMITY": {"OTHER"},
    "FULL BODY": set(MSK_BODY_PARTS),
}

MSK_TUMOR_SITES = [
    "ADRENAL GLANDS",
    "BONE",
    "CNS/BRAIN",
    "INTRA-ABDOMINAL",
    "LIVER",
    "LUNG",
    "LYMPH NODES",
    "OTHER",
    "PLEURA",
    "REPRODUCTIVE ORGANS",
]
DIST_METS_TO_TUMOR_SITE: Dict[str, List[str]] = {
    "dist_mets_adrenal": ["ADRENAL GLANDS"],
    "dist_mets_bone": ["BONE"],
    "dist_mets_bone_marrow": ["OTHER"],
    "dist_mets_brain_cns": ["CNS/BRAIN"],
    "dist_mets_abdomen": ["INTRA-ABDOMINAL"],
    "dist_mets_peritoneum_and_malignant_peritoneal_effusion": ["OTHER"],
    "dist_mets_pelvis": ["OTHER"],
    "dist_mets_liver": ["LIVER"],
    "dist_mets_pulmonary": ["LUNG"],
    "dist_mets_thorax": ["OTHER"],
    "dist_mets_pleura_and_malignant_pleural_effusion": ["PLEURA"],
    "dist_mets_pericardial_and_malignant_pericardial_effusion": ["OTHER"],
    "dist_mets_lymph_nodes": ["LYMPH NODES"],
    "dist_mets_head_and_neck": ["OTHER"],
    "dist_mets_breast": ["OTHER"],
    "dist_mets_skin": ["OTHER"],
    "dist_mets_other": ["OTHER"],
}

GENIE_DRUG_TO_AGENT: Dict[str, str] = {
    # --- Hormone therapy ---
    "ANASTROZOLE": "HORMONE_ANASTROZOLE",
    "EXEMESTANE": "HORMONE_EXEMESTANE",
    "FULVESTRANT": "HORMONE_FULVESTRANT",
    "LETROZOLE": "HORMONE_LETROZOLE",
    "TAMOXIFEN": "HORMONE_TAMOXIFEN",
    "LEUPROLIDE": "HORMONE_OTHER",
    "GOSERLIN ACETATE": "HORMONE_OTHER",
    "MEGESTROL ACETATE": "HORMONE_OTHER",
    "TOREMIFENE": "HORMONE_OTHER",
    "BICALUTAMIDE": "HORMONE_OTHER",
    # --- Cytotoxic chemotherapy ---
    "CAPECITABINE": "CHEMO_CAPECITABINE",
    "PACLITAXEL": "CHEMO_PACLITAXEL",
    "GEMCITABINE HCL": "CHEMO_GEMCITABINE",
    "DOXORUBICIN HCL": "CHEMO_DOXORUBICIN",
    "CYCLOPHOSPHAMIDE": "CHEMO_CYCLOPHOSPHAMIDE",
    "CARBOPLATIN": "CHEMO_OTHER",
    "CISPLATIN": "CHEMO_OTHER",
    "OXALIPLATIN": "CHEMO_OTHER",
    "DOCETAXEL": "CHEMO_OTHER",
    "NABPACLITAXEL": "CHEMO_OTHER",
    "PEGYLATED LIPOSOMAL DOXORUBICIN": "CHEMO_OTHER",
    "EPIRUBICIN HCL": "CHEMO_OTHER",
    "ERIBULIN MESYLATE": "CHEMO_OTHER",
    "VINORELBINE TARTRATE": "CHEMO_OTHER",
    "IXABEPILONE": "CHEMO_OTHER",
    "FLUOROURACIL": "CHEMO_OTHER",
    "FLOXURIDINE": "CHEMO_OTHER",
    "METHOTREXATE": "CHEMO_OTHER",
    "PEMETREXED DISODIUM": "CHEMO_OTHER",
    "ETOPOSIDE": "CHEMO_OTHER",
    "IRINOTECAN HCL": "CHEMO_OTHER",
    "IRINOTECAN LIPOSOME": "CHEMO_OTHER",
    "TOPOTECAN HCL": "CHEMO_OTHER",
    "BENDAMUSTINE": "CHEMO_OTHER",
    "DACARBAZINE": "CHEMO_OTHER",
    "TEMOZOLOMIDE": "CHEMO_OTHER",
    "IFOSFAMIDE": "CHEMO_OTHER",
    "LEUCOVORIN": "CHEMO_OTHER",
    "LURBINECTEDIN": "CHEMO_OTHER",
    # --- Biologics / antibodies / ADCs ---
    "PERTUZUMAB": "BIOLOGIC_PERTUZUMAB",
    "TRASTUZUMAB": "BIOLOGIC_TRASTUZUMAB",
    "TRASTUZUMAB EMTANSINE": "BIOLOGIC_ADO-TRASTUZUMAB EMTANSINE",
    "TRASTUZUMAB DERUXTECAN": "BIOLOGIC_FAM-TRASTUZUMAB DERUXTECAN",
    "BEVACIZUMAB": "BIOLOGIC_OTHER",
    "RAMUCIRUMAB": "BIOLOGIC_OTHER",
    "CETUXIMAB": "BIOLOGIC_OTHER",
    "RITUXIMAB": "BIOLOGIC_OTHER",
    "SACITUZUMAB GOVITECAN": "BIOLOGIC_OTHER",
    # --- Targeted small molecules ---
    "ABEMACICLIB": "TARGETED_ABEMACICLIB",
    "PALBOCICLIB": "TARGETED_PALBOCICLIB",
    "EVEROLIMUS": "TARGETED_EVEROLIMUS",
    "ALPELISIB": "TARGETED_ALPELISIB",
    "RIBOCICLIB": "TARGETED_OTHER",
    "LAPATINIB DITOSYLATE": "TARGETED_OTHER",
    "NERATINIB": "TARGETED_OTHER",
    "TUCATINIB": "TARGETED_OTHER",
    "ERLOTINIB HCL": "TARGETED_OTHER",
    "DASATINIB": "TARGETED_OTHER",
    "SORAFENIB TOSYLATE": "TARGETED_OTHER",
    "PAZOPANIB HCL": "TARGETED_OTHER",
    "PONATINIB HCL": "TARGETED_OTHER",
    "LENVATINIB MESYLATE": "TARGETED_OTHER",
    "TRAMETINIB": "TARGETED_OTHER",
    "LAROTRECTINIB": "TARGETED_OTHER",
    "OLAPARIB": "TARGETED_OTHER",
    "TALAZOPARIB": "TARGETED_OTHER",
    # --- Immunotherapy ---
    "PEMBROLIZUMAB": "IMMUNO_PEMBROLIZUMAB",
    "ATEZOLIZUMAB": "IMMUNO_ATEZOLIZUMAB",
    "IPILIMUMAB": "IMMUNO_IPILIMUMAB",
    "NIVOLUMAB": "IMMUNO_NIVOLUMAB",
    # --- Investigational (MSK records these as the OTHER_INVESTIGATIVE agent) ---
    "INVESTIGATIONAL DRUG": "OTHER_INVESTIGATIVE",
}


def line_agent_suffixes(drugs: object) -> Optional[Set[str]]:
    suffixes: Set[str] = set()
    for token in str(drugs).split(","):
        token = token.strip().upper()
        if not token:
            continue
        mapped = GENIE_DRUG_TO_AGENT.get(token)
        if mapped is None:
            return None
        suffixes.add(mapped)
    return suffixes if suffixes else None

def normalize_drug(name: object) -> str:
    return str(name).strip().upper()


AGENT_SALT_WORDS: Set[str] = {
    "HCL", "HYDROCHLORIDE", "DIHYDROCHLORIDE", "MESYLATE", "TARTRATE", "ACETATE",
    "DITOSYLATE", "TOSYLATE", "DISODIUM", "SODIUM", "POTASSIUM", "SULFATE",
    "SULPHATE", "CITRATE", "MALEATE", "DIMALEATE", "SUCCINATE", "PHOSPHATE",
    "CALCIUM", "LACTATE", "FUMARATE", "BROMIDE",
}
AGENT_SYNONYMS: Dict[str, str] = {
    "INVESTIGATIONAL DRUG": "INVESTIGATIVE",
    "NABPACLITAXEL": "PACLITAXEL PROTEIN-BOUND",
    "NAB-PACLITAXEL": "PACLITAXEL PROTEIN-BOUND",
    "GOSERLIN": "GOSERELIN",
    "TRASTUZUMAB DERUXTECAN": "FAM-TRASTUZUMAB DERUXTECAN",
    "TRASTUZUMAB EMTANSINE": "ADO-TRASTUZUMAB EMTANSINE",
    "PEGYLATED LIPOSOMAL DOXORUBICIN": "DOXORUBICIN LIPOSOMAL",
    "IRINOTECAN LIPOSOME": "IRINOTECAN LIPOSOMAL",
}
AGENT_COMBO_SPLITS: Dict[str, List[str]] = {
    "LETROZOLE-RIBOCICLIB": ["LETROZOLE", "RIBOCICLIB"],
}


def canonical_agent(name: object) -> str:
    """Normalise an MSK ``AGENT`` or GENIE drug name to one shared token."""
    s = re.sub(r"\s+", " ", str(name).strip().upper())
    if s in AGENT_SYNONYMS:
        return AGENT_SYNONYMS[s]
    toks = s.split(" ")
    while len(toks) > 1 and toks[-1] in AGENT_SALT_WORDS:
        toks.pop()
    s = " ".join(toks)
    return AGENT_SYNONYMS.get(s, s)


def canonical_regimen(tokens: Iterable[object]) -> FrozenSet[str]:
    """Canonical frozenset of agents for a regimen, expanding combo products."""
    agents: Set[str] = set()
    for tok in tokens:
        a = canonical_agent(tok)
        if not a or a == "NAN":
            continue
        if a in AGENT_COMBO_SPLITS:
            agents.update(AGENT_COMBO_SPLITS[a])
        else:
            agents.add(a)
    return frozenset(agents)


def is_positive(value: object) -> bool:
    """True for GENIE biomarker results that indicate positivity/elevation."""
    if pd.isna(value):
        return False
    return "POSITIVE" in str(value).strip().upper()


def cancer_status_bucket(value: object) -> str:
    """Map a GENIE imaging CANCER_STATUS string to Y / N / INDET."""
    text = "" if pd.isna(value) else str(value).strip().upper()
    if text.startswith("YES"):
        return "Y"
    if text.startswith("NO") or "DOES NOT MENTION" in text:
        return "N"
    return "INDET"


def scan_sites_to_parts(value: object) -> Set[str]:
    """Translate a comma-joined SCAN_SITES string to MSK body parts."""
    parts: Set[str] = set()
    if pd.isna(value):
        return parts
    for token in str(value).split(","):
        token = token.strip().upper()
        if token in SCAN_SITE_TO_BODY_PART:
            parts |= SCAN_SITE_TO_BODY_PART[token]
    return parts


def collapse_stage_digit(code: object) -> Optional[str]:
    """First numeric digit of a NAACCR group-stage code (e.g. '2A' -> '2')."""
    if pd.isna(code):
        return None
    for ch in str(code).strip().upper():
        if ch.isdigit():
            return ch
    return None


def seer_summary_category(code: object) -> str:
    if pd.isna(code):
        return "UNKNOWN"
    try:
        digit = int(float(code))
    except (TypeError, ValueError):
        return "UNKNOWN"
    if digit == 0:
        return "IN_SITU"
    if digit == 1:
        return "LOCALIZED"
    if digit in (2, 3, 4, 5):
        return "REGIONAL"
    if digit == 7:
        return "DISTANT"
    return "UNKNOWN"


def histologic_category(ca_histology: object, ca_hist_brca: object) -> str:
    text = " ".join(
        str(v).strip().upper()
        for v in (ca_histology, ca_hist_brca)
        if not pd.isna(v)
    )
    has_duct = "DUCT" in text or "NO SPECIAL TYPE" in text
    has_lobular = "LOBULAR" in text
    if has_duct and has_lobular:
        return "HISTOLOGIC_LOBULAR CA + IFDC/DCIS"
    if has_lobular:
        return "HISTOLOGIC_LOBULAR CARCINOMA, NOS"
    if has_duct:
        return "HISTOLOGIC_INFILTRATING DUCT CARCINOMA"
    return "HISTOLOGIC_OTHER"


def site_subsite_category(ca_d_site: object) -> str:
    text = "" if pd.isna(ca_d_site) else str(ca_d_site).strip().upper()
    if text == "C50.9":
        return "CANCER_SITE_SUBSITE_NOS"
    if text == "C50.8":
        return "CANCER_SITE_SUBSITE_OVERLAPPING LESION OF BREAST"
    if text == "C50.4":
        return "CANCER_SITE_SUBSITE_UOQ"
    return "CANCER_SITE_SUBSITE_OTHER"
