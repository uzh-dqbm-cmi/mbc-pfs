"""Shared feature grouping utilities (extracted from manuscript.py)."""

from __future__ import annotations


def feature_module(name: str) -> str | None:
    """
    Map a raw feature name to a clinical module.
    Return None to drop the feature from grouped SHAP.
    """

    if name.startswith("RECEIVED_"):
        return None

    # --- ECOG ---------------------------------------------------------
    if name in ["ECOG_LAST_OBS", "EVER_GE2_180D"]:
        return "ECOG status"
    if name in ["ECOG_LAST_OBS_DAY", "ECOG_MISSING"]:
        return "ECOG monitoring"

    # --- Tumor markers: CA15-3 ---------------------------------------
    if name.startswith("CA15_3_"):
        burden_keys = [
            "LAST_OBS_LOG",
            "SHORT_MAX",
            "SHORT_MIN",
            "SHORT_RW_MEAN",
            "LONG_MAX",
            "LONG_MIN",
            "LONG_RW_MEAN",
            "SHORT_SLOPE",
            "LONG_SLOPE",
            "SHORT_DELTA_LAST",
            "LONG_DELTA_LAST",
            "SHORT_OVER_LONG",
            "SHORT_MINUS_LONG",
            "ABOVE_CUTOFF",
            "RISE_GT20PCT",
        ]
        if any(k in name for k in burden_keys):
            return "CA15-3 kinetics"
        return "CA15-3 monitoring"

    # --- Tumor markers: CEA ------------------------------------------
    if name.startswith("CEA_"):
        burden_keys = [
            "LAST_OBS_LOG",
            "SHORT_MAX",
            "SHORT_MIN",
            "SHORT_RW_MEAN",
            "LONG_MAX",
            "LONG_MIN",
            "LONG_RW_MEAN",
            "SHORT_SLOPE",
            "LONG_SLOPE",
            "SHORT_DELTA_LAST",
            "LONG_DELTA_LAST",
            "SHORT_OVER_LONG",
            "SHORT_MINUS_LONG",
            "ABOVE_CUTOFF",
            "RISE_GT20PCT",
        ]
        if any(k in name for k in burden_keys):
            return "CEA kinetics"
        return "CEA monitoring"

    # --- Metastatic sites (tumor_site_*) -----------------------------
    if name.startswith("TUMOR_SITE_"):
        if name.endswith("_EVER"):
            return "Metastases burden: historic"
        if name.endswith("_IN_WINDOW"):
            return "Metastases burden: recent"
        return "Metastases burden"

    # --- Imaging (CANCER_*) ------------------------------------------
    if name.startswith("CANCER_"):
        if (
            "IMAGED_Y_COUNT" in name
            or "IMAGED_INDET_COUNT" in name
            or name.endswith("_EVER")
        ):
            return "Imaging: positive/indeterminate counts"
        if "IMAGED_N_COUNT" in name or name.endswith("_NOT_IMAGED"):
            return "Imaging: negative/none counts"

    # --- Prior treatment history -------------------------------------
    if name == "NUM_PRIOR_LINES":
        return "Prior treatment intensity"
    if name.startswith("HISTORY_") and name.endswith("_365D_COUNT"):
        return "Prior treatment intensity"
    if name.startswith("HISTORY_") and name.endswith("_LAST_OBS_DAY"):
        return "Time since last treatment"

    # --- Current line regimen (planned / within-line) ----------------
    if name.startswith("PLANNED_TREATMENT_") or name.startswith("PLANNED_AGENT_"):
        return "Planned treatment"

    # --- Genomics -------------------------------
    if name.startswith("GENOMICS_"):
        return "Genomic alterations"

    if name == "AGE" or name == "HR" or name == "HER2":
        return name

    if name.startswith("PDL1_") or name.startswith("MMR_"):
        return "Immunotherapy biomarkers"

    return "Diagnosis"


__all__ = ["feature_module"]
