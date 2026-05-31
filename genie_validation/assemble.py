from __future__ import annotations

import json
from typing import List, Tuple

import pandas as pd

from config import EVENT_COL, TIME_COL
from . import features, labels, mapping, paths


def _load_context(label_df: pd.DataFrame) -> features.FeatureContext:
    cna = pd.read_csv(paths.CNA_FILE, sep="\t", low_memory=False, index_col=0)
    cna = cna[~cna.index.duplicated(keep="first")]
    return features.FeatureContext(
        labels=label_df,
        cancer=pd.read_csv(paths.CANCER_FILE, low_memory=False),
        patient=pd.read_csv(paths.PATIENT_FILE, low_memory=False),
        imaging=pd.read_csv(paths.IMAGING_FILE, sep="\t", low_memory=False),
        labtest=pd.read_csv(paths.LABTEST_FILE, sep="\t", low_memory=False),
        pathology=pd.read_csv(paths.PATHOLOGY_FILE, sep="\t", low_memory=False),
        panel=pd.read_csv(paths.PANEL_FILE, low_memory=False),
        mutations=pd.read_csv(
            paths.MUTATIONS_FILE, sep="\t", low_memory=False,
            usecols=["Hugo_Symbol", "Tumor_Sample_Barcode"],
        ),
        cna=cna,
        groups=paths.load_msk_feature_groups(),
    )


def build_design_matrix(
    label_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[mapping.FeatureMapping]]:
    ctx = _load_context(label_df)
    feature_cols = paths.msk_feature_columns()

    matrix = label_df[["PATIENT_ID", "LINE"]].copy()
    all_mappings: List[mapping.FeatureMapping] = []
    for builder in features.BUILDERS:
        frame, mappings = builder(ctx)
        if frame[["PATIENT_ID", "LINE"]].duplicated().any():
            raise ValueError(f"{builder.__name__} produced duplicate (PATIENT_ID, LINE)")
        matrix = matrix.merge(frame, on=["PATIENT_ID", "LINE"], how="left", validate="one_to_one")
        all_mappings.extend(mappings)

    produced = set(matrix.columns) - {"PATIENT_ID", "LINE"}
    expected = set(feature_cols)
    if produced != expected:
        raise ValueError(
            "Feature column mismatch vs MSK.\n"
            f"  missing: {sorted(expected - produced)}\n"
            f"  extra:   {sorted(produced - expected)}"
        )
    audited = {m.feature for m in all_mappings}
    if audited != expected:
        raise ValueError(
            "Audit/feature mismatch.\n"
            f"  unaudited: {sorted(expected - audited)}\n"
            f"  spurious:  {sorted(audited - expected)}"
        )

    keys = label_df[["PATIENT_ID", "CA_SEQ", "LINE", "LINE_START", TIME_COL, EVENT_COL]].copy()
    design = keys.merge(matrix, on=["PATIENT_ID", "LINE"], how="left", validate="one_to_one")
    design = design[
        ["PATIENT_ID", "CA_SEQ", "LINE", "LINE_START", TIME_COL, EVENT_COL, *feature_cols]
    ]
    design["LINE_SOURCE"] = labels.LINE_SOURCE

    if design[["PATIENT_ID", "LINE"]].duplicated().any():
        raise ValueError("Duplicate (PATIENT_ID, LINE) in final design matrix")
    return design, all_mappings


def build_and_write() -> pd.DataFrame:
    paths.GENIE_DIR.mkdir(parents=True, exist_ok=True)
    label_df = labels.write_labels()
    design, all_mappings = build_design_matrix(label_df)

    design.to_csv(paths.DESIGN_MATRIX_OUT, index=False)
    with open(paths.FEATURES_DICT_OUT, "w", encoding="utf-8") as handle:
        json.dump(paths.load_msk_feature_groups(), handle, indent=2)

    print(f"  wrote {paths.DESIGN_MATRIX_OUT} {design.shape}")
    print(f"  patients={design['PATIENT_ID'].nunique()} lines={len(design)}")
    print(f"  event distribution:\n{design[EVENT_COL].value_counts().to_string()}")
    return design
