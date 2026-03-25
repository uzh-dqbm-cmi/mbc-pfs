from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .modalities.base import ModalityFrame


def _apply_genomics_timing_rules(
    design_matrix: pd.DataFrame, features_dict: Dict[str, List[str]]
) -> pd.DataFrame:
    if "SEQUENCING_DATE" not in design_matrix.columns:
        return design_matrix

    genomics_feature_cols = [
        col
        for col in design_matrix.columns
        if col.startswith("GENOMICS_") and col != "GENOMICS_MISSING"
    ]
    design_matrix["LINE_START"] = pd.to_numeric(
        design_matrix["LINE_START"], errors="raise"
    ).astype(int)
    design_matrix["SEQUENCING_DATE"] = pd.to_numeric(
        design_matrix["SEQUENCING_DATE"], errors="coerce"
    ).astype("Int64")

    invalid_genomics_mask = design_matrix["SEQUENCING_DATE"].isna() | (
        design_matrix["SEQUENCING_DATE"] > design_matrix["LINE_START"]
    )
    design_matrix.loc[invalid_genomics_mask, genomics_feature_cols] = 0
    design_matrix.loc[invalid_genomics_mask, "GENOMICS_MISSING"] = 1

    design_matrix = design_matrix.drop(columns=["SEQUENCING_DATE"])
    if "GENOMICS" in features_dict:
        features_dict["GENOMICS"] = [
            col for col in features_dict["GENOMICS"] if col != "SEQUENCING_DATE"
        ]
    return design_matrix


def assemble_design_matrix(
    ctx, modalities: List[ModalityFrame]
) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    features_dict: Dict[str, List[str]] = {}

    temporal_frames = [m for m in modalities if m.temporal]
    static_frames = [m for m in modalities if not m.temporal]
    # set all column names to uppercase
    for frame in temporal_frames + static_frames:
        frame.frame.columns = frame.frame.columns.str.upper()

    design_matrix = ctx.patient_lot_info[["PATIENT_ID", "LINE", "LINE_START"]].copy()
    design_matrix = design_matrix.set_index(["PATIENT_ID", "LINE"])

    for frame in temporal_frames:
        frame.frame.set_index(["PATIENT_ID", "LINE"], inplace=True)
        design_matrix = design_matrix.join(
            frame.frame,
            how="inner",
            validate="one_to_one",
        )
        # get names of the feature columns
        features_dict[frame.name] = frame.feature_columns().tolist()

    design_matrix = design_matrix.reset_index()

    for frame in static_frames:
        # print out the shape of each frame and the name of it
        print(f"Merging static modality: {frame.name} with shape {frame.frame.shape}")
        design_matrix = design_matrix.merge(
            frame.frame, on="PATIENT_ID", how="inner", validate="many_to_one"
        )
        features_dict[frame.name] = frame.feature_columns().tolist()

    design_matrix = _apply_genomics_timing_rules(design_matrix, features_dict)

    design_matrix = design_matrix.sort_values(["PATIENT_ID", "LINE"]).reset_index(
        drop=True
    )

    outcomes = pd.read_csv(ctx.outcomes_path)
    outcomes = outcomes.sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)
    design_matrix = design_matrix.merge(
        outcomes[outcomes["PATIENT_ID"].isin(ctx.type_specific_patients)].drop(
            columns=["LINE_START"], errors="ignore"
        ),
        on=["PATIENT_ID", "LINE"],
        how="inner",
        validate="one_to_one",
    )
    design_matrix.drop(columns=["EVENT_DAY"], inplace=True)

    return design_matrix, features_dict
