from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .modalities.base import ModalityFrame


def assemble_design_matrix(
    ctx, modalities: List[ModalityFrame]
) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    features_dict: Dict[str, List[str]] = {}

    temporal_frames = [m for m in modalities if m.temporal]
    static_frames = [m for m in modalities if not m.temporal]
    # set all column names to uppercase
    for frame in temporal_frames + static_frames:
        frame.frame.columns = frame.frame.columns.str.upper()

    design_matrix = ctx.patient_lot_info[["PATIENT_ID", "LINE"]].copy()
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

    design_matrix = design_matrix.sort_values(["PATIENT_ID", "LINE"]).reset_index(
        drop=True
    )

    outcomes = pd.read_csv(ctx.outcomes_path)
    outcomes = outcomes.sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)
    design_matrix = design_matrix.merge(
        outcomes[outcomes["PATIENT_ID"].isin(ctx.type_specific_patients)],
        on=["PATIENT_ID", "LINE"],
        how="inner",
        validate="one_to_one",
    )
    design_matrix.drop(columns=["EVENT_DAY"], inplace=True)

    return design_matrix, features_dict
