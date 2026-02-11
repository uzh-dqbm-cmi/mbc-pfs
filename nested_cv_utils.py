from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sksurv.util import Surv

from model_setup import EVENT_COL, TIME_COL


@dataclass
class PreparedDesignMatrix:
    cfg: Dict[str, Any]
    seed: int
    design_matrix: pd.DataFrame
    feature_info: Dict[str, Any]
    age_label: Optional[str]


def load_split_indices(split_csv: str | Path) -> Dict[str, Any]:
    df = pd.read_csv(
        split_csv,
        dtype={"split": str, "idx": int, "PATIENT_ID": str, "LINE": int},
    )
    idx_train = df.loc[df["split"] == "train", "idx"].to_numpy(dtype=int)
    idx_val = df.loc[df["split"] == "val", "idx"].to_numpy(dtype=int)
    idx_test = df.loc[df["split"] == "test", "idx"].to_numpy(dtype=int)
    meta = {
        "outer_fold": (
            int(df["outer_fold"].iloc[0])
            if "outer_fold" in df.columns and len(df)
            else None
        ),
        "inner_fold": (
            int(df["inner_fold"].iloc[0])
            if "inner_fold" in df.columns and len(df)
            else None
        ),
    }
    return {
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "meta": meta,
    }


def extract_split_data(
    design_matrix: pd.DataFrame,
    feature_info: Dict[str, Any],
    idxs: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = feature_info["num_cols"] + feature_info["bin_cols"]
    if idxs.size == 0:
        return (
            pd.DataFrame(columns=feature_cols),
            np.asarray([], dtype=float),
            np.asarray([], dtype=bool),
            _empty_surv(),
        )
    X = design_matrix.iloc[idxs][feature_cols].reset_index(drop=True)
    times = design_matrix.iloc[idxs][TIME_COL].to_numpy(dtype=float)
    events = design_matrix.iloc[idxs][EVENT_COL].to_numpy(dtype=bool)
    y = Surv.from_arrays(event=events, time=times)
    return X, times, events, y


def _empty_surv() -> np.ndarray:
    return Surv.from_arrays(
        event=np.asarray([], dtype=bool), time=np.asarray([], dtype=float)
    )
