from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from config import CANCER_TYPE, LINE_START_WINDOW_DAYS, MSK_DATA_DIR
from data_paths import data_dir


@dataclass
class DesignMatrixContext:
    data_dir: Path
    outcomes_path: Path
    diag_path: Path
    type_specific_patients: Set[str]
    patient_lot_info: pd.DataFrame
    window_days: int = LINE_START_WINDOW_DAYS
    top_agents: Optional[Dict[str, Set[str]]] = None
    diagnosis_categories: Optional[Dict[str, pd.Index]] = None


def create_context(window_days: int = LINE_START_WINDOW_DAYS) -> DesignMatrixContext:
    out_dir = data_dir(window_days)
    outcomes_path = out_dir / f"{CANCER_TYPE}_pfs.csv"
    outcomes_raw = pd.read_csv(outcomes_path)
    patient_lot_info = outcomes_raw[
        ["PATIENT_ID", "LINE", "LINE_START", "EVENT_DAY"]
    ].copy()
    return DesignMatrixContext(
        data_dir=MSK_DATA_DIR,
        outcomes_path=outcomes_path,
        diag_path=out_dir / "diagnosis_parsed.csv",
        type_specific_patients=set(outcomes_raw["PATIENT_ID"]),
        patient_lot_info=patient_lot_info,
        window_days=window_days,
    )
