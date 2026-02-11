from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ModalityFrame:
    name: str
    frame: pd.DataFrame
    temporal: bool
    description: Optional[str] = None

    def feature_columns(self) -> pd.Index:
        cols = list(self.frame.columns)
        drop = {"PATIENT_ID"}
        if self.temporal:
            drop.add("LINE")
            drop.add("LINE_START")
        return pd.Index([c for c in cols if c not in drop])
