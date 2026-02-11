from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

TIME_COL = "PFS_TIME_DAYS"
EVENT_COL = "PFS_EVENT"
GROUP_COL = "PATIENT_ID"

PREVALENCE_THRESHOLD = 0.03


def _seed_reproducibly(seed: int) -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="The NumPy global RNG was seeded.*",
    )
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    if hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


@dataclass
class FeatureSelectionResult:
    drop_cols: List[str]
    select_cols: List[str]
    importances: Dict[str, float]
    source: str  # "cache" or "fitted"


@dataclass
class ModelSetupResult:
    cfg: Dict[str, Any]
    seed: int
    features_dict: Dict[str, List[str]]
    design_matrix: pd.DataFrame
    feature_info: Dict[str, Any]
    splits: Dict[str, Any]
    age_label: Optional[str]
    split_log_path: Path


def _sanitize_label(label: Optional[str]) -> str:
    if not label:
        return "full"
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in label)


def _record_or_validate_splits(
    seed: int,
    age_label: Optional[str],
    design_matrix: pd.DataFrame,
    splits: Dict[str, Any],
    data_suffix: str,
) -> Path:
    suffix_fragment = data_suffix or ""
    log_dir = Path(f"splits/{suffix_fragment}" if suffix_fragment else Path("splits"))
    log_dir.mkdir(parents=True, exist_ok=True)
    label = _sanitize_label(age_label)
    log_path = log_dir / f"{label}_seed_{seed}.csv"

    current_log = _build_split_log(design_matrix, splits)

    if log_path.exists():
        previous = pd.read_csv(
            log_path, dtype={"split": str, "idx": int, "PATIENT_ID": str, "LINE": int}
        )
        if not previous.equals(current_log):
            raise ValueError(
                f"Existing split log at {log_path} does not match the current split for seed {seed}."
            )
        else:
            print(f"Split log at {log_path} matches the current split.")
    else:
        print(f"Writing new split log to {log_path}.")
        current_log.to_csv(log_path, index=False)
    return log_path


def _record_or_validate_splits_classification(
    seed: int,
    horizon: int,
    design_matrix: pd.DataFrame,
    splits: Dict[str, Any],
    data_suffix: str,
) -> Path:
    suffix_fragment = data_suffix or ""
    log_dir = Path(f"splits/{suffix_fragment}" if suffix_fragment else Path("splits"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{horizon}_seed_{seed}.csv"

    current_log = _build_split_log(design_matrix, splits)

    if log_path.exists():
        previous = pd.read_csv(
            log_path, dtype={"split": str, "idx": int, "PATIENT_ID": str, "LINE": int}
        )
        if not previous.equals(current_log):
            raise ValueError(
                f"Existing split log at {log_path} does not match the current split for seed {seed}."
            )
        else:
            print(f"Split log at {log_path} matches the current split.")
    else:
        print(f"Writing new split log to {log_path}.")
        current_log.to_csv(log_path, index=False)
    return log_path


def _build_split_log(
    design_matrix: pd.DataFrame,
    splits: Dict[str, Any],
) -> pd.DataFrame:
    """Build a DataFrame logging indices and patient IDs for each split."""
    rows: List[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        idxs = np.asarray(splits[f"idx_{split_name}"], dtype=int)
        patient_ids = design_matrix.iloc[idxs]["PATIENT_ID"].astype(str).to_numpy()
        line = design_matrix.iloc[idxs]["LINE"].astype(int).to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "idx": idxs,
                    "PATIENT_ID": patient_ids,
                    "LINE": line,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
