from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sksurv.util import Surv

from config import ADMIN_CENSOR_DAYS, EVENT_COL, IGNORE_COLS, IGNORE_PREFIXES, TIME_COL


class NanRobustScaler(BaseEstimator, TransformerMixin):
    """Robustly scale features while leaving NaNs in place."""

    def __init__(self, with_centering: bool = True, with_scaling: bool = True) -> None:
        self.with_centering = with_centering
        self.with_scaling = with_scaling

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        n_features = X_arr.shape[1]
        if self.with_centering:
            center = np.nanmedian(X_arr, axis=0)
            center = np.where(np.isfinite(center), center, 0.0)
        else:
            center = np.zeros(n_features, dtype=float)

        if self.with_scaling:
            q75 = np.nanpercentile(X_arr, 75, axis=0)
            q25 = np.nanpercentile(X_arr, 25, axis=0)
            scale = q75 - q25
            scale = np.where(np.isfinite(scale) & (scale != 0), scale, 1.0)
        else:
            scale = np.ones(n_features, dtype=float)

        self.center_ = center
        self.scale_ = scale
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.with_centering:
            X_arr = X_arr - self.center_
        if self.with_scaling:
            X_arr = X_arr / self.scale_
        return X_arr


def preprocess_df(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    design_matrix: pd.DataFrame,
    ignore_prefix: str | None = None,
) -> dict:
    """Basic preprocessing steps on raw design matrix DataFrame."""

    train_df = design_matrix.iloc[train_idx]
    val_df = design_matrix.iloc[val_idx]
    test_df = design_matrix.iloc[test_idx]
    X_train_df, time_train_series, event_train_series, numeric_cols, binary_cols = (
        typecast(train_df, ignore_prefix=ignore_prefix)
    )
    X_val_df, time_val_series, event_val_series, _, _ = typecast(
        val_df, ignore_prefix=ignore_prefix
    )
    X_test_df, time_test_series, event_test_series, _, _ = typecast(
        test_df, ignore_prefix=ignore_prefix
    )

    # Standardize numeric features for consistency across model families.
    num_pipe = Pipeline(
        [("scale", NanRobustScaler(with_centering=True, with_scaling=True))]
    )
    num_transformer = num_pipe
    preproc = ColumnTransformer(
        [
            ("num", num_transformer, numeric_cols),
            ("bin", "passthrough", binary_cols),
        ],
        remainder="drop",
    )
    X_train_np = preproc.fit_transform(X_train_df)
    X_val_np = preproc.transform(X_val_df)
    X_test_np = preproc.transform(X_test_df)
    num_feature_count = len(numeric_cols)
    if num_feature_count:
        num_slice = slice(0, num_feature_count)
        X_train_np[:, num_slice] = np.nan_to_num(X_train_np[:, num_slice], nan=0.0)
        X_val_np[:, num_slice] = np.nan_to_num(X_val_np[:, num_slice], nan=0.0)
        X_test_np[:, num_slice] = np.nan_to_num(X_test_np[:, num_slice], nan=0.0)

    time_train_arr = time_train_series.to_numpy(dtype=np.float32)
    event_train_arr = event_train_series.to_numpy(dtype=np.int32)
    time_val_arr = time_val_series.to_numpy(dtype=np.float32)
    event_val_arr = event_val_series.to_numpy(dtype=np.int32)
    time_test_arr = time_test_series.to_numpy(dtype=np.float32)
    event_test_arr = event_test_series.to_numpy(dtype=np.int32)
    y_train_sksurv = Surv.from_arrays(
        event=event_train_arr.astype(bool), time=time_train_arr
    )
    y_val_sksurv = Surv.from_arrays(event=event_val_arr.astype(bool), time=time_val_arr)
    y_test_sksurv = Surv.from_arrays(
        event=event_test_arr.astype(bool), time=time_test_arr
    )

    return {
        "idx_train": np.asarray(train_idx, dtype=int),
        "idx_val": np.asarray(val_idx, dtype=int),
        "idx_test": np.asarray(test_idx, dtype=int),
        "preprocessor": preproc,
        "X_train_np": X_train_np,
        "time_train_arr": time_train_arr,
        "event_train_arr": event_train_arr,
        "X_val_np": X_val_np,
        "time_val_arr": time_val_arr,
        "event_val_arr": event_val_arr,
        "X_test_np": X_test_np,
        "time_test_arr": time_test_arr,
        "event_test_arr": event_test_arr,
        "y_train_sksurv": y_train_sksurv,
        "y_val_sksurv": y_val_sksurv,
        "y_test_sksurv": y_test_sksurv,
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
        "feature_cols": numeric_cols + binary_cols,
    }


def _binary_label(series: pd.Series, name: str) -> str:
    """Ensure a binary column is consistent within a group and return its label."""

    values = pd.to_numeric(series, errors="raise").astype(int)
    uniques = values.unique()
    if len(uniques) != 1:
        raise ValueError(f"{name} not consistent within group: {uniques}")
    val = int(uniques[0])
    if val not in (0, 1):
        raise ValueError(f"{name} must be 0/1, got {val}")
    return str(val)


def _admin_censor_flag(grp: pd.DataFrame, event_col: str, time_col: str) -> str:
    """Flag if any line for the patient is administratively censored at the configured horizon."""

    events = pd.to_numeric(grp[event_col], errors="raise").astype(int)
    times = pd.to_numeric(grp[time_col], errors="raise").astype(float)
    admin_censored = bool(((events == 0) & (times >= ADMIN_CENSOR_DAYS)).any())
    return f"AC{int(admin_censored)}"


def _hr_her2_admin_strata(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    event_col: str,
    hr_col: str = "HR",
    her2_col: str = "HER2",
) -> np.ndarray:
    """
    Derive a single stratification label per patient combining HR, HER2, and whether any LoT was administratively censored.
    Raises if any group has mixed values for HR/HER2.
    """

    missing = [
        col
        for col in (time_col, event_col, hr_col, her2_col, group_col)
        if col not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns for stratification: {missing}")

    group_labels: Dict[str, str] = {}
    for pid, grp in df.groupby(group_col, sort=False):
        admin_flag = _admin_censor_flag(grp, event_col=event_col, time_col=time_col)
        hr_label = _binary_label(grp[hr_col], name=hr_col)
        her2_label = _binary_label(grp[her2_col], name=her2_col)
        group_labels[str(pid)] = f"{admin_flag}_HR{hr_label}_HER2{her2_label}"

    return df[group_col].astype(str).map(group_labels).to_numpy()


# ----------------- 2) Typecasting -----------------


def typecast(
    df: pd.DataFrame,
    ignore_prefix: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, object]]:
    df = df.copy()

    time_col = df[TIME_COL].astype(np.float32).copy()
    event_col = df[EVENT_COL].astype(np.int8).copy()

    if ignore_prefix is None:
        ignore_prefixes = list(IGNORE_PREFIXES)
    else:
        ignore_prefixes = [ignore_prefix] if str(ignore_prefix).strip() else []
    drop_cols = [
        c for c in df.columns if any(c.startswith(prefix) for prefix in ignore_prefixes)
    ] + IGNORE_COLS
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols + [TIME_COL, EVENT_COL])

    bin_cols = ["ECOG_LAST_OBS"] if "ECOG_LAST_OBS" in df.columns else []

    bin_cols += [c for c in df.columns if set(df[c].unique()) == {0, 1}]
    num_cols = [c for c in df.columns if c not in bin_cols]

    df[num_cols] = df[num_cols].astype(np.float32)
    df[bin_cols] = df[bin_cols].astype(np.int8)

    return df, time_col, event_col, num_cols, bin_cols
