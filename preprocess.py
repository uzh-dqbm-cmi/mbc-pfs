from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from attr import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sksurv.util import Surv

from config import ADMIN_CENSOR_DAYS
from data_paths import results_root as resolve_results_root


@dataclass
class PreprocessResult:
    X_train: np.ndarray
    Y_train: np.ndarray
    D_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    D_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    D_test: np.ndarray
    numeric_cols: List[str]
    binary_cols: List[str]


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
) -> PreprocessResult:
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


def _progression_rate_quartiles(
    df: pd.DataFrame,
    group_col: str,
    event_col: str,
) -> pd.Series:
    """
    Compute per-patient progression rates (proportion of progressed LoTs) and bucket into quartiles.
    """
    missing = [col for col in (group_col, event_col) if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns for progression rate stratification: {missing}"
        )

    df2 = df[[group_col, event_col]].copy()
    df2[group_col] = df2[group_col].astype(str)
    df2["_event_numeric"] = pd.to_numeric(df2[event_col], errors="raise").astype(float)
    grouped = df2.groupby(group_col, sort=False)
    progressed = grouped["_event_numeric"].sum()
    total = grouped["_event_numeric"].count().clip(lower=1)
    rates = (progressed / total).fillna(0.0)
    try:
        quartiles = pd.qcut(rates, q=4, labels=False, duplicates="drop")
    except ValueError:
        quartiles = pd.Series(
            np.zeros(len(rates), dtype=int),
            index=rates.index,
        )
    return quartiles.fillna(0).astype(int)


def _hr_her2_progress_quartile_strata(
    df: pd.DataFrame,
    group_col: str,
    event_col: str,
    hr_col: str = "HR",
    her2_col: str = "HER2",
) -> np.ndarray:
    """
    Stratify patients by HR/HER2 status and progression-rate quartiles.
    """
    missing = [
        col for col in (event_col, hr_col, her2_col, group_col) if col not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns for stratification: {missing}")

    quartiles = _progression_rate_quartiles(
        df, group_col=group_col, event_col=event_col
    )
    quartile_map = {str(pid): int(q) for pid, q in quartiles.items()}

    group_labels: Dict[str, str] = {}
    for pid, grp in df.groupby(group_col, sort=False):
        pid_str = str(pid)
        progress_label = f"Q{quartile_map.get(pid_str, 0)}"
        hr_label = _binary_label(grp[hr_col], name=hr_col)
        her2_label = _binary_label(grp[her2_col], name=her2_col)
        group_labels[pid_str] = f"{progress_label}_HR{hr_label}_HER2{her2_label}"

    return df[group_col].astype(str).map(group_labels).to_numpy()


def _pick_fold_by_size(splits, target_frac: float, n_samples: int):
    target = target_frac * n_samples
    best = None
    best_gap = float("inf")
    for tr, te in splits:
        gap = abs(len(te) - target)
        if gap < best_gap:
            best = (tr, te)
            best_gap = gap
    return best


def group_train_val_test_split(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Dict[str, Any]:
    """Two-stage, group-exclusive, stratified by (any admin-censored line × HR × HER2)."""

    # feature columns
    reserved = {time_col, event_col, group_col}
    feature_cols = [c for c in df.columns if c not in reserved]
    strata = _hr_her2_admin_strata(
        df, group_col=group_col, time_col=time_col, event_col=event_col
    )

    groups_all = df[group_col].astype(str).to_numpy()

    # sanity
    assert len(groups_all) == len(df) == len(strata)

    # ---- Stage 1: train+val vs test ----
    n_splits1 = max(2, int(round(1.0 / test_size)))
    sgkf1 = StratifiedGroupKFold(
        n_splits=n_splits1, shuffle=True, random_state=random_state
    )
    trva_idx, te_idx = _pick_fold_by_size(
        sgkf1.split(df[feature_cols], strata, groups=groups_all),
        target_frac=test_size,
        n_samples=len(df),
    )

    # ---- Stage 2: train vs val within train+val ----
    y_trva = strata[trva_idx]
    groups_trva = groups_all[trva_idx]
    n_trva = len(trva_idx)
    val_rel = val_size / (1.0 - test_size)
    n_splits2 = max(2, int(round(1.0 / val_rel)))
    sgkf2 = StratifiedGroupKFold(
        n_splits=n_splits2, shuffle=True, random_state=random_state
    )
    tr_idx_local, va_idx_local = _pick_fold_by_size(
        sgkf2.split(df.iloc[trva_idx][feature_cols], y_trva, groups=groups_trva),
        target_frac=val_rel,
        n_samples=n_trva,
    )
    tr_idx = trva_idx[tr_idx_local]
    va_idx = trva_idx[va_idx_local]

    # pack
    def _pack(idxs):
        X = df.iloc[idxs][feature_cols]
        t = df.iloc[idxs][time_col].to_numpy(dtype=float)
        e = df.iloc[idxs][event_col].to_numpy(dtype=bool)
        y = Surv.from_arrays(event=e, time=t)
        return X, y, t, e, np.asarray(idxs, dtype=int)

    X_train, y_train, t_train, e_train, idx_train = _pack(tr_idx)
    X_val, y_val, t_val, e_val, idx_val = _pack(va_idx)
    X_test, y_test, t_test, e_test, idx_test = _pack(te_idx)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "time_train": t_train,
        "event_train": e_train,
        "idx_train": idx_train,
        "X_val": X_val,
        "y_val": y_val,
        "time_val": t_val,
        "event_val": e_val,
        "idx_val": idx_val,
        "X_test": X_test,
        "y_test": y_test,
        "time_test": t_test,
        "event_test": e_test,
        "idx_test": idx_test,
        "strata_all": strata,
    }


def group_train_val_test_split_classification(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
    hr_col: str = "HR",
    her2_col: str = "HER2",
) -> Dict[str, Any]:
    """
    Group-exclusive split stratified by HR/HER2/progression-rate quartiles.
    """
    reserved = {time_col, event_col, group_col}
    feature_cols = [c for c in df.columns if c not in reserved]
    strata = _hr_her2_progress_quartile_strata(
        df,
        group_col=group_col,
        event_col=event_col,
        hr_col=hr_col,
        her2_col=her2_col,
    )

    groups_all = df[group_col].astype(str).to_numpy()
    assert len(groups_all) == len(df) == len(strata)

    n_splits1 = max(2, int(round(1.0 / test_size)))
    sgkf1 = StratifiedGroupKFold(
        n_splits=n_splits1, shuffle=True, random_state=random_state
    )
    trva_idx, te_idx = _pick_fold_by_size(
        sgkf1.split(df[feature_cols], strata, groups=groups_all),
        target_frac=test_size,
        n_samples=len(df),
    )

    y_trva = strata[trva_idx]
    groups_trva = groups_all[trva_idx]
    n_trva = len(trva_idx)
    val_rel = val_size / (1.0 - test_size)
    n_splits2 = max(2, int(round(1.0 / val_rel)))
    sgkf2 = StratifiedGroupKFold(
        n_splits=n_splits2, shuffle=True, random_state=random_state
    )
    tr_idx_local, va_idx_local = _pick_fold_by_size(
        sgkf2.split(df.iloc[trva_idx][feature_cols], y_trva, groups=groups_trva),
        target_frac=val_rel,
        n_samples=n_trva,
    )
    tr_idx = trva_idx[tr_idx_local]
    va_idx = trva_idx[va_idx_local]

    def _pack(idxs):
        X = df.iloc[idxs][feature_cols]
        t = df.iloc[idxs][time_col].to_numpy(dtype=float)
        e = df.iloc[idxs][event_col].to_numpy(dtype=bool)
        y = Surv.from_arrays(event=e, time=t)
        return X, y, t, e, np.asarray(idxs, dtype=int)

    X_train, y_train, t_train, e_train, idx_train = _pack(tr_idx)
    X_val, y_val, t_val, e_val, idx_val = _pack(va_idx)
    X_test, y_test, t_test, e_test, idx_test = _pack(te_idx)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "time_train": t_train,
        "event_train": e_train,
        "idx_train": idx_train,
        "X_val": X_val,
        "y_val": y_val,
        "time_val": t_val,
        "event_val": e_val,
        "idx_val": idx_val,
        "X_test": X_test,
        "y_test": y_test,
        "time_test": t_test,
        "event_test": e_test,
        "idx_test": idx_test,
        "strata_all": strata,
    }


# ----------------- Filtering helpers (age/modality) -----------------


def _drop_modalities(
    df: pd.DataFrame, features_dict: Dict[str, List[str]], exclude: List[str]
) -> pd.DataFrame:
    """Drop columns belonging to specified modalities using FEATURES_DICT keys only.

    Only modalities present as keys in features_dict are considered, and their
    associated columns are dropped. No prefix-based or header substring matching
    is performed to keep behavior simple and explicit.
    """
    if not exclude:
        return df
    df = df.copy()
    ex = {e.strip(): e.strip() for e in exclude if e and str(e).strip()}
    to_drop: List[str] = []
    for k in ex:
        if k in features_dict:
            to_drop.extend([c for c in features_dict[k] if c in df.columns])
    if to_drop:
        df = df.drop(columns=sorted(set(to_drop)))
    return df


def filter_rows_by_age_range(
    df: pd.DataFrame,
    age_range: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Filter rows by an inclusive age range; returns (filtered_df, label)."""
    if age_range is None:
        return df.reset_index(drop=True), None
    if "AGE" not in df.columns:
        raise KeyError("AGE column required for age_range filtering.")
    lo, hi = map(float, age_range)
    df2 = df.loc[(df["AGE"] >= lo) & (df["AGE"] <= hi)].copy()
    age_label = f"age_{int(lo)}_{int(hi)}"
    return df2.reset_index(drop=True), age_label


def apply_filters(
    df: pd.DataFrame,
    features_dict: Dict[str, List[str]],
    exclude_modalities: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply column-level filters (modalities and explicit columns)."""
    df2 = _drop_modalities(df, features_dict, exclude_modalities or [])
    # Drop explicitly named columns, if requested (e.g., AGE)
    if exclude_columns:
        print(f"excluding columns: {exclude_columns}")
        drop_exact = [c for c in exclude_columns if c in df2.columns]
        if drop_exact:
            df2 = df2.drop(columns=drop_exact)
    return df2


def build_results_subdir(
    model_name: str,
    age_label: Optional[str],
    exclude_modalities: Optional[List[str]] = None,
    group_label: Optional[str] = None,
    results_root: Optional[str] = None,
) -> str:
    """Build results subdir name under results/{model_name}/.

    Precedence: group_label if provided; else age_label or 'full', with suffixes '-no_<mod>'.
    """
    root = results_root or resolve_results_root()
    base = group_label or age_label or "full"
    ex = [m.strip().lower() for m in (exclude_modalities or []) if m and str(m).strip()]
    if ex and not group_label:
        base += "-" + "-".join(sorted({f"no_{m}" for m in ex}))
    return f"{root}/{model_name}/{base}"


# ----------------- 2) Typecasting -----------------

from config import EVENT_COL, IGNORE_COLS, IGNORE_PREFIXES, TIME_COL


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
