import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.nonparametric import CensoringDistributionEstimator
from SurvivalEVAL.Evaluator import SurvivalEvaluator

from config import AUC, C_INDEX, EVAL_HORIZONS

# Silence SurvivalEVAL padding warnings about time grids not starting at 0.
warnings.filterwarnings(
    "ignore",
    message="The first time coordinate is not 0.*",
    category=UserWarning,
)


def save_eval_metrics(metrics: Dict[str, Any], outdir: Path) -> str:
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def eval_model(
    time_grid_train_np: np.ndarray,
    surv_test_np: np.ndarray,
    surv_test_df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    eval_pycox = EvalSurv(
        surv=surv_test_df,
        durations=splits["time_test_arr"],
        events=splits["event_test_arr"],
    )
    C_index = float(eval_pycox.concordance_td("adj_antolini"))

    auc_info = compute_time_dependent_auc(
        surv_test_np=surv_test_np,
        time_grid_train_np=time_grid_train_np,
        y_train=splits["y_train_sksurv"].copy(),
        y_test=splits["y_test_sksurv"].copy(),
        eval_horizons=EVAL_HORIZONS,
    )

    surv_evaluator = SurvivalEvaluator(
        pred_survs=surv_test_np,
        time_coordinates=time_grid_train_np,
        test_event_times=splits["time_test_arr"],
        test_event_indicators=splits["event_test_arr"],
        train_event_times=splits["time_train_arr"],
        train_event_indicators=splits["event_train_arr"],
    )

    (
        eval_time_indices,
        ipcw,
        brier_score,
        IBS,
        p_value,
        weighted_mae_margin,
        weighted_mae_po,
    ) = time_dependent_evals(
        eval_times=auc_info["horizon_times"],
        time_grid_train_np=time_grid_train_np,
        y_train=splits["y_train_sksurv"].copy(),
        y_test=splits["y_test_sksurv"].copy(),
        surv_test_np=surv_test_np,
        surv_evaluator=surv_evaluator,
    )

    return {
        "time": eval_time_indices,
        "eval_horizons": [float(x) for x in EVAL_HORIZONS],
        C_INDEX: float(C_index),
        "ipcw": ipcw,
        "auc": auc_info["horizon_values"],
        AUC: float(auc_info["mean_auc"]),
        "censoring_last_time": float(auc_info["censoring_last_time"]),
        "brier_score": brier_score,
        "IBS": IBS,
        "p_value": p_value,
        "weighted_MAE_margin": float(weighted_mae_margin),
        "weighted_MAE_PO": float(weighted_mae_po),
    }


def save_full_survival_curves(
    outdir: str,
    time_grid: np.ndarray,
    surv_test_np: np.ndarray,
    idx_test_array: np.ndarray,
    filename: str = "surv_test.npz",
) -> str:
    """Persist the complete survival matrix aligned to the provided time grid."""

    time_grid = np.asarray(time_grid, dtype=float).reshape(-1)
    surv_test_np = np.asarray(surv_test_np, dtype=float)
    if surv_test_np.shape[0] != idx_test_array.shape[0]:
        raise ValueError("surv_test_np rows must match idx_test_array length.")
    if surv_test_np.ndim != 2 or surv_test_np.shape[1] != time_grid.shape[0]:
        raise ValueError(
            "surv_test_np must be a (n_samples, n_times) array aligned with time_grid."
        )

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, filename)
    np.savez_compressed(
        out_path,
        time=time_grid,
        surv=surv_test_np,
        idx_test=np.asarray(idx_test_array, dtype=int),
    )
    return out_path


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(np.float64)


def time_dependent_evals(
    eval_times: Sequence[float],
    time_grid_train_np: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    surv_test_np: np.ndarray,
    surv_evaluator: SurvivalEvaluator,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    float,
    float,
    float,
    float,
]:
    """
    Evaluate time-dependent IPCW C-index, Brier scores, and classification metrics.

    Parameters
    ----------
    eval_times : Sequence[float]
        Requested evaluation horizons (in days).
    time_grid_train_np : np.ndarray
        Sorted unique event times from the training split (aligns with survival curves).
    y_train, y_test : np.ndarray
        Structured Surv arrays describing (event, time).
    surv_test_np : np.ndarray
        Predicted survival curves (n_test, len(time_grid_train_np)).
    surv_evaluator : SurvivalEvaluator
        Helper for Brier/IBS calculations.
    progression_threshold : float, default=0.5
        Risk cut-off for classifying a sample as progressed (> threshold -> progressed).

    Returns
    -------
    resolved_eval_times : List[float]
        Actual time-grid points used for evaluation.
    ipcw : List[float]
        IPCW-adjusted C-index at each horizon.
    brier_score : List[float]
        Brier score at each horizon.
    precision_at_threshold : List[float]
        Precision achieved when classifying risk > progression_threshold as progressed.
    recall_at_threshold : List[float]
        Recall achieved under the same threshold.
    IBS : float
        Integrated Brier score.
    p_value : float
        D-calibration p-value.
    weighted_MAE_margin : float
        Weighted MAE under the Margin method (truncated at 365 days).
    weighted_MAE_PO : float
        Weighted MAE under pseudo-observation method (truncated at 365 days).
    """
    resolved_eval_times: List[float] = []
    ipcw: List[float] = []
    brier_score: List[float] = []

    for eval_time in eval_times:
        interp_time_index = int(np.argmin(np.abs(eval_time - time_grid_train_np)))
        actual_time = float(time_grid_train_np[interp_time_index])
        resolved_eval_times.append(actual_time)
        surv_values_at_eval_time_np = surv_test_np[:, interp_time_index]
        estimated_risks_np = 1.0 - surv_values_at_eval_time_np

        try:
            cindex, _, _, _, _ = concordance_index_ipcw(
                y_train,
                y_test,
                estimated_risks_np,
                tau=actual_time,
            )
            ipcw.append(float(cindex))
        except Exception as e:
            print(f"[WARNING]: IPCW calculation failed with error: {e}")
            ipcw.append(float("nan"))

        try:
            brier_score.append(float(surv_evaluator.brier_score(actual_time)))
        except Exception as e:
            print(f"[WARNING]: Brier score calculation failed with error: {e}")
            brier_score.append(float("nan"))

    try:
        IBS = float(surv_evaluator.integrated_brier_score())
    except Exception as e:
        print(f"[WARNING]: IBS calculation failed with error: {e}")
        IBS = float("nan")
    try:
        p_value, _ = surv_evaluator.d_calibration()
    except Exception as e:
        print(f"[WARNING]: p_value calculation failed with error: {e}")
        p_value = float("nan")
    try:
        weighted_MAE_margin = float(
            surv_evaluator.mae(method="Margin", weighted=True, truncated_time=730)
        )
        weighted_MAE_PO = float(
            surv_evaluator.mae(method="Pseudo_obs", weighted=True, truncated_time=730)
        )
    except Exception as e:
        print(f"[WARNING]: MAE calculation failed with error: {e}")
        weighted_MAE_margin = float("nan")
        weighted_MAE_PO = float("nan")

    return (
        resolved_eval_times,
        ipcw,
        brier_score,
        IBS,
        float(p_value),
        weighted_MAE_margin,
        weighted_MAE_PO,
    )


def compute_time_dependent_auc(
    surv_test_np: np.ndarray,
    time_grid_train_np: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    eval_horizons: Sequence[float],
) -> Dict[str, object]:
    """Compute td-AUC curve and align it to requested evaluation horizons."""
    censoring_estimator = CensoringDistributionEstimator().fit(y_train)
    censoring_surv = censoring_estimator.predict_proba(time_grid_train_np)
    censoring_valid_mask = censoring_surv > 0.0
    censoring_last_time = float(np.max(time_grid_train_np[censoring_valid_mask]))
    # print(f"Censoring survival support ends at time: {censoring_last_time}")

    event_mask = y_test["event"].astype(bool)
    if not np.any(event_mask):
        raise ValueError("No progressed events in test split; td-AUC is undefined.")

    event_times = y_test["time"][event_mask]
    first_event_time = float(np.min(event_times))
    last_event_time = float(np.max(event_times))
    follow_up_max = float(np.max(y_test["time"]))
    if follow_up_max <= first_event_time:
        raise ValueError(
            "Test follow-up upper bound is not greater than the first progressed event."
        )

    time_window_mask = (time_grid_train_np >= first_event_time) & (
        time_grid_train_np <= last_event_time
    )
    if not np.any(time_window_mask):
        raise ValueError(
            "Time grid has no points within the observed progressed-event window."
        )

    combined_mask = (
        time_window_mask
        & (time_grid_train_np < follow_up_max)
        & (time_grid_train_np <= censoring_last_time)
    )
    if not np.any(combined_mask):
        raise ValueError(
            "Time grid has no points within the observed progressed-event window "
            "after enforcing the test follow-up bound and censoring survival support "
            f"(last valid time: {censoring_last_time})."
        )
    if not np.array_equal(combined_mask, time_window_mask):
        adjusted_eval_time = float(np.max(time_grid_train_np[combined_mask]))
        print(
            f"[WARNING]: Adjusted AUC eval time to {adjusted_eval_time} due to data limits."
        )
    if np.any(time_grid_train_np > censoring_last_time):
        print(
            "[WARNING]: Dropped eval times beyond censoring survival support "
            f"(last valid time: {censoring_last_time})."
        )

    eval_time_grid = time_grid_train_np[combined_mask]
    risk_estimates = 1.0 - surv_test_np
    risk_window = risk_estimates[:, combined_mask]

    try:
        auc_values, mean_auc = cumulative_dynamic_auc(
            y_train,
            y_test,
            risk_window,
            times=eval_time_grid,
        )
    except Exception as e:
        print(
            f"[ERROR]: cumulative td-AUC calculation failed with error under the integrated window: {e}"
        )
        auc_values = np.full(shape=eval_time_grid.shape, fill_value=np.nan)
        mean_auc = float("nan")

    horizon_times: List[float] = []
    horizon_values: List[float] = []
    for horizon in eval_horizons:
        idx = int(np.argmin(np.abs(eval_time_grid - horizon)))
        resolved_time = float(eval_time_grid[idx])
        horizon_times.append(resolved_time)
        horizon_values.append(float(auc_values[idx]))
        if np.isnan(auc_values[idx]):
            # attempt to evaluate at this horizon directly
            try:
                # `idx` is relative to `eval_time_grid` / `risk_window`, not the full time grid.
                risk_at_time = risk_window[:, idx]
                auc_value, _ = cumulative_dynamic_auc(
                    y_train,
                    y_test,
                    risk_at_time,
                    times=[resolved_time],
                )
                horizon_values[-1] = float(auc_value[0])
            except Exception as e:
                print(
                    f"[ERROR]: td-AUC calculation at horizon {horizon} failed with error under the eval horizons: {e}"
                )

    return {
        "eval_time_grid": eval_time_grid.astype(float).tolist(),
        "auc_values": [float(x) for x in auc_values],
        "mean_auc": float(mean_auc),
        "censoring_last_time": censoring_last_time,
        "horizon_times": horizon_times,
        "horizon_values": horizon_values,
    }


def evaluate_lot_and_stage_metrics(
    design_matrix: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    surv_test_df: pd.DataFrame,
    time_grid_train_np: np.ndarray,
    eval_horizons: Sequence[float],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    """
    Evaluate downstream metrics for Line 1 vs. Line 2+ and HR/HER2 subgroups.

    Returns
    -------
    lot_metrics : Dict[str, Dict[str, object]]
        Metrics for LINE_1 and LINE_2_PLUS cohorts (if present).
    hr_her2_metrics : Dict[str, Dict[str, object]]
        Metrics for each observed HR/HER2 combination (HR+/-, HER2+/-).
    """
    lot_metrics: Dict[str, Dict[str, object]] = {}
    hr_her2_metrics: Dict[str, Dict[str, object]] = {}

    if "LINE" not in design_matrix.columns:
        return lot_metrics, hr_her2_metrics

    test_cols = ["LINE"]
    if "HR" in design_matrix.columns:
        test_cols.append("HR")
    if "HER2" in design_matrix.columns:
        test_cols.append("HER2")

    test_meta = (
        design_matrix.loc[splits["idx_test"], test_cols].reset_index(drop=True).copy()
    )
    line_values = pd.to_numeric(test_meta["LINE"], errors="coerce")

    def _subset_metrics(mask: np.ndarray) -> Optional[Dict[str, object]]:
        mask = pd.Series(mask, dtype="boolean").fillna(False).to_numpy(dtype=bool)
        if not mask.any():
            return None

        subset_surv_df = surv_test_df.iloc[:, mask]
        subset_surv_np = subset_surv_df.to_numpy().T
        subset_time = splits["time_test_arr"].copy()[mask]
        subset_event = splits["event_test_arr"].copy()[mask]
        subset_y_test = splits["y_test_sksurv"].copy()[mask]

        surv_eval = SurvivalEvaluator(
            pred_survs=subset_surv_np,
            time_coordinates=time_grid_train_np,
            test_event_times=subset_time,
            test_event_indicators=subset_event,
            train_event_times=splits["time_train_arr"].copy(),
            train_event_indicators=splits["event_train_arr"].copy(),
        )

        auc_info = compute_time_dependent_auc(
            surv_test_np=subset_surv_np,
            time_grid_train_np=time_grid_train_np,
            y_train=splits["y_train_sksurv"].copy(),
            y_test=subset_y_test,
            eval_horizons=eval_horizons,
        )

        (
            eval_times,
            subset_ipcw,
            subset_brier,
            subset_IBS,
            subset_p_value,
            subset_mae_margin,
            subset_mae_po,
        ) = time_dependent_evals(
            eval_times=auc_info["horizon_times"],
            time_grid_train_np=time_grid_train_np,
            y_train=splits["y_train_sksurv"].copy(),
            y_test=subset_y_test,
            surv_test_np=subset_surv_np,
            surv_evaluator=surv_eval,
        )

        eval_pycox = EvalSurv(subset_surv_df, subset_time, subset_event)
        subset_c_td = float(eval_pycox.concordance_td("adj_antolini"))

        return {
            C_INDEX: subset_c_td,
            "eval_times": [float(t) for t in eval_times],
            "ipcw": [float(x) for x in subset_ipcw],
            "auc": [float(x) for x in auc_info["horizon_values"]],
            AUC: float(auc_info["mean_auc"]),
            "brier_score": [float(x) for x in subset_brier],
            "IBS": float(subset_IBS),
            "p_value": float(subset_p_value),
            "weighted_MAE_margin": float(subset_mae_margin),
            "weighted_MAE_PO": float(subset_mae_po),
        }

    lot_masks = {
        "LINE_1": line_values == 1,
        "LINE_2_PLUS": line_values >= 2,
    }
    for label, mask in lot_masks.items():
        lot_result = _subset_metrics(mask)
        if lot_result is not None:
            lot_metrics[label] = lot_result

    if {"HR", "HER2"} <= set(test_meta.columns):
        hr_values = pd.to_numeric(test_meta["HR"], errors="coerce").round()
        her2_values = pd.to_numeric(test_meta["HER2"], errors="coerce").round()
        valid_mask = hr_values.notna() & her2_values.notna()

        def _hr_her2_label(hr: int, her2: int) -> str:
            hr_label = "HR+" if hr == 1 else "HR-"
            her2_label = "HER2+" if her2 == 1 else "HER2-"
            return f"{hr_label}_{her2_label}"

        for hr in (0, 1):
            for her2 in (0, 1):
                combo_mask = valid_mask & (hr_values == hr) & (her2_values == her2)
                combo_result = _subset_metrics(combo_mask)
                if combo_result is not None:
                    hr_her2_metrics[_hr_her2_label(hr, her2)] = combo_result

    return lot_metrics, hr_her2_metrics
