# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import ADMIN_CENSOR_DAYS, EVAL_HORIZONS
from preprocess import preprocess_df
from utils import (
    eval_model,
    evaluate_lot_and_stage_metrics,
    save_eval_metrics,
    save_full_survival_curves,
)

# Silence SurvivalEVAL padding warnings about non-zero starting time grids.
warnings.filterwarnings(
    "ignore",
    message="The first time coordinate is not 0.*",
    category=UserWarning,
)

MODEL_NAME = "DRYRUN"


def train_dryrun(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    design_matrix: pd.DataFrame,
    config: dict,
    outer: bool,
    outdir: Path,
    run_ablation: bool,
) -> dict:
    splits = preprocess_df(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        design_matrix=design_matrix,
        ignore_prefix=config.get("ignore_prefix"),
    )
    rng = np.random.default_rng(config["seed"])
    n_test = len(test_idx)
    time_grid_train_np = np.sort(np.unique(splits["time_train_arr"].astype(float)))

    hazard_increments = rng.uniform(0.001, 0.05, size=(n_test, len(time_grid_train_np)))
    cumulative_hazard = np.maximum.accumulate(np.cumsum(hazard_increments, axis=1))
    cumulative_hazard[:, 0] = 0.0
    surv_test_np = np.exp(-cumulative_hazard)
    horizon_mask = time_grid_train_np < float(ADMIN_CENSOR_DAYS)
    surv_test_np = surv_test_np[:, horizon_mask]
    time_grid_train_np = time_grid_train_np[horizon_mask]
    surv_test_df = pd.DataFrame(surv_test_np.T, index=time_grid_train_np)

    metrics = eval_model(
        time_grid_train_np=time_grid_train_np,
        surv_test_np=surv_test_np,
        surv_test_df=surv_test_df,
        splits=splits,
    )
    metrics["config_name"] = config["name"]
    if outer:
        if not run_ablation:
            lot_metrics, hr_her2_metrics = evaluate_lot_and_stage_metrics(
                design_matrix=design_matrix,
                splits=splits,
                surv_test_df=surv_test_df,
                time_grid_train_np=time_grid_train_np,
                eval_horizons=EVAL_HORIZONS,
            )
            metrics["lot_metrics"] = lot_metrics
            metrics["hr_her2_metrics"] = hr_her2_metrics

            save_eval_metrics(
                metrics=metrics,
                outdir=outdir,
            )

            save_full_survival_curves(
                outdir=outdir,
                time_grid=time_grid_train_np,
                surv_test_np=surv_test_np,
                idx_test_array=test_idx,
                filename="surv_test.npz",
            )

    return metrics
