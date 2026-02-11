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
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis

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

MODEL_NAME = "CoxPH"


def train_coxph(
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

    model = CoxPHSurvivalAnalysis(alpha=config["alpha"], ties="efron", n_iter=100_000)
    model.fit(X=splits["X_train_np"], y=splits["y_train_sksurv"])
    surv_test_np = model.predict_survival_function(
        splits["X_test_np"], return_array=True
    )
    time_grid_train_np = np.asarray(model.unique_times_, dtype=float)
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

    if outer and not run_ablation:
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

        feature_names = splits["numeric_cols"] + splits["binary_cols"]
        coefs = pd.Series(model.coef_.reshape(-1), index=feature_names).sort_values()
        with open(f"{outdir}/feature_importance.json", "w") as f:
            json.dump(dict(coefs), f, indent=4)

    return metrics
