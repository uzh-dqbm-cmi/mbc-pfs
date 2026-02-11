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
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from config import ADMIN_CENSOR_DAYS, EVAL_HORIZONS
from preprocess import preprocess_df
from utils import (
    eval_model,
    evaluate_lot_and_stage_metrics,
    save_eval_metrics,
    save_full_survival_curves,
)

FEATURE_LABELS = json.load(open("data/feature_display_names.json", "r"))


def train_gbsa(
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
    model = GradientBoostingSurvivalAnalysis(
        loss="coxph",
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        subsample=config["subsample"],
        max_features=config["max_features"],
        random_state=config["random_state"],
    )
    model.fit(splits["X_train_np"], splits["y_train_sksurv"])
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

            outdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, outdir / "model.joblib")

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
            gbsa_importances = getattr(model, "feature_importances_", None)
            builtin_items = sorted(
                [
                    (str(name), float(score))
                    for name, score in zip(feature_names, gbsa_importances)
                ],
                key=lambda kv: kv[1],
                reverse=True,
            )
            with open(f"{outdir}/gbsa_built_in_feature_importance.json", "w") as f:
                json.dump(builtin_items, f, indent=4)

    return metrics
