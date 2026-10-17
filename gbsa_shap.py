from __future__ import annotations

import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from config import DESIGN_MATRIX_PATH, RANDOM_STATE, RESULTS_PATH, SHAP_GBSA_BACKGROUND
from cv_config import K_OUTER, MODEL_CONFIGS
from gbsa import train_gbsa
from preprocess import preprocess_df


def _make_survival_probability_predictor(model: object, target_time: float) -> callable:
    time_grid = np.asarray(getattr(model, "unique_times_", []), dtype=float).ravel()
    target_time = float(target_time)

    matches = np.flatnonzero(np.isclose(time_grid, target_time, atol=1e-6))
    if matches.size == 0:
        closest = float(time_grid[np.argmin(np.abs(time_grid - target_time))])
        raise ValueError(
            f"No exact surv prob available for {target_time} day "
            f"(closest available: {closest} day)."
        )
    target_idx = int(matches[0])

    def _predict(X: np.ndarray) -> np.ndarray:
        surv = model.predict_survival_function(X, return_array=True)
        surv = np.asarray(surv, dtype=float)
        if surv.ndim == 1:
            surv = surv.reshape(1, -1)
        return surv[:, target_idx]

    return _predict


def shap_gbsa(
    splits: dict,
    outdir: Path,
    target_time: float,
) -> None:
    X_train_np = splits["X_train_np"]
    X_test_np = splits["X_test_np"]
    feature_names = splits["numeric_cols"] + splits["binary_cols"]
    model = joblib.load(outdir / "model.joblib")

    random.seed(int(RANDOM_STATE))
    np.random.seed(int(RANDOM_STATE))

    bg_n = max(1, int(min(SHAP_GBSA_BACKGROUND, X_train_np.shape[0])))
    rng_state = np.random.get_state()
    np.random.seed(RANDOM_STATE)
    try:
        background_data = shap.kmeans(X_train_np, bg_n)
    finally:
        np.random.set_state(rng_state)
    background_np = np.asarray(
        getattr(background_data, "data", background_data),
        dtype=float,
    )
    shap_eval_np = X_test_np
    predict_fn = _make_survival_probability_predictor(model, target_time)
    output_target = f"survival_prob_{int(round(float(target_time)))}d"
    target_time_val = float(target_time)
    explainer = shap.PermutationExplainer(
        predict_fn,
        masker=shap.maskers.Independent(background_np),
        feature_names=feature_names,
        seed=RANDOM_STATE,
    )
    explanation = explainer(
        shap_eval_np,
        max_evals="auto",
        main_effects=False,
        error_bounds=False,
        batch_size="auto",
        outputs=None,
        silent=False,
    )
    np.savez_compressed(
        outdir / "shap_values.npz",
        shap_values=explanation.values,
        feature_names=feature_names,
        X=shap_eval_np,
        sample_idx=splits["idx_test"].copy(),
        expected_value=explanation.base_values,
        output_target=output_target,
        target_time=target_time_val,
    )


def _run_outer_fold(
    outer_idx: int,
    eval_entry: dict,
    design_matrix_path: Path,
    gbsa_configs: list[dict],
) -> tuple[int, bool, str]:
    try:
        random.seed(int(RANDOM_STATE) + int(outer_idx))
        np.random.seed(int(RANDOM_STATE) + int(outer_idx))
        design_matrix = pd.read_csv(design_matrix_path)
        design_matrix = design_matrix[design_matrix["PFS_EVENT"] != -1].reset_index(
            drop=True
        )

        split = pd.read_csv(eval_entry["split_path"])
        train_idx = split[split["split"] != "test"]["idx"].to_numpy(dtype=int)
        test_idx = split[split["split"] == "test"]["idx"].to_numpy(dtype=int)

        outdir = Path(eval_entry["results_path"])
        best_config_name = eval_entry.get("name") or outdir.name
        config = next(
            cfg for cfg in gbsa_configs if cfg.get("name") == best_config_name
        )

        model_path = outdir / "model.joblib"
        if not model_path.exists():
            print(
                f"Training GBSA for outer fold {outer_idx} with config {best_config_name}"
            )
            train_gbsa(
                train_idx=train_idx,
                val_idx=test_idx,
                test_idx=test_idx,
                design_matrix=design_matrix,
                config=config,
                outer=True,
                outdir=outdir,
                run_ablation=False,
            )

        splits = preprocess_df(
            train_idx=train_idx,
            val_idx=test_idx,
            test_idx=test_idx,
            design_matrix=design_matrix,
        )
        shap_gbsa(splits=splits, outdir=outdir, target_time=365.0)
        return outer_idx, True, f"Completed outer fold {outer_idx}"
    except Exception as exc:
        return outer_idx, False, f"[ERROR] outer{outer_idx}: {exc}"


if __name__ == "__main__":
    eval_metrics = json.load(open(RESULTS_PATH / "gbsa" / "eval_metrics.json"))[
        "outer_folds"
    ]

    gbsa_configs = MODEL_CONFIGS["gbsa"]

    jobs = {}
    max_workers = min(K_OUTER, 8)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(K_OUTER):
            jobs[
                executor.submit(
                    _run_outer_fold,
                    i,
                    eval_metrics[f"outer{i}"],
                    DESIGN_MATRIX_PATH,
                    gbsa_configs,
                )
            ] = i

        for future in as_completed(jobs):
            outer_idx, ok, msg = future.result()
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {msg}")
