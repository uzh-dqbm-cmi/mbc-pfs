"""Score the GENIE validation matrix with the trained MSK GBSA fold models.

The 5 nested-CV fold models (``results/gbsa/outer{0..4}/gbsa-simple-400``) are
clean, deterministic 205-feature views of the MSK design matrix
(``typecast`` with ``RECEIVED_`` dropped). We:

1. align the GENIE matrix to the MSK training schema,
2. fit the MSK preprocessing (NaN-robust scaling) on the *full* MSK cohort and
   transform GENIE through it (the per-fold scalers were not persisted; the
   full-cohort scaler is a deterministic, auditable stand-in),
3. score each fold model on GENIE and emit gbsa.py-style artifacts
   (``eval_metrics.json`` + ``surv_test.npz``) per fold, plus a cross-fold
   summary, a 365-day risk table, and a calibration plot.

Each external GENIE center is scored as its own independent cohort (a fresh
``MSK-train + one-center-test`` frame per center). GENIE-MSK is excluded because
those patients overlap the MSK-CHORD training set, so only the truly external
centers (DFCI, VICC) are evaluated. Outputs are written under
``outdir/<CENTER>/`` with a combined ``summary_by_institution.json``.

Run from the repository root:

    python -m genie_validation.run_gbsa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402

from config import ADMIN_CENSOR_DAYS, EVAL_HORIZONS, EVENT_COL, TIME_COL
from preprocess import preprocess_df
from utils import (
    eval_model,
    evaluate_lot_and_stage_metrics,
    save_eval_metrics,
    save_full_survival_curves,
)
from . import paths

DEFAULT_FOLD_GLOB = "results/gbsa/outer*/gbsa-simple-400/model.joblib"
DEFAULT_OUTDIR = Path("validation_outputs/genie_gbsa")
RISK_HORIZON = 365.0


def _load_msk(msk_path: Path) -> pd.DataFrame:
    msk = pd.read_csv(msk_path)
    return msk[msk[EVENT_COL] != -1].reset_index(drop=True)


def _institution_of(genie: pd.DataFrame) -> pd.Series:
    """GENIE contributing center from the sample id prefix (GENIE-<CENTER>-...)."""
    return genie["PATIENT_ID"].astype(str).str.split("-").str[1].fillna("UNKNOWN")


def _align_frames(msk: pd.DataFrame, genie: pd.DataFrame):
    """Stack MSK training rows above the GENIE test rows on the MSK schema."""
    schema = list(msk.columns)  # MSK training schema (authoritative)
    feature_set = set(paths.msk_feature_columns())
    missing = [c for c in feature_set if c not in genie.columns]
    if missing:
        raise ValueError(f"GENIE matrix missing MSK feature columns: {missing}")
    genie_aligned = genie.reindex(columns=schema)
    combined = pd.concat([msk[schema], genie_aligned], ignore_index=True)
    return combined, len(msk)


def _survival_at(surv_np: np.ndarray, time_grid: np.ndarray, horizon: float) -> np.ndarray:
    idx = int(np.searchsorted(time_grid, horizon, side="right") - 1)
    idx = max(0, min(idx, len(time_grid) - 1))
    return surv_np[:, idx]


def _score_fold(model, splits) -> Dict[str, object]:
    x_test = splits["X_test_np"]
    if model.n_features_in_ != x_test.shape[1]:
        raise ValueError(
            f"Model expects {model.n_features_in_} features but GENIE provides "
            f"{x_test.shape[1]}; fold is not the clean 205-feature view."
        )
    surv_np = model.predict_survival_function(x_test, return_array=True)
    time_grid = np.asarray(model.unique_times_, dtype=float)
    mask = time_grid < float(ADMIN_CENSOR_DAYS)
    surv_np = surv_np[:, mask]
    time_grid = time_grid[mask]
    surv_df = pd.DataFrame(surv_np.T, index=time_grid)
    return {"surv_np": surv_np, "time_grid": time_grid, "surv_df": surv_df}


def _core_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    keep = {}
    for k in ("C", "mean_auc", "IBS", "weighted_MAE_margin", "weighted_MAE_PO"):
        if k in metrics and metrics[k] is not None:
            keep[k] = float(metrics[k])
    return keep


def _score_cohort(
    msk: pd.DataFrame,
    genie: pd.DataFrame,
    fold_paths: List[Path],
    outdir: Path,
) -> Dict[str, object]:
    """Score all fold models against one GENIE cohort and write its artifacts.

    A fresh ``(MSK-train + this-cohort-test)`` frame is preprocessed here, so each
    cohort is an independent external evaluation that shares nothing with any
    other cohort except the (fixed) MSK training rows and the trained models.
    """
    combined, n_train = _align_frames(msk, genie)
    n_total = len(combined)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_total)

    splits = preprocess_df(
        train_idx=train_idx,
        val_idx=test_idx,
        test_idx=test_idx,
        design_matrix=combined,
        ignore_prefix=None,
    )
    n_genie = len(test_idx)
    if splits["X_test_np"].shape[0] != n_genie:
        raise ValueError("Preprocessing dropped GENIE rows unexpectedly.")

    outdir.mkdir(parents=True, exist_ok=True)
    per_fold: Dict[str, Dict[str, float]] = {}
    risk_365 = np.zeros((n_genie, len(fold_paths)), dtype=float)
    for i, fpath in enumerate(fold_paths):
        model = joblib.load(fpath)
        scored = _score_fold(model, splits)
        metrics = eval_model(
            time_grid_train_np=scored["time_grid"],
            surv_test_np=scored["surv_np"],
            surv_test_df=scored["surv_df"],
            splits=splits,
        )
        try:
            lot_metrics, hr_her2_metrics = evaluate_lot_and_stage_metrics(
                design_matrix=combined,
                splits=splits,
                surv_test_df=scored["surv_df"],
                time_grid_train_np=scored["time_grid"],
                eval_horizons=EVAL_HORIZONS,
            )
            metrics["lot_metrics"] = lot_metrics
            metrics["hr_her2_metrics"] = hr_her2_metrics
        except Exception as exc:  # subgroup metrics are best-effort
            metrics["subgroup_metrics_error"] = str(exc)
        metrics["fold_model"] = str(fpath)

        fold_dir = outdir / f"fold{i}"
        save_eval_metrics(metrics=metrics, outdir=fold_dir)
        save_full_survival_curves(
            outdir=str(fold_dir),
            time_grid=scored["time_grid"],
            surv_test_np=scored["surv_np"],
            idx_test_array=test_idx,
            filename="surv_test.npz",
        )
        per_fold[f"fold{i}"] = _core_metrics(metrics)
        risk_365[:, i] = 1.0 - _survival_at(
            scored["surv_np"], scored["time_grid"], RISK_HORIZON
        )
        print(f"  [fold{i}] {Path(fpath).parts[-3]}: {per_fold[f'fold{i}']}")

    summary = _summarize(per_fold, n_genie, genie)
    with open(outdir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_risk_and_calibration(outdir, genie, risk_365)
    return summary


# GENIE centers whose patients overlap the MSK-CHORD training cohort and are
# therefore NOT independent external data (GENIE-MSK is the MSK-CHORD source).
EXCLUDE_INSTITUTIONS: Tuple[str, ...] = ("MSK",)


def run(
    msk_path: Path = paths.MSK_DESIGN_MATRIX,
    genie_path: Path = paths.DESIGN_MATRIX_OUT,
    fold_glob: str = DEFAULT_FOLD_GLOB,
    outdir: Path = DEFAULT_OUTDIR,
    exclude_institutions: Tuple[str, ...] = EXCLUDE_INSTITUTIONS,
) -> Dict[str, object]:
    """Score each external GENIE center as its own independent cohort.

    For every center not in ``exclude_institutions`` a separate cohort dataframe
    is built and scored via :func:`_score_cohort`; results are written under
    ``outdir/<CENTER>/`` and collected in ``summary_by_institution.json``.
    GENIE-MSK is excluded by default (it overlaps the MSK-CHORD training set).
    """
    msk = _load_msk(msk_path)
    genie = pd.read_csv(genie_path)
    institution = _institution_of(genie)

    fold_paths = sorted(Path().glob(fold_glob))
    if not fold_paths:
        raise FileNotFoundError(f"No fold models matched {fold_glob}")
    outdir.mkdir(parents=True, exist_ok=True)

    centers = [c for c in sorted(institution.unique()) if c not in exclude_institutions]
    print(f"  external centers: {centers}  (excluded: {list(exclude_institutions)})")

    summaries: Dict[str, object] = {}
    for name in centers:
        sub = genie[(institution == name).to_numpy()].reset_index(drop=True)
        print(f"\n== {name}: {len(sub)} lines / {sub['PATIENT_ID'].nunique()} patients ==")
        summary = _score_cohort(msk, sub, fold_paths, outdir / name)
        summaries[name] = summary
        print(f"  [{name}] cross-fold C-index: {summary['mean'].get('C'):.4f} "
              f"+/- {summary['std'].get('C'):.4f}")

    with open(outdir / "summary_by_institution.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"\n  wrote {outdir}/summary_by_institution.json")
    return summaries


def _summarize(per_fold, n_genie, genie) -> Dict[str, object]:
    keys = sorted({k for f in per_fold.values() for k in f})
    mean = {k: float(np.mean([f[k] for f in per_fold.values() if k in f])) for k in keys}
    std = {k: float(np.std([f[k] for f in per_fold.values() if k in f])) for k in keys}
    return {
        "n_genie_lines": int(n_genie),
        "n_genie_patients": int(genie["PATIENT_ID"].nunique()),
        "event_distribution": {int(k): int(v) for k, v in genie[EVENT_COL].value_counts().items()},
        "per_fold": per_fold,
        "mean": mean,
        "std": std,
    }


def _write_risk_and_calibration(outdir: Path, genie: pd.DataFrame, risk_365: np.ndarray) -> None:
    mean_risk = risk_365.mean(axis=1)
    risk_df = pd.DataFrame(
        {
            "PATIENT_ID": genie["PATIENT_ID"].values,
            "LINE": genie["LINE"].values,
            "PFS_TIME_DAYS": genie[TIME_COL].values,
            "PFS_EVENT": genie[EVENT_COL].values,
            "PRED_RISK_365D": mean_risk,
        }
    )
    risk_df.to_csv(outdir / "genie_365d_risk.csv", index=False)

    # calibration: observed KM event prob at 365d vs mean predicted risk, by decile
    try:
        deciles = pd.qcut(mean_risk, q=10, labels=False, duplicates="drop")
        obs, pred = [], []
        kmf = KaplanMeierFitter()
        for d in sorted(pd.unique(deciles[~pd.isna(deciles)])):
            m = deciles == d
            if m.sum() < 5:
                continue
            kmf.fit(risk_df["PFS_TIME_DAYS"][m], risk_df["PFS_EVENT"][m])
            surv365 = float(kmf.predict(RISK_HORIZON))
            obs.append(1.0 - surv365)
            pred.append(float(mean_risk[m].mean()))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
        ax.scatter(pred, obs, color="tab:blue")
        ax.set_xlabel("Mean predicted 365d risk")
        ax.set_ylabel("Observed 365d event rate (KM)")
        ax.set_title("GENIE external validation calibration (365d)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "calibration_365d.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  calibration plot skipped: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msk-design", type=Path, default=paths.MSK_DESIGN_MATRIX)
    parser.add_argument("--genie-design", type=Path, default=paths.DESIGN_MATRIX_OUT)
    parser.add_argument("--fold-glob", type=str, default=DEFAULT_FOLD_GLOB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--exclude-institutions", nargs="*", default=list(EXCLUDE_INSTITUTIONS),
        help="GENIE centers to drop (default: MSK, which overlaps MSK-CHORD).",
    )
    args = parser.parse_args()
    print("Scoring GENIE external centers with MSK GBSA fold models...")
    run(args.msk_design, args.genie_design, args.fold_glob, args.outdir,
        tuple(args.exclude_institutions))
    print("Done.")


if __name__ == "__main__":
    main()
