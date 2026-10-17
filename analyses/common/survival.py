import argparse
import json
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from pycox.evaluation.concordance import _is_comparable, _is_concordant
from SurvivalEVAL.Evaluator import SurvivalEvaluator

from manuscript_utils import DEEPHIT_BOOTSTRAP_INTERP_GRID, load_oof_surv_only

MODEL_ROOT = Path("results/rebuttal")


def patient_draws(dm, n_boot=1000, seed=42, weights=None):
    pid, ids = pd.factorize(dm.PATIENT_ID, sort=False)
    if weights is None:
        weights = np.random.default_rng(seed).multinomial(
            len(ids), np.full(len(ids), 1 / len(ids)), size=n_boot).astype(float)
    return pid, ids, weights


def prediction_paths(name, folder="results"):
    name = name.removeprefix("full_")
    metadata = Path(folder) / name / "eval_metrics.json"
    if metadata.exists():
        folds = json.loads(metadata.read_text())["outer_folds"]
        paths = {fold: Path(info["results_path"]) / "surv_test.npz" for fold, info in folds.items()}
    else:
        paths = {path.parent.name: path for path in sorted((MODEL_ROOT / name).glob("outer*/surv_test.npz"))}
    if len(paths) != 5:
        raise FileNotFoundError(f"Expected five saved folds for {name}. Run --fit {name} if needed.")
    return paths


def load_curves(dm, name, folder="results", metric_grid=False):
    target = DEEPHIT_BOOTSTRAP_INTERP_GRID if metric_grid and name.removeprefix("full_") == "deephit" else None
    return load_oof_surv_only(dm, list(prediction_paths(name, folder).values()), target_time_grid=target)


def ci(samples):
    low, high = np.quantile(samples, [.025, .975])
    return low, high


def line_missingness_columns(dm):
    return [c for c in dm if c == "IS_MLOT1" or c.endswith(("_MISSING", "_UNKNOWN"))]


def site_columns(dm):
    return [c for c in dm if c.startswith("TUMOR_SITE_") and c.endswith("_IN_WINDOW")]


def ibs_evaluator(curves, grid, dm):
    t, e = dm.PFS_TIME_DAYS.to_numpy(), dm.PFS_EVENT.to_numpy()
    return SurvivalEvaluator(pred_survs=curves, time_coordinates=grid, test_event_times=t,
                             test_event_indicators=e, train_event_times=t, train_event_indicators=e)


@numba.njit
def aggregate(curves, at_time, t, e, pid, p, need_den):
    num = np.zeros((p, p))
    den = np.zeros((p, p)) if need_den else np.zeros((0, 0))
    for i in range(len(t)):
        for j in range(len(t)):
            if i != j and _is_comparable(t[i], t[j], e[i], e[j]):
                a, b = pid[i], pid[j]
                num[a, b] += _is_concordant(
                    curves[i, at_time[i]], curves[j, at_time[i]], t[i], t[j], e[i], e[j])
                if need_den:
                    den[a, b] += 1
    return num, den


def bootstrap_samples(num, den, self_events, weights):
    numerator, denominator = num.copy(), den.copy()
    diagonal = np.diag_indices(len(self_events))
    numerator[diagonal] += self_events
    denominator[diagonal] += self_events
    subtract = weights @ self_events
    return ((np.sum((weights @ numerator) * weights, axis=1) - subtract) /
            (np.sum((weights @ denominator) * weights, axis=1) - subtract))


def patient_bootstrap(curves, grid, dm, n_boot=1000, seed=42, weights=None, den=None):
    t, e = dm.PFS_TIME_DAYS.to_numpy(), dm.PFS_EVENT.to_numpy()
    pid, ids, weights = patient_draws(dm, n_boot, seed, weights)
    ev = EvalSurv(pd.DataFrame(curves.T, index=grid), t, e)
    num, calculated_den = aggregate(curves, ev._duration_idx(), t, e, pid, len(ids), den is None)
    if den is None:
        den = calculated_den
    samples = bootstrap_samples(num, den, np.bincount(pid, weights=e), weights)
    return dict(C=float(num.sum() / den.sum()), samples=samples, num=num, den=den)


class SurvivalContext:
    def __init__(self):
        self.dm = pd.read_csv("data/design_matrix.csv")
        self.pid, self.ids, self.weights = patient_draws(self.dm)
        self.self_events = np.bincount(self.pid, weights=self.dm.PFS_EVENT)
        self.results = {}
        self.den = None

    def evaluate(self, name):
        if name in self.results:
            return self.results[name]
        paths = prediction_paths(name)
        sf, grid = load_curves(self.dm, name)
        curves = sf.to_numpy()
        t, e = self.dm.PFS_TIME_DAYS.to_numpy(), self.dm.PFS_EVENT.to_numpy()
        result = patient_bootstrap(curves, grid, self.dm, weights=self.weights, den=self.den)
        self.den = result["den"]
        fold_c = {}
        for fold, path in paths.items():
            with np.load(path) as saved:
                idx = saved["idx_test"]
                fold_sf = pd.DataFrame(saved["surv"].T, index=saved["time"])
            fold_c[fold] = float(EvalSurv(fold_sf, t[idx], e[idx]).concordance_td("adj_antolini"))
        result.update(IBS=float(ibs_evaluator(curves, grid, self.dm).integrated_brier_score()),
                      sf=sf, grid=grid, fold_C=fold_c)
        self.results[name] = result
        return result

    def pooled(self, names):
        rows = []
        for name in names:
            result = self.evaluate(name)
            low, high = ci(result["samples"])
            rows.append(dict(name=name, C=result["C"], C_low=low, C_high=high, IBS=result["IBS"]))
        return pd.DataFrame(rows).set_index("name")

    def compare(self, a, b):
        first, second = self.evaluate(a), self.evaluate(b)
        low, high = ci(first["samples"] - second["samples"])
        return dict(delta_C=first["C"] - second["C"], CI_low=low, CI_high=high,
                    delta_IBS=first["IBS"] - second["IBS"])


def fit_model(name, force=False):
    import joblib
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from preprocess import preprocess_df

    dm = pd.read_csv("data/design_matrix.csv")
    identity = ["PATIENT_ID", "LINE", "LINE_START", "PFS_TIME_DAYS", "PFS_EVENT"]
    clinical = ["HR", "HER2", "ECOG_LAST_OBS", "TUMOR_SITE_LIVER_IN_WINDOW", "CA15_3_LAST_OBS_LOG"]
    if name == "ref_mlot1_ca15_3":
        frame = dm[identity + clinical + ["IS_MLOT1"]].copy()
    elif name == "clinical_cox":
        frame = dm[identity + clinical + ["ECOG_MISSING"]].copy()
        frame["CLINICAL_LINE_NUMBER"] = dm.LINE
        frame["CA15_3_LATEST_MISSING"] = dm.CA15_3_LAST_OBS_LOG.isna().astype(int)
    elif name == "gbsa_no_line_missingness":
        frame = dm.drop(columns=line_missingness_columns(dm))
    elif name == "gbsa_site_count":
        frame = dm.copy()
        frame["RECENT_METASTATIC_SITE_COUNT"] = dm[site_columns(dm)].sum(axis=1)
    else:
        raise ValueError(name)
    folds = json.loads(Path("results/gbsa/eval_metrics.json").read_text())["outer_folds"]
    for fold, info in folds.items():
        path = MODEL_ROOT / name / fold / "surv_test.npz"
        if path.exists() and not force:
            print(f"Reuse {name}, {fold}.", flush=True)
            continue
        split = pd.read_csv(info["split_path"])
        train, test = (split.loc[split.split.ne("test"), "idx"].to_numpy(),
                       split.loc[split.split.eq("test"), "idx"].to_numpy())
        prep = preprocess_df(train, test, test, frame)
        model = (CoxPHSurvivalAnalysis(alpha=1.0, ties="efron", n_iter=1000)
                 if name in ("ref_mlot1_ca15_3", "clinical_cox") else
                 GradientBoostingSurvivalAnalysis(**joblib.load(Path(info["results_path"]) / "model.joblib").get_params()))
        model.fit(prep["X_train_np"], prep["y_train_sksurv"])
        curves = model.predict_survival_function(prep["X_test_np"], return_array=True)
        keep = model.unique_times_ < 730
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, time=model.unique_times_[keep], surv=curves[:, keep], idx_test=test)
        print(f"Saved {name}, {fold}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", required=True, choices=["ref_mlot1_ca15_3", "clinical_cox", "gbsa_no_line_missingness", "gbsa_site_count"])
    parser.add_argument("--force", action="store_true", help="Replace the saved fold predictions.")
    args = parser.parse_args()
    fit_model(args.fit, args.force)
