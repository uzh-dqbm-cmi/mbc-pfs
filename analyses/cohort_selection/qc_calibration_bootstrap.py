import json
import joblib
import numpy as np
from lifelines import KaplanMeierFitter
from preprocess import preprocess_df
from analyses.common.survival import patient_draws
from analyses.cohort_selection.qc_refit_eval import Models, OUT, surv_at
from analyses.cohort_selection.qc_refit import OUT as REFIT, build_cohort, fold_rows, load_inputs


def main():
    models = Models()
    info = models.info("late525")
    original, original_grid = models.original_for(info)
    p0, p1 = surv_at(original, original_grid, 365), np.full(len(info), np.nan)
    base, added, meta, folds, splits = load_inputs()
    features, fold_info = build_cohort("late525", base, added, meta)
    for fold in folds:
        train, test = fold_rows(splits[fold], fold_info, fold)
        prep = preprocess_df(train, test, test, features)
        model = joblib.load(REFIT / "late525" / fold / "model.joblib")
        p1[test] = surv_at(model.predict_survival_function(prep["X_test_np"], return_array=True), model.unique_times_, 365)
        print(f"Scored saved {fold} model.", flush=True)
    assert np.isfinite(p1).all()
    results = []
    for subtype in ("HR+/HER2+", "HR+/HER2-"):
        selected = info.FIRST & info.START_CLASS_ALL.eq("Chemotherapy") & info.SUBTYPE_ALL.eq(subtype)
        data = info.loc[selected].reset_index(drop=True)
        a, b = p0[selected], p1[selected]
        time, event = data.PFS_TIME_DAYS.to_numpy(), data.PFS_EVENT.to_numpy()
        pid, ids, weights = patient_draws(data, n_boot=1000, seed=42)
        def calculate(rows):
            observed = float(KaplanMeierFitter().fit(time[rows], event[rows]).predict(365))
            g0, g1 = a[rows].mean() - observed, b[rows].mean() - observed
            return np.array([g0, g1, g1 - g0, abs(g0) - abs(g1)]) * 100
        point = calculate(np.arange(len(data)))
        samples = np.array([calculate(np.repeat(np.arange(len(data)), w[pid].astype(int))) for w in weights])
        names = ["Original mean predicted minus KM", "Refit mean predicted minus KM",
                 "Change in mean predicted PFS (refit minus original)", "Reduction in absolute mean calibration gap"]
        for name, value, draws in zip(names, point, samples.T):
            lo, hi = np.quantile(draws, [.025, .975])
            results.append(dict(subtype=subtype, lines=len(data), patients=len(ids), metric=name,
                                estimate=float(value), ci_low=float(lo), ci_high=float(hi)))
    note = ("Exploratory analysis added after review of the existing results. One-year PFS; full-PFS "
            f"{len(info):,}-line cohort; two first-line chemotherapy-start strata; 1,000 paired patient resamples, seed 42. "
            "KM and mean predictions are recomputed in each resample. Predictions are fixed. All estimates "
            "are percentage points. Percentile intervals do not include model-training uncertainty. "
            "A reduction in the absolute mean gap does not establish improvement in ICI or calibration slope.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration_gap_bootstrap.json").write_text(json.dumps(dict(method=note, results=results), indent=2) + "\n")
    report = note + "\n\n" + "\n".join(f"{r['subtype']} (n={r['lines']}): {r['metric']}: {r['estimate']:+.2f} "
        f"[95% CI {r['ci_low']:+.2f}, {r['ci_high']:+.2f}]" for r in results) + "\n"
    (OUT / "calibration_gap_bootstrap.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
