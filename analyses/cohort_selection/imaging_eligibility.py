from pathlib import Path

import numpy as np
import pandas as pd
from SurvivalEVAL.Evaluator import SurvivalEvaluator

from analyses.common.survival import SurvivalContext, ci, load_curves, patient_bootstrap, patient_draws

OUT = Path("results/rebuttal/eligibility")
DATA = Path("data/rebuttal/eligibility")
COHORTS = ["Original 90-day cohort", "Expanded 180-day cohort", "Added 91-180-day group"]
CHANGE = "Change: expanded minus original"


def main():
    original = pd.read_csv("data/design_matrix.csv")
    added = pd.read_csv(DATA / "added_180_day_features.csv")
    frames = [original, pd.concat([original, added], ignore_index=True), added]
    sf, grid = load_curves(original, "full_gbsa")
    with np.load(DATA / "added_180_day_survival.npz") as saved:
        curves = [sf.to_numpy(), np.vstack([sf.to_numpy(), saved["surv"]]), saved["surv"]]
    _, ids, draws = patient_draws(frames[1])
    ibs_times = np.unique(original.loc[original.PFS_EVENT.eq(0), "PFS_TIME_DAYS"])
    rows, intervals, samples = [], [], []
    for name, frame, survival in zip(COHORTS, frames, curves):
        t, e = frame.PFS_TIME_DAYS.to_numpy(float), frame.PFS_EVENT.to_numpy(int)
        cohort_ids = frame.PATIENT_ID.unique()
        result = patient_bootstrap(survival, grid, frame, weights=draws[:, ids.get_indexer(cohort_ids)])
        point, values = result["C"], result["samples"]
        samples.append(values)
        lo, hi = ci(values)
        intervals.append(dict(Cohort=name, C=point, CI_low=lo, CI_high=hi))
        evaluator = SurvivalEvaluator(pred_survs=survival, time_coordinates=grid,
            test_event_times=t, test_event_indicators=e, train_event_times=t, train_event_indicators=e)
        bs = evaluator.brier_score_multiple_points(ibs_times)
        rows.append(dict(Cohort=name, Lines=len(frame), Patients=len(cohort_ids), C=point,
                         IBS=float(np.trapezoid(bs, ibs_times) / (ibs_times[-1] - ibs_times[0]))))
        print(name, point, flush=True)
    lo, hi = ci(samples[1] - samples[0])
    intervals.append(dict(Cohort=CHANGE, C=rows[1]["C"] - rows[0]["C"], CI_low=lo, CI_high=hi))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "eligibility_sensitivity.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "eligibility_C_intervals.csv", index=False)


def imaging_eligibility(context=None):
    cohorts = pd.read_csv(OUT / "eligibility_sensitivity.csv")
    intervals = pd.read_csv(OUT / "eligibility_C_intervals.csv")
    samples = (context or SurvivalContext()).evaluate("full_gbsa")["samples"]
    intervals.loc[intervals.Cohort.eq(COHORTS[0]), ["CI_low", "CI_high"]] = ci(samples)
    return cohorts.merge(intervals.drop(columns="C"), on="Cohort"), intervals[intervals.Cohort.eq(CHANGE)]


if __name__ == "__main__":
    main()
