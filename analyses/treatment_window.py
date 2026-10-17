from pathlib import Path

import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from tabulate import tabulate

import build_pfs
from analyses.common.survival import SurvivalContext, load_curves

OUT = Path("results/rebuttal/landmark/landmark.csv")


def line_groups(d):
    return [("all lines", np.ones(len(d), bool)), ("first lines", d.LINE.eq(1)), ("later lines", d.LINE.gt(1))]


def landmark():
    d = pd.read_csv("data/design_matrix.csv")
    sf, grid = load_curves(d, "full_gbsa")
    reports = build_pfs.prep_progression(set(d.PATIENT_ID))
    positive = {p: np.sort(x.START_DATE.unique()) for p, x in reports.loc[reports.PROGRESSION.eq("Y")].groupby("PATIENT_ID")}
    early = np.array([np.any((positive.get(r.PATIENT_ID, np.array([])) > r.LINE_START) &
                              (positive.get(r.PATIENT_ID, np.array([])) <= r.LINE_START + 28)) for r in d.itertuples()])
    keep = ~early & d.PFS_TIME_DAYS.gt(28)
    j28 = max(0, np.searchsorted(grid, 28, side="right") - 1)
    use = grid >= 28
    conditional = pd.DataFrame(np.clip(sf.to_numpy()[:, use] / sf.iloc[:, j28].to_numpy()[:, None], 0, 1),
                               index=d.index, columns=grid[use] - 28)
    times = np.linspace(90., 365., 100)
    rows = []
    for label, mask in line_groups(d):
        idx = d.index[mask & keep]
        ev = EvalSurv(conditional.loc[idx].T, (d.loc[idx, "PFS_TIME_DAYS"] - 28).to_numpy(),
                      d.loc[idx, "PFS_EVENT"].to_numpy(), censor_surv="km")
        point = ev.brier_score(np.array([180., 365.])).to_numpy()
        bs = ev.brier_score(times).to_numpy()
        rows.append(dict(group=label, n_original=int(np.sum(mask)), n_landmark=len(idx),
            excluded_early_Y=int((mask & early).sum()), excluded_time_equal_28=int((mask & d.PFS_TIME_DAYS.eq(28)).sum()),
            excluded_either=int((mask & ~keep).sum()), C=float(ev.concordance_td("adj_antolini")),
            BS_180_after_landmark=float(point[0]), BS_365_after_landmark=float(point[1]),
            IBS_90_365_after_landmark=float(np.trapezoid(bs, times) / 275)))
    return pd.DataFrame(rows)


def main():
    table = landmark()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))


def fmt(frame):
    return tabulate(frame, headers="keys", showindex=False, tablefmt="psql", floatfmt=".4f")


def generate(context=None):
    d = pd.read_csv(OUT)
    ctx = context or SurvivalContext()
    dm = ctx.dm
    sf = ctx.evaluate("full_gbsa")["sf"]
    baseline = []
    for label, mask in line_groups(dm)[:2]:
        idx = dm.index[mask]
        c = EvalSurv(sf.loc[idx].T, dm.loc[idx, "PFS_TIME_DAYS"].to_numpy(),
                     dm.loc[idx, "PFS_EVENT"].to_numpy()).concordance_td("adj_antolini")
        baseline.append({"Baseline group": label.capitalize(), "C": c})
    counts = d.iloc[:, :6]
    scores = d[["group", "C", "BS_180_after_landmark", "BS_365_after_landmark", "IBS_90_365_after_landmark"]]
    return "\n\n".join([
        "Day-28 landmark: fixed saved models. Exclude early positive reports and follow-up at or before day 28.",
        "The exclusion columns can overlap. The union counts each line once.", fmt(counts),
        "Brier times start at the landmark. IBS uses days 90-365 after the landmark.", fmt(scores),
        "Original pooled held-out C", fmt(baseline),
        "Both development and external eligibility exclude durations below 28 days; follow-up is capped at 730 days.",
        "Recalculate: python -m analyses.treatment_window",
        "Inputs: data/design_matrix.csv; source MSK progression reports; saved GBSA curves."]
    ) + "\n"


if __name__ == "__main__":
    main()
