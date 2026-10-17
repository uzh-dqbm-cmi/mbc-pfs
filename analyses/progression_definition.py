import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv

import build_pfs as B
from config import DAY_SPLIT, LINE_START_WINDOW_DAYS
from analyses.common import cohort as C
from analyses.common.survival import load_curves, prediction_paths

WINDOWS = (30, 60, 90)
RULES = ["frozen"] + [f"no_negative_{d}" for d in WINDOWS] + ["midpoint"]


def modeled_lines(dm):
    return dm.merge(C.load_breast_pfs()[["PATIENT_ID", "LINE", "EVENT_DAY", "LINE_SOURCE"]],
                    on=["PATIENT_ID", "LINE"], validate="one_to_one")


def imaging_event(lines):
    return lines.PFS_EVENT.eq(1) & lines.LINE_SOURCE.eq("labelled_by_progression_event")


def no_negative_stream(g, window):
    ys = g.loc[g.PROGRESSION.eq("Y"), "START_DATE"].unique()
    ns = g.loc[g.PROGRESSION.eq("N"), "START_DATE"].to_numpy()
    accepted = {d for d in ys if not np.any((ns > d) & (ns <= d + window))}
    return g[g.PROGRESSION.eq("N") | g.START_DATE.isin(accepted)]


def derive(dm, first_only=True):
    keep = dm.LINE.eq(1) if first_only else dm.LINE.notna()
    first = modeled_lines(dm[keep])
    first.index = dm.index[keep]
    first["IMAGING_EVENT"] = imaging_event(first)
    ids = set(first.PATIENT_ID)
    tx = pd.read_csv(C.MSK_DIR / "data_timeline_treatment.tsv", sep="\t",
                     usecols=["PATIENT_ID", "START_DATE", "AGENT", "SUBTYPE"])
    tx = tx[tx.PATIENT_ID.isin(ids) & tx.SUBTYPE.ne("Bone Treatment") & tx.AGENT.ne("INVESTIGATIVE")]
    tx["START_DATE"] = tx.START_DATE.astype(int)
    tx = dict(tuple(tx.sort_values(["PATIENT_ID", "START_DATE"], kind="mergesort").groupby("PATIENT_ID")))
    progression = B.prep_progression(ids)[["PATIENT_ID", "START_DATE", "PROGRESSION"]]
    progression = dict(tuple(progression.groupby("PATIENT_ID")))
    deaths = B.prep_death_times(ids)
    B.prior_measurement_map = None
    rows, labelled = [], {window: [] for window in WINDOWS}
    for r in first.itertuples():
        g = progression[r.PATIENT_ID]
        tail = tx[r.PATIENT_ID].loc[tx[r.PATIENT_ID].START_DATE.ge(r.LINE_START)].reset_index(drop=True)
        if tail.empty or tail.START_DATE.iloc[0] != r.LINE_START:
            raise ValueError(f"No treatment record at the selected start: {r.PATIENT_ID}")
        row = dict(index=r.Index, frozen_t=r.PFS_TIME_DAYS, frozen_e=r.PFS_EVENT)
        for window in WINDOWS:
            _, labels = B.assign_patient_lines_and_pfs(tail, no_negative_stream(g, window),
                                                       DAY_SPLIT, LINE_START_WINDOW_DAYS, deaths.get(r.PATIENT_ID))
            labelled[window].append(labels[0])
        midpoint = float(r.PFS_TIME_DAYS)
        if r.IMAGING_EVENT:
            ndays = g.loc[g.PROGRESSION.eq("N") & g.START_DATE.lt(r.EVENT_DAY), "START_DATE"]
            lower = max(r.LINE_START, ndays.max()) if len(ndays) else r.LINE_START
            midpoint = (lower + r.EVENT_DAY) / 2 - r.LINE_START
        row["midpoint_t"], row["midpoint_e"] = midpoint, r.PFS_EVENT
        rows.append(row)
    out = pd.DataFrame(rows).set_index("index")
    for window in WINDOWS:
        capped = B.apply_censor(pd.DataFrame(labelled[window], index=out.index).astype({"PFS_TIME_DAYS": float}))
        out[f"no_negative_{window}_t"], out[f"no_negative_{window}_e"] = capped.PFS_TIME_DAYS, capped.PFS_EVENT
    return out


def concordance(r):
    return f"{r.C_mean:.3f} ± {r.C_sd:.3f}"


def brier(r):
    return f"{r.BS_mean:.3f} ± {r.BS_sd:.3f}"


def sensitivity(dm, first_only=True):
    d = derive(dm, first_only)
    paths = prediction_paths("gbsa").values()
    survival, _ = load_curves(dm, "gbsa")
    rows = []
    for rule in RULES:
        first = d[d[rule + "_e"].isin([0, 1]) & d[rule + "_t"].gt(0)]
        folds = []
        for path in paths:
            with np.load(path) as saved:
                idx = first.index.intersection(saved["idx_test"])
            x = d.loc[idx]
            ev = EvalSurv(survival.loc[idx].T, x[rule + "_t"].to_numpy(),
                          x[rule + "_e"].to_numpy(), censor_surv="km")
            folds.append([ev.concordance_td("adj_antolini"), ev.brier_score(np.array([365.])).iloc[0]])
        folds = np.array(folds)
        rows.append(dict(rule=rule, **C.km(first[rule + "_t"], first[rule + "_e"]),
                         C_mean=folds[:, 0].mean(), C_sd=folds[:, 0].std(ddof=0),
                         BS_mean=folds[:, 1].mean(), BS_sd=folds[:, 1].std(ddof=0)))
    return pd.DataFrame(rows)
