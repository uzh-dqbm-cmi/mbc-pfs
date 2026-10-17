import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv

import build_pfs as B
from config import DAY_SPLIT, LINE_START_WINDOW_DAYS
from analyses.common import agreement as G, bpc_data as BPC, cohort as C
from analyses.common.survival import load_curves, prediction_paths
from analyses.reconstruction import km_ci, load_bpc_endpoints, show

WINDOWS = (30, 60, 90)
RULES = ["frozen"] + [f"no_negative_{d}" for d in WINDOWS] + ["midpoint"]
THRESHOLDS = (.90, .99)


def modeled_lines(dm):
    return dm.merge(C.load_breast_pfs()[["PATIENT_ID", "LINE", "EVENT_DAY", "LINE_SOURCE"]],
                    on=["PATIENT_ID", "LINE"], validate="one_to_one")


def imaging_event(lines):
    return lines.PFS_EVENT.eq(1) & lines.LINE_SOURCE.eq("labelled_by_progression_event")


def report_counts(dm):
    original = C.load_breast_pfs()
    lines = modeled_lines(dm)
    lines["END"] = lines.EVENT_DAY.astype(float).where(imaging_event(lines), lines.LINE_START + lines.PFS_TIME_DAYS)
    starts = pd.concat([original, C.load_dropped_pfs_lines()])[
        ["PATIENT_ID", "LINE_START"]].drop_duplicates().sort_values(["PATIENT_ID", "LINE_START"])
    starts["NEXT_START"] = starts.groupby("PATIENT_ID").LINE_START.shift(-1)
    lines = lines.merge(starts, on=["PATIENT_ID", "LINE_START"], validate="one_to_one")
    reports = dict(tuple(C.load_msk_progression(set(lines.PATIENT_ID)).groupby("PATIENT_ID")))
    to_end, between = [], []
    for r in lines.itertuples():
        g = reports[r.PATIENT_ID]
        inside = g.DAY.gt(r.LINE_START) & g.PROGRESSION.isin(["Y", "N"])
        to_end.append(int((inside & g.DAY.le(r.END)).sum()))
        between.append(int((inside & (g.DAY.le(r.NEXT_START) if pd.notna(r.NEXT_START) else True)).sum()))
    counts = []
    bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, np.inf)]
    for name, values in [("Line start to next start; last line to last report", between),
                         ("Line start through event or censoring date", to_end)]:
        s = pd.Series(values)
        row = {"Reports counted": name, "Lines": f"{len(s):,}",
               "Median (IQR)": f"{s.median():g} ({s.quantile(.25):g}–{s.quantile(.75):g})"}
        for lo, hi in bins:
            label = "10 or more" if lo == 10 else str(lo) if lo == hi else f"{lo}–{hi}"
            row[label] = f"{s.between(lo, hi).mean():.1%}"
        counts.append(row)
    timing = []
    events = lines[imaging_event(lines)]
    for name, d in [("All lines", events), ("First lines", events[events.LINE.eq(1)]),
                    ("Later lines", events[events.LINE.gt(1)])]:
        elapsed = d.EVENT_DAY - d.LINE_START
        timing.append({"Group": name, "Imaging events": f"{len(d):,}",
                       "Start to event report, days, median (IQR)":
                       f"{elapsed.median():.0f} ({elapsed.quantile(.25):.0f}–{elapsed.quantile(.75):.0f})"})
    return pd.DataFrame(counts), pd.DataFrame(timing)


def event_probability(dm):
    events = modeled_lines(dm)
    events = events[imaging_event(events)]
    reports = C.load_msk_progression(set(events.PATIENT_ID))
    top = reports[reports.PROGRESSION.eq("Y")].groupby(["PATIENT_ID", "DAY"]).PROB.max().rename("PROB").reset_index()
    probability = events.merge(top, left_on=["PATIENT_ID", "EVENT_DAY"], right_on=["PATIENT_ID", "DAY"], how="left").PROB
    return pd.DataFrame([{"Highest NLP probability of a positive report on the event date": f">= {threshold:.2f}",
                          "Imaging events": f"{(probability >= threshold).sum():,} of {len(probability):,}",
                          "Share": f"{(probability >= threshold).mean():.1%}"} for threshold in THRESHOLDS])


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


def treatment_change_after_event(dm):
    from genie_validation.regimen_compare import build_msk_line_regimens
    reg = build_msk_line_regimens()[["PATIENT_ID", "LINE", "REGIMEN_SET"]]
    reg["AGENTS"] = reg.REGIMEN_SET.map(C.canonical_agents)
    lines = C.load_breast_pfs().sort_values(["PATIENT_ID", "LINE"])
    lines["NEXT_START"] = lines.groupby("PATIENT_ID").LINE_START.shift(-1)
    lines["NEXT_LINE"] = lines.groupby("PATIENT_ID").LINE.shift(-1)
    e = dm.merge(lines[["PATIENT_ID", "LINE", "EVENT_DAY", "LINE_SOURCE", "NEXT_START", "NEXT_LINE"]], on=["PATIENT_ID", "LINE"])
    e = e[imaging_event(e)]
    e = e.merge(reg[["PATIENT_ID", "LINE", "AGENTS"]], on=["PATIENT_ID", "LINE"], how="left")
    e = e.merge(reg.rename(columns={"LINE": "NEXT_LINE", "AGENTS": "NEXT_AGENTS"})[["PATIENT_ID", "NEXT_LINE", "NEXT_AGENTS"]],
                on=["PATIENT_ID", "NEXT_LINE"], how="left")
    gap = e.NEXT_START - e.EVENT_DAY
    has = gap.notna()
    same = has & (e.NEXT_AGENTS == e.AGENTS)
    n = len(e)
    rows = [{"Imaging-progression events in modeled lines": "Events", "Share": f"{n:,}"},
            {"Imaging-progression events in modeled lines": "A later captured line exists", "Share": f"{has.mean():.0%}"}]
    for w in (30, 60, 90, 180):
        rows.append({"Imaging-progression events in modeled lines": f"Next line starts within {w} days: different drug set / same drug set",
                     "Share": f"{(has & ~same & gap.le(w)).mean():.0%} / {(same & gap.le(w)).mean():.0%}"})
    rows.append({"Imaging-progression events in modeled lines": "Median days from event report to next line start", "Share": f"{gap.median():.0f}"})
    rows.append({"Imaging-progression events in modeled lines": "No later captured line (last captured line)", "Share": f"{(~has).mean():.0%}"})
    return pd.DataFrame(rows)


def bpc_note_vs_imaging():
    _, met = load_bpc_endpoints()
    img = BPC.load_imaging()
    md = pd.read_csv(BPC.GENIE / "med_onc_note_level_dataset.csv", low_memory=False).rename(
        columns={"md_onc_visit_int": "day", "md_ca_status": "status"})[["record_id", "day", "status"]]
    PM, SI = [BPC.PD, BPC.MIXED], list(BPC.NON_PD)
    img_pm, img_p, img_si = (BPC.days_by_patient(img, s) for s in (PM, [BPC.PD], SI))
    md_pm, md_si = BPC.days_by_patient(md, PM), BPC.days_by_patient(md, SI)

    met = BPC.mixed_endpoints(met, md, ("note_event", "note_time"))
    met["hi"] = np.where(met.is_lhrh | met.next_start.isna(), np.inf, met.next_start)
    met["ti"] = met.start + met.mixed_time
    met["tm"] = met.start + met.note_time

    def on_day(days, pid, day):
        return bool(np.any(np.isclose(days.get(pid, BPC.EMPTY), day)))

    def inside(days, pid, lo, hi, cut):
        d = days.get(pid, BPC.EMPTY)
        return bool(np.any((d >= lo) & (d <= hi) & (d < cut)))

    scans = met[met.mixed_event.eq(1) & np.array([on_day(img_pm, r.record_id, r.ti) for r in met.itertuples()], bool)].copy()
    scans["pwe"] = [on_day(img_p, r.record_id, r.ti) for r in scans.itertuples()]
    notes = met[met.note_event.eq(1) & np.array([on_day(md_pm, r.record_id, r.tm) for r in met.itertuples()], bool)]

    def after_scan(window):
        pm = np.array([inside(md_pm, r.record_id, r.ti, r.ti + window, r.hi) for r in scans.itertuples()])
        si = np.array([inside(md_si, r.record_id, r.ti, r.ti + window, r.hi) for r in scans.itertuples()])
        return pm, pm | si

    def around_note(window):
        pm = np.array([inside(img_pm, r.record_id, max(r.tm - window, r.start + 1), r.tm + window, r.hi) for r in notes.itertuples()])
        si = np.array([inside(img_si, r.record_id, max(r.tm - window, r.start + 1), r.tm + window, r.hi) for r in notes.itertuples()])
        return pm, pm | si

    def median_730(frame, time, event):
        t, e = frame[time].astype(float), frame[event].astype(float)
        ok = t.notna() & e.notna()
        return C.km(*G.censor_730(t[ok], e[ok]))["median"]

    pm30, any30 = after_scan(30)
    pm90, any90 = after_scan(90)
    pm_end, any_end = after_scan(np.inf)
    npm30, nany30 = around_note(30)
    pwe = scans.pwe.to_numpy()
    first = met[met.RANK.eq(1)]
    rows = [("Breast regimens with PFS labels; progression scans (progressing, worsening, enlarging or mixed)",
             f"{len(met):,}; {len(scans):,}"),
            ("Scans with a clinician status note within 30 days; of these, note states progression or mixed",
             f"{any30.mean():.0%}; {pm30[any30].mean():.0%}"),
            ("Scans with a clinician status note within 90 days; of these, note states progression or mixed",
             f"{any90.mean():.0%}; {pm90[any90].mean():.0%}"),
            ("Scans with a clinician status note by regimen end; of these, note states progression or mixed",
             f"{any_end.mean():.0%}; {pm_end[any_end].mean():.0%}"),
            ("Concordance at 30 days: scans read as progressing / scans read as mixed",
             f"{pm30[any30 & pwe].mean():.0%} / {pm30[any30 & ~pwe].mean():.0%}"),
            ("Scans contradicted at 30 days that receive a progression note later in the regimen",
             f"{pm_end[any30 & ~pm30].mean():.0%}"),
            ("Clinician progression notes; with a status scan within 30 days; of these, scan shows progression or mixed",
             f"{len(notes):,}; {nany30.mean():.0%}; {npm30[nany30].mean():.0%}"),
            ("Clinician notes with any status that state no disease status",
             f"{md.status.eq('Not stated/Indeterminate').sum() / md.status.notna().sum():.0%}"),
            ("Imaging assessments with any status that state no disease status",
             f"{img.status.eq('Not stated/Indeterminate').sum() / img.status.notna().sum():.0%}"),
            (f"First regimens (n = {len(first):,}; 730-day cap): median PFS, days, note / imaging",
             f"{median_730(first, 'note_time', 'note_event'):.0f} / {median_730(first, 'mixed_time', 'mixed_event'):.0f}")]
    return pd.DataFrame(rows, columns=["GENIE BPC, all breast regimens with PFS labels; same-regimen windows", "Value"])

def generate():
    dm = C.load_design_matrix()
    counts, timing = report_counts(dm)
    names = {"frozen": "Original", "midpoint": "Midpoint imputation",
             **{f"no_negative_{w}": f"No N report within {w} days" for w in WINDOWS}}

    def formatted(result):
        return pd.DataFrame({"Progression definition": result.rule.map(names), "Lines": result.n,
                             "Events": result.events,
                             "Median PFS, days (95% CI)": result.apply(km_ci, axis=1),
                             "Time-dependent C, mean ± SD": result.apply(concordance, axis=1),
                             "Brier at 365 days, mean ± SD": result.apply(brier, axis=1)})
    return "\n\n".join(["Progression-definition sensitivity. MSK-CHORD.",
                          "Reports per line", show(counts), "Event report timing", show(timing),
                          "NLP probability of the event report (imaging-defined events, all modeled lines)",
                          show(event_probability(dm)),
                          "First mLoTs; unchanged GBSA test predictions; mean ± SD across five test folds.",
                          "No-N windows are 30, 60, and 90 days. A later report is not required. This is not observed confirmation.",
                          show(formatted(sensitivity(dm))),
                          "Midpoint uses the previous N report, or line start. Death and censoring dates stay the same.",
                          f"Same rules, all {len(dm):,} modeled lines (first and later lines; each line relabelled from its own start).",
                          show(formatted(sensitivity(dm, first_only=False))),
                          "Treatment change after an imaging-progression event (all modeled lines). The next captured line and its drug set come from the study line table.",
                          show(treatment_change_after_event(dm)),
                          "GENIE BPC: does a clinician note confirm a progression scan? Progression scans are the imaging events of the PFS-I + Mixed rule (first assessment stating Progressing/Worsening/Enlarging or Mixed after regimen start and before the next regimen start), deaths excluded because there is no scan to confirm. Windows start at the scan and stop at the next regimen start. Confirmed: a medical-oncologist note in the window states Progressing/Worsening/Enlarging or Mixed; contradicted: the notes in the window state only Stable or Improving. The reverse direction treats clinician progression notes the same way. Medians: PFS-I + Mixed and the same rule applied to notes, first regimens, 730-day cap.",
                          show(bpc_note_vs_imaging()),
                          ]) + "\n"


if __name__ == "__main__":
    print(generate())
