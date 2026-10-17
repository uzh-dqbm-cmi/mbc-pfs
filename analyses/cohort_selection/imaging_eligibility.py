from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from tabulate import tabulate
from SurvivalEVAL.Evaluator import SurvivalEvaluator

from qc import agent_to_equiv_group
from analyses.common import cohort as C
from analyses.common.survival import SurvivalContext, ci, load_curves, patient_bootstrap, patient_draws
from analyses.cohort_selection.qc_late_start import line_ends, most_common
from analyses.cohort_selection.qc_refit import restated_rule_note

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
    imaging_cadence().to_csv(OUT / "imaging_cadence.csv", index=False)
    chemo_endocrine_anatomy()[1].to_csv(OUT / "qc_chemo_endocrine_anatomy.csv", index=False)


def imaging_eligibility(context=None):
    cohorts = pd.read_csv(OUT / "eligibility_sensitivity.csv")
    intervals = pd.read_csv(OUT / "eligibility_C_intervals.csv")
    samples = (context or SurvivalContext()).evaluate("full_gbsa")["samples"]
    intervals.loc[intervals.Cohort.eq(COHORTS[0]), ["CI_low", "CI_high"]] = ci(samples)
    return cohorts.merge(intervals.drop(columns="C"), on="Cohort"), intervals[intervals.Cohort.eq(CHANGE)]


def cohort_tables():
    eligible = pd.read_csv("data/BREAST_pfs.csv")
    excluded = pd.read_csv("data/BREAST_dropped_pfs_lines.csv")
    qc = pd.read_csv("data/dropped_lines.csv")
    qc = qc.merge(eligible[["PATIENT_ID", "LINE"]], on=["PATIENT_ID", "LINE"], validate="one_to_one")
    dm = pd.read_csv("data/design_matrix.csv")
    clinical = pd.read_csv("data/msk_chord_2024/data_clinical_patient.tsv", sep="\t",
                           skiprows=4, usecols=["PATIENT_ID", "HR"])
    clinical["HR"] = clinical.HR.str.upper().eq("YES").astype(int)
    ages = pd.read_csv("data/msk_chord_2024/brca_dx_1st_seq_OS.csv", usecols=["PATIENT_ID", "AGE"])
    clinical = clinical.merge(ages, on="PATIENT_ID")
    groups = []
    for label, reason in [("No imaging in prior 90 days", "no_radiology_within_90_prior"),
                          ("No follow-up PFS label", "no_radiology_follow_up"),
                          ("Duration below 28 days", "line_too_short_for_effectiveness")]:
        d = excluded.loc[excluded.LINE_SOURCE.eq(reason)].drop(columns=["AGE", "HR"], errors="ignore")
        d = d.merge(clinical, on="PATIENT_ID")
        d["AGE"] += np.floor_divide(d.LINE_START, 365)
        groups.append((label, d))
    chemo = qc.RECEIVED_TREATMENT_CHEMO.gt(0) & qc.RECEIVED_TREATMENT_HORMONE.gt(0)
    hrnegative = qc.HR.eq(0) & qc.RECEIVED_TREATMENT_HORMONE.gt(0)
    groups.extend([("QC: chemotherapy + endocrine only", qc.loc[chemo & ~hrnegative]),
                   ("QC: HR-negative + endocrine only", qc.loc[hrnegative & ~chemo]),
                   ("QC: both reasons", qc.loc[chemo & hrnegative]), ("Retained", dm)])
    rows = []
    remaining = len(eligible) + len(excluded)
    for label, d in groups:
        if label != "Retained":
            remaining -= len(d)
        age = d.AGE.quantile([.25, .5, .75]).to_numpy()
        rows.append({"Group": label, "Lines": len(d), "Patients": d.PATIENT_ID.nunique(),
                     "Age, median (IQR)": f"{age[1]:.0f} ({age[0]:.0f}-{age[2]:.0f})",
                     "HR positive": f"{100*d.HR.mean():.1f}%", "Lines remaining": remaining})
    original = eligible.copy()
    original["PFS_TIME_DAYS"] = original.EVENT_DAY - original.LINE_START
    original["PFS_EVENT"] = original.LINE_SOURCE.isin(["labelled_by_progression_event", "labelled_by_death_event"]).astype(int)
    removed = eligible.merge(qc[["PATIENT_ID", "LINE"]], on=["PATIENT_ID", "LINE"], validate="one_to_one")
    retained = eligible.merge(dm[["PATIENT_ID", "LINE"]], on=["PATIENT_ID", "LINE"], validate="one_to_one")
    flow = []
    for label, d in [("Before 730-day cap", original), ("After cap, before QC", eligible),
                     ("Removed by QC", removed), ("Retained after QC", retained)]:
        flow.append({"Step": label, "Total lines": len(d),
                     "Progression events": int((d.PFS_EVENT.eq(1) & d.LINE_SOURCE.eq("labelled_by_progression_event")).sum()),
                     "Death events": int((d.PFS_EVENT.eq(1) & d.LINE_SOURCE.eq("labelled_by_death_event")).sum()),
                     "Censored": int(d.PFS_EVENT.eq(0).sum())})
    capped = original.PFS_TIME_DAYS.gt(730)
    counts = [(label, int((capped & original.LINE_SOURCE.eq(reason)).sum())) for label, reason in
              [("Progression changed to censoring", "labelled_by_progression_event"),
               ("Death changed to censoring", "labelled_by_death_event"),
               ("Already-censored duration shortened", "labelled_by_non_progression")]]
    medians = []
    for label, d in [("QC excluded", removed), ("Retained", retained)]:
        d = d.loc[d.LINE.eq(1)]
        km = KaplanMeierFitter().fit(d.PFS_TIME_DAYS, d.PFS_EVENT)
        medians.append({"First-line group": label, "Lines": len(d), "Median PFS, days": km.median_survival_time_})
    return rows, flow, counts, medians, len(eligible) + len(excluded)


def pct(x):
    return f"{x:.1%}"


def iqr(s):
    q = s.quantile([.25, .5, .75]).to_numpy()
    return f"{q[1]:.0f} ({q[0]:.0f}{' to ' if q[0] < 0 else '-'}{q[2]:.0f})"


def km_median(d):
    median = KaplanMeierFitter().fit(d.PFS_TIME_DAYS, d.PFS_EVENT).median_survival_time_
    return f"{median:.0f}" if np.isfinite(median) else "not reached"


def imaging_cadence():
    eligible, excluded, ends = C.load_breast_pfs(), C.load_dropped_pfs_lines(), line_ends()
    gated = excluded.loc[excluded.LINE_SOURCE.eq("no_radiology_within_90_prior")]
    t, e = gated.ORIGINAL_PFS_TIME_DAYS, gated.ORIGINAL_PFS_EVENT
    gated = gated.assign(PFS_TIME_DAYS=t.clip(upper=730), PFS_EVENT=e.mask(t.gt(730) & e.ge(0), 0))
    qc = pd.read_csv("data/dropped_lines.csv", usecols=["PATIENT_ID", "LINE"]).merge(
        eligible, on=["PATIENT_ID", "LINE"], validate="one_to_one")
    reports = C.load_msk_progression()
    reports = reports.loc[reports.PROGRESSION.isin(["Y", "N"]), ["PATIENT_ID", "DAY"]]
    table = {}
    for label, d in [("No imaging in prior 90 days", gated), ("Retained", C.load_design_matrix()), ("QC excluded", qc)]:
        lines = d.merge(ends, on=["PATIENT_ID", "LINE_START"], validate="one_to_one").reset_index(names="ID")
        m = lines.merge(reports, on="PATIENT_ID", how="left")
        after = m.DAY.gt(m.LINE_START)
        per = pd.DataFrame({"ID": m.ID, "next": (m.DAY - m.LINE_START).where(after),
                            "prior": (m.LINE_START - m.DAY).where(m.DAY.le(m.LINE_START)),
                            "count": after & (m.DAY.le(m.NEXT_START) | m.NEXT_START.isna())}
                           ).groupby("ID").agg({"next": "min", "prior": "min", "count": "sum"})
        nxt, labelled = per["next"], lines.loc[lines.PFS_EVENT.isin([0, 1])]
        table[label] = {"Lines": f"{len(lines):,}", "With a report after line start": f"{nxt.notna().sum():,}",
                        "No report after line start": pct(nxt.isna().mean()),
                        "Days to next report, median (IQR)": iqr(nxt.dropna()),
                        **{f"Next report within {w} days": pct(nxt.le(w).mean()) for w in (90, 180, 365)},
                        "No report at or before line start": pct(per.prior.isna().mean()),
                        "Days since prior report, median (IQR)": iqr(per.prior.dropna()),
                        "Reports per line, median (IQR)": iqr(per["count"]),
                        "Lines with a PFS label": f"{len(labelled):,}", "KM median PFS, days": km_median(labelled)}
    return pd.DataFrame(table).rename_axis("Measure").reset_index()


def overlap(a, b):
    if a.empty or b.empty:
        return -np.inf
    return (np.minimum.outer(a.E.to_numpy(), b.E.to_numpy()) - np.maximum.outer(a.S.to_numpy(), b.S.to_numpy())).max()


ANATOMY_RULES = [
    "Records: MSK-CHORD treatment rows with START <= event day - 28 and STOP >= line start, the window that set RECEIVED_TREATMENT_* for the QC flag.",
    "Chemo = CHEMO_ group or MSK subtype Chemo; endocrine = ENDO_ group or subtype Hormone; GnRH = leuprolide, goserelin, triptorelin, histrelin, degarelix, relugolix.",
    "Records are clipped to [line start, next line start] (final line: no end clip). Overlap = days shared by one chemo and one endocrine record. Categories are applied in order:",
    "1 GnRH-only overlap: no non-GnRH endocrine record overlaps chemo, and a GnRH record overlaps chemo or GnRH is the only endocrine agent.",
    "2 Boundary carry-over: largest chemo/non-GnRH endocrine overlap <= 28 days (2a: 1-28 days; 2b: no overlap).",
    "3 Sequential within line: overlap > 28 days; the first clipped starts of the two classes differ by > 28 days.",
    "4 Together from start: overlap > 28 days; first clipped starts within 28 days.",
    "Endocrine means GnRH in category 1 and non-GnRH endocrine otherwise. Overlap: largest per line; negative values are gaps.",
    "Chemo first, Endo first and Same day compare first clipped starts. Carried in: the class's first record began before line start.",
    "PFS: pre-QC labels (BREAST_pfs.csv, 730-day cap)."]


def chemo_endocrine_anatomy():
    reasons = pd.read_csv("data/dropped_lines.csv", usecols=["PATIENT_ID", "LINE", "QC_REASONS"])
    tally = reasons.QC_REASONS.str.strip("()").str.replace("'", "").str.rstrip(",").value_counts()
    tally = tally.rename_axis("QC_REASONS").reset_index(name="Lines")
    flagged = reasons.QC_REASONS.str.contains("chemo+hormone in same LoT", regex=False)
    hrneg = flagged & reasons.QC_REASONS.str.contains("HRneg_endocrine")
    notes = [f"chemo+hormone in same LoT: {flagged.sum():,} lines ({(flagged & ~hrneg).sum():,} without and {hrneg.sum():,} with "
             "HRneg_endocrine). qc_from_received_flags sets endo+chemo_overlap_any_line by the same rule; it does not test overlap."]
    lines = C.load_breast_pfs().merge(line_ends(), on=["PATIENT_ID", "LINE_START"], validate="one_to_one").merge(
        reasons.loc[flagged, ["PATIENT_ID", "LINE"]].assign(FLAG=True), on=["PATIENT_ID", "LINE"], how="left", validate="one_to_one")
    lines["FLAG"] = lines.FLAG.eq(True)
    tx = pd.read_csv(C.MSK_DIR / "data_timeline_treatment.tsv", sep="\t",
                     usecols=["PATIENT_ID", "START_DATE", "STOP_DATE", "SUBTYPE", "AGENT"])
    tx["STOP_DATE"] = tx.STOP_DATE.fillna(tx.START_DATE)
    group = tx.AGENT.astype(str).str.upper().str.strip().map(agent_to_equiv_group)
    tx = tx.assign(CHEMO=group.str.startswith("CHEMO_") | tx.SUBTYPE.eq("Chemo"),
                   ENDO=group.str.startswith("ENDO_") | tx.SUBTYPE.eq("Hormone"),
                   GNRH=group.eq("ENDO_GNRH") | tx.AGENT.isin(C.GNRH_AGENTS), NAMED=group.str.startswith(("CHEMO_", "ENDO_")))
    m = lines.merge(tx.loc[tx.CHEMO | tx.ENDO], on="PATIENT_ID")
    m = m.loc[m.START_DATE.le(m.EVENT_DAY - 28) & m.STOP_DATE.ge(m.LINE_START)]
    both = m.groupby(["PATIENT_ID", "LINE"]).agg(c=("CHEMO", "any"), e=("ENDO", "any"))
    rebuilt = lines.set_index(["PATIENT_ID", "LINE"]).index.isin(both.index[both.c & both.e])
    notes.append(f"The reconstructed flag agrees with the QC flag for {(rebuilt == lines.FLAG).sum():,} of {len(lines):,} eligible lines.")
    m = m.loc[m.FLAG].assign(S=lambda x: x.START_DATE.clip(lower=x.LINE_START), E=lambda x: np.fmin(x.STOP_DATE, x.NEXT_START))
    rows = []
    for (pid, line), d in m.groupby(["PATIENT_ID", "LINE"]):
        c, g, h = d.loc[d.CHEMO], d.loc[d.ENDO & d.GNRH], d.loc[d.ENDO & ~d.GNRH]
        gnrh, other = overlap(c, g), overlap(c, h)
        if h.empty or (gnrh > 0 and other <= 0):
            category, e = "1 GnRH-only overlap", g
        elif other <= 28:
            category, e = ("2a Overlap 1-28 days" if other > 0 else "2b No overlap"), h
        else:
            category, e = ("3 Sequential within line" if abs(h.S.min() - c.S.min()) > 28 else "4 Together from start"), h
        rows.append(dict(PATIENT_ID=pid, LINE=line, Category=category, FIRST=np.sign(e.S.min() - c.S.min()), OVERLAP=overlap(c, e),
                         ENDO_IN=e.START_DATE.min() < d.LINE_START.iat[0], CHEMO_IN=c.START_DATE.min() < d.LINE_START.iat[0],
                         CHEMO_NAMED=c.NAMED.any(), ENDO_NAMED=d.loc[d.ENDO, "NAMED"].any()))
    anatomy = pd.DataFrame(rows).merge(lines[["PATIENT_ID", "LINE", "PFS_TIME_DAYS", "PFS_EVENT"]], on=["PATIENT_ID", "LINE"])
    summary = []
    for label, d in [*anatomy.groupby("Category"), ("All flagged lines", anatomy)]:
        summary.append({"Category": label, "Lines": len(d), "Share": pct(len(d) / len(anatomy)), "First lines": int(d.LINE.eq(1).sum()),
                        "Event rate": pct(d.PFS_EVENT.mean()), "Median PFS, days": km_median(d), "Overlap, days": iqr(d.OVERLAP),
                        "Chemo first": int(d.FIRST.gt(0).sum()), "Endo first": int(d.FIRST.lt(0).sum()), "Same day": int(d.FIRST.eq(0).sum()),
                        "Endo carried in": int(d.ENDO_IN.sum()), "Chemo carried in": int(d.CHEMO_IN.sum())})
    summary = pd.DataFrame(summary)
    for named, column, label in [("CHEMO_NAMED", "CHEMO", "chemotherapy"), ("ENDO_NAMED", "ENDO", "endocrine therapy")]:
        records = m.merge(anatomy.loc[~anatomy[named], ["PATIENT_ID", "LINE"]]).loc[lambda x: x[column]]
        top = ", ".join(most_common(records.AGENT.str.lower(), 4).index)
        notes.append(f"{(~anatomy[named]).sum():,} flagged lines have {label} only from agents outside the {column}_ groups ({top}).")
    return tally, summary, notes


def saved_report(path, title):
    if not path.exists():
        return [f"{title}: not computed (see the module docstring for the command)."]
    return [f"==== {title} ====", path.read_text().strip()]


def generate(context=None):
    rows, flow, counts, medians, total = cohort_tables()
    fmt = lambda frame: tabulate(frame, headers="keys", showindex=False, tablefmt="psql", floatfmt=".4f")
    performance, delta = imaging_eligibility(context)
    cadence = imaging_cadence()
    tally, anatomy, notes = chemo_endocrine_anatomy()
    return "\n\n".join([
        f"Segmented lines: {total:,}. Eligible before QC: {sum(r['Lines'] for r in rows[3:]):,}.",
        "Exclusion groups are sequential. Each line is counted once. Patient counts can overlap. Age uses years. "
        "No follow-up PFS label means no qualifying imaging outcome or death supplied a label. "
        "There are no additional unlabelled lines after the first three filters.", fmt(rows),
        "Separate administrative censoring and QC", fmt(flow),
        "Lines changed by the 730-day cap", fmt(pd.DataFrame(counts, columns=["Change", "Lines"])),
        "First-line Kaplan-Meier medians", fmt(medians),
        "Fixed-model imaging-eligibility sensitivity", fmt(performance),
        "Paired change: expanded minus original", fmt(delta),
        "Imaging cadence by line group",
        "Reports have PROGRESSION Y or N. Reports per line: after line start up to the next line start (final line: all later reports). "
        "PFS: 90-day group ORIGINAL_PFS_* (labelled lines, 730-day cap); QC group pre-QC labels.", fmt(cadence),
        "QC_REASONS of QC-excluded lines", fmt(tally), notes[0],
        "Chemotherapy + endocrine QC lines by treatment overlap", "\n".join(ANATOMY_RULES), fmt(anatomy), "\n".join(notes[1:]),
        (OUT / "qc_sensitivity.txt").read_text().strip(),
        *saved_report(OUT / "late_start" / "late_start.txt", "Late-start chemo/endocrine exclusions (qc_late_start)"),
        *saved_report(Path("results/rebuttal/qc_refit/eval/qc_refit_eval.txt"), "Refits with late-start lines restored (qc_refit, qc_refit_eval)"),
        restated_rule_note(),
        *saved_report(OUT / "late_start" / "bpc_agreement_pre_qc.txt", "Reconstruction agreement with GENIE BPC before QC (bpc_agreement_pre_qc)"),
        *saved_report(Path("results/rebuttal/qc_refit/eval/calibration_gap_bootstrap.txt"),
                      "Exploratory paired calibration-gap analysis (qc_calibration_bootstrap)"),
        "Recalculate the exploratory block: python -m analyses.cohort_selection.qc_calibration_bootstrap",
        "Recalculate: python -m analyses.cohort_selection.imaging_eligibility",
        "Inputs: saved MSK line records; frozen added features and curves in data/rebuttal/eligibility; saved GBSA curves; "
        "MSK treatment and radiology report timelines."
    ]) + "\n"


if __name__ == "__main__":
    main()
