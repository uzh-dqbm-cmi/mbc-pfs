from pathlib import Path

import numpy as np
import pandas as pd

from analyses.common import agreement as G
from analyses.common import bpc_adapter as A
from analyses.common import cohort as C
from analyses import reconstruction as K
from analyses.cohort_selection.qc_late_start import categorize, class_flags

OUT = Path("results/rebuttal/eligibility/late_start")
WINDOW = 28
ELIGIBLE = ("final", "excluded_qc")
SITES = K.SITE_GROUPS
ALL = "All sites"

LATE_CHEMO = "Late-start: chemotherapy start, endocrine added"
LATE_ENDO = "Late-start: endocrine start, chemotherapy added"
LATE_NEITHER = "Late-start: neither at start, both added"
CO_NONGNRH = "Co-prescription: chemotherapy + non-GnRH endocrine at line start"
CO_GNRH_ONLY = "Co-prescription: GnRH agent only (no other endocrine agent in the episode)"
CO_GNRH_LATE = "Co-prescription: GnRH agent at start, non-GnRH endocrine added after day 28"
LATE = (LATE_CHEMO, LATE_ENDO, LATE_NEITHER)
CO = (CO_NONGNRH, CO_GNRH_ONLY, CO_GNRH_LATE)


def load():
    d = K.reconstruction()
    courses = A.build_drug_courses(d.g)
    return d, courses[courses.PATIENT_ID.isin(A.build_meta(d.g).PATIENT_ID)]


def pairs_pre_qc(lines, ref):
    eligible = lines[lines.STAGE.isin(ELIGIBLE)].reset_index(drop=True)
    matched = K.matched_pairs(eligible.assign(STAGE="final"), ref).set_index("A_ID")
    return eligible.join(matched[["B_ID", "OUR_EVENT", "BPC_EVENT", "SAME_LABEL", "EXACT"]])


def agreement(p):
    m = p[p.B_ID.notna()]
    informative = len(m) > 1 and m.OUR_EVENT.nunique() == 2 and m.BPC_EVENT.nunique() == 2
    kappa = G.kappa(m.OUR_EVENT, m.BPC_EVENT) if informative else np.nan
    return {"Episodes": f"{len(p):,}",
            "Same regimen": K.frac(len(m), len(p)) if len(p) else "-",
            "Same event | same regimen": K.frac(m.SAME_LABEL.sum(), len(m)) if len(m) else "-",
            "Same event and exact duration | same regimen": K.frac(m.EXACT.sum(), len(m)) if len(m) else "-",
            "Cohen's kappa": f"{kappa:.3f}" if np.isfinite(kappa) else "-"}


def by_site(p, label, sites=SITES, pooled=False):
    rows = [{"Episode set": label, "Site": s, **agreement(p[p.SITE.eq(s)])} for s in sites]
    if pooled:
        rows.append({"Episode set": label, "Site": ALL, **agreement(p)})
    return rows


def episode_sets(p):
    qc = p.STAGE.eq("excluded_qc")
    sets = [("Retained episodes (Extended Data Table 19b)", p[p.STAGE.eq("final")]),
            ("All formed eligible episodes before QC", p),
            ("QC-excluded episodes, all", p[qc]),
            ("QC-excluded: chemotherapy + endocrine in the same episode", p[qc & p.QC_CHEMO_ENDO]),
            ("QC-excluded: endocrine therapy in HR-negative disease (not chemo + endocrine)",
             p[qc & p.QC_HRNEG_ENDO & ~p.QC_CHEMO_ENDO])]
    return pd.DataFrame([row for label, frame in sets for row in by_site(frame, label)])


def check_table_19b(table, reference):
    retained = table[table["Episode set"].str.startswith("Retained")].set_index("Site")
    columns = ["Episodes", "Same regimen", "Same event | same regimen", "Same event and exact duration | same regimen"]
    for r in reference.itertuples():
        expected = [f"{r.RETAINED:,}", K.frac(r.PAIRS, r.RETAINED), K.frac(r.SAME_LABEL, r.PAIRS), K.frac(r.EXACT, r.PAIRS)]
        assert retained.loc[r.SITE, columns].tolist() == expected, (r.SITE, retained.loc[r.SITE, columns].tolist(), expected)


def class_presence(lines, courses):
    key = ["PATIENT_ID", "LINE"]
    m = lines[key + ["LINE_START", "EVENT_DAY"]].merge(
        courses[["PATIENT_ID", "START_DATE", "STOP_DATE", "SUBTYPE", "AGENT", "REGIMEN_NUMBER"]], on="PATIENT_ID")
    m["STOP"] = m.STOP_DATE.fillna(m.START_DATE + 1)
    m = m[m.STOP.ge(m.LINE_START)].copy()
    m["IN28"] = m.START_DATE.le(m.LINE_START + WINDOW)
    m["INR"] = m.START_DATE.le(m.EVENT_DAY - WINDOW)
    m["CHEMO"] = m.SUBTYPE.eq("CHEMO")
    m["GNRH"] = m.SUBTYPE.eq("HORMONE") & m.AGENT.isin(C.GNRH_AGENTS)
    m["NONGNRH"] = m.SUBTYPE.eq("HORMONE") & ~m.AGENT.isin(C.GNRH_AGENTS)
    return class_flags(lines, m), m[m.INR]


def classify_chemo_endo(lines, courses):
    flags, received = class_presence(lines, courses)
    lines = lines.merge(flags, on=["PATIENT_ID", "LINE"], validate="one_to_one")
    rebuilt = lines.CHEMO_R & lines.HORMONE_R & lines.PFS_EVENT.ne(-1)
    assert (rebuilt == lines.QC_CHEMO_ENDO).all(), "received-window reconstruction disagrees with QC_CHEMO_ENDO"
    cat = categorize(lines, LATE + CO)
    lines["CATEGORY"] = cat
    lines["LATE_START"] = cat.isin(LATE)
    key = ["PATIENT_ID", "LINE"]
    idx = pd.MultiIndex.from_frame(lines[key])
    first = {}
    for cls, sel in [("CHEMO", received.CHEMO), ("HORMONE", received.SUBTYPE.eq("HORMONE")), ("NONGNRH", received.NONGNRH)]:
        grp = received.loc[sel].groupby(key)
        first[cls] = (grp.START_DATE.min() - grp.LINE_START.first()).reindex(idx).to_numpy()
    add = pd.Series(np.nan, index=lines.index)
    c = cat.to_numpy()
    add[c == LATE_CHEMO] = first["HORMONE"][c == LATE_CHEMO]
    add[c == LATE_ENDO] = first["CHEMO"][c == LATE_ENDO]
    add[c == LATE_NEITHER] = np.fmin(first["CHEMO"], first["HORMONE"])[c == LATE_NEITHER]
    add[c == CO_GNRH_LATE] = first["NONGNRH"][c == CO_GNRH_LATE]
    lines["ADD_DAY"] = add
    assert lines.loc[lines.LATE_START, "ADD_DAY"].gt(WINDOW).all()
    return lines


def add_day(d):
    a = d.ADD_DAY.dropna()
    if a.empty:
        return "-"
    q = a.quantile([.25, .5, .75]).to_numpy()
    return f"{q[1]:.0f} ({q[0]:.0f}-{q[2]:.0f})"


def chemo_endo_table(p):
    f = p[p.STAGE.eq("excluded_qc") & p.QC_CHEMO_ENDO]
    groups = [(c, f[f.CATEGORY.eq(c)]) for c in CO + LATE if f.CATEGORY.eq(c).any()]
    groups += [("Co-prescription at line start, all", f[~f.LATE_START]), ("Late-start, all", f[f.LATE_START]),
               ("All chemotherapy + endocrine exclusions", f)]
    timed = set(LATE) | {CO_GNRH_LATE, "Late-start, all"}
    rows = []
    for label, d in groups:
        for site in list(SITES) + [ALL]:
            s = d if site == ALL else d[d.SITE.eq(site)]
            rows.append({"Category": label, "Site": site, **agreement(s),
                         "Second-class start, median (IQR) day": add_day(s) if label in timed else "-"})
    return pd.DataFrame(rows)


def day28_rematch(lines, courses, courses_assigned, ref):
    key = ["PATIENT_ID", "LINE"]
    eligible = lines[lines.STAGE.isin(ELIGIBLE)].copy()
    late = eligible.loc[eligible.STAGE.eq("excluded_qc") & eligible.LATE_START, key + ["SITE", "CATEGORY", "LINE_START", "EVENT_DAY"]]
    _, received = class_presence(late, courses)
    received = received.merge(late[key + ["CATEGORY"]], on=key)
    later = ~received.IN28 & np.where(received.CATEGORY.eq(LATE_CHEMO), received.SUBTYPE.eq("HORMONE"), received.CHEMO)
    present = received[received.IN28].groupby(key).REGIMEN_NUMBER.agg(set)
    added = received[later].groupby(key).REGIMEN_NUMBER.agg(set)
    late_idx = pd.MultiIndex.from_frame(late[key])
    separate = pd.Series([isinstance(p, set) and isinstance(a, set) and p.isdisjoint(a)
                          for p, a in zip(present.reindex(late_idx), added.reindex(late_idx))], index=late.index)
    own = courses_assigned.merge(late[key + ["LINE_START"]], on=key)
    day28 = own[own.START_DATE.le(own.LINE_START + WINDOW)].groupby(key).AGENT.agg(lambda s: "+".join(sorted(set(s))))
    swap = pd.MultiIndex.from_frame(eligible[key]).isin(late_idx)
    eligible.loc[swap, "AGENTS"] = day28.reindex(pd.MultiIndex.from_frame(eligible.loc[swap, key])).to_numpy()
    before = pairs_pre_qc(lines, ref).merge(late[key], on=key)
    after = pairs_pre_qc(eligible, ref).merge(late[key], on=key)
    rows = []
    for site in list(SITES) + [ALL]:
        pick = (lambda d: d) if site == ALL else (lambda d: d[d.SITE.eq(site)])
        n = len(pick(late))
        rows.append({"Site": site, "Late-start episodes": f"{n:,}",
                     "Later class in a separate BPC regimen": K.frac(separate[pick(late).index].sum(), n) if n else "-",
                     "Same regimen, full drug set": K.frac(pick(before).B_ID.notna().sum(), n) if n else "-",
                     "Same regimen, drugs started by day 28": K.frac(pick(after).B_ID.notna().sum(), n) if n else "-"})
    return pd.DataFrame(rows)


def unmatched_reasons(lines, courses_assigned, regimens, ref):
    eligible = lines[lines.STAGE.isin(ELIGIBLE)].assign(STAGE="final")
    r, causes = K.unmatched_causes(eligible, courses_assigned, regimens, ref)
    r = r.assign(STAGE=lines.loc[lines.STAGE.isin(ELIGIBLE), "STAGE"].to_numpy())
    qc = r[r.STAGE.eq("excluded_qc") & r.B_ID.isna()]
    reason = causes.reindex(qc.index)
    order = ["treatment changed without counted progression", "BPC records drugs started together as separate regimens",
             "drug carried over from an earlier episode", "GnRH agonist started first", "BPC regimen has no PFS label",
             "investigational drug in the BPC regimen", "no named drug", "other grouping difference"]
    rows = []
    for label in [o for o in order if reason.eq(o).any()] + ["Unmatched QC-excluded episodes"]:
        sel = qc if label == "Unmatched QC-excluded episodes" else qc[reason.eq(label).to_numpy()]
        rows.append({"Reason no BPC regimen has the same drugs and start": label,
                     **{s: f"{sel.SITE.eq(s).sum():,}" for s in SITES}, ALL: f"{len(sel):,}"})
    return pd.DataFrame(rows)


def main():
    d, courses = load()
    lines = classify_chemo_endo(d.lines, courses)
    p = pairs_pre_qc(lines, d.ref)
    same_pool = p.loc[p.STAGE.eq("final") & p.B_ID.notna()].set_index(["PATIENT_ID", "LINE"]).B_ID.sort_index().equals(
        d.pairs.set_index(["PATIENT_ID", "LINE"]).B_ID.sort_index())
    sets = episode_sets(p)
    check_table_19b(sets, K.conditional_agreement(d.lines, d.pairs))
    chemo_endo = chemo_endo_table(p)
    rematch = day28_rematch(lines, courses, d.courses, d.ref)
    reasons = unmatched_reasons(lines, d.courses, d.regimens, d.ref)
    qc = lines[lines.STAGE.eq("excluded_qc")]
    both = int((qc.QC_CHEMO_ENDO & qc.QC_HRNEG_ENDO).sum())
    assigned_only, _ = class_presence(lines, d.courses)
    missed = int((lines.QC_CHEMO_ENDO & ~(assigned_only.CHEMO_R & assigned_only.HORMONE_R).to_numpy()).sum())
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([sets.assign(Table="agreement_by_episode_set"),
               chemo_endo.assign(Table="chemo_endocrine_exclusions_by_day28_category"),
               rematch.assign(Table="late_start_day28_rematch"),
               reasons.assign(Table="unmatched_qc_excluded_reasons")],
              ignore_index=True).to_csv(OUT / "bpc_agreement_pre_qc.csv", index=False)
    text = "\n\n".join([
        "Reconstruction agreement with GENIE BPC (PFS-I + Mixed) before QC. ",
        "Same regimen: same patient, same named drug set without GnRH agents, starts within 14 days (greedy one-to-one). "
        "Event indicator and PFS duration are compared after the 730-day cap on both sides. Cohen's kappa is shown only when "
        "both sources have events and censorings among the pairs. Eligible episodes before QC: "
        "STAGE final or excluded_qc (labelled, PFS >= 28 days, imaging in the 90 days before the start). "
        "Matching pool: all eligible episodes before QC; "
        + ("every retained episode keeps the match it has in Table 19b." if same_pool else
           "WARNING: some retained episodes change match when QC-excluded episodes join the pool."),
        "Agreement by episode set", K.show(sets, left=2),
        f"QC-excluded episodes failing both rules ({both}) are counted under chemotherapy + endocrine, as in the reconstruction cohort flow.",
        "Chemotherapy + endocrine QC exclusions by what was known at day 28 (classification of qc_late_start.py applied to BPC). "
        "Courses: every BPC drug course of the patient's index breast cancer regimens, the table that set the QC flag; "
        f"the received-window rebuild reproduces QC_CHEMO_ENDO for all {int(lines.PFS_EVENT.ne(-1).sum()):,} labelled episodes "
        f"(rec.courses_assigned alone, which omits courses started before metastatic onset, misses {missed} flags). "
        "Chemo = class CHEMO; endocrine = class HORMONE (GnRH: leuprolide, goserelin, triptorelin, histrelin). "
        "Day-28 presence: a course overlapping [line start, line start + 28]. Received window: start <= event day - 28 and stop >= line start. "
        "Late-start: the two classes are not both present at day 28. Second-class start: first start of the absent class, days from line start "
        "(for the GnRH-at-start group, the non-GnRH endocrine agent). Categories with no BPC episode are omitted.",
        K.show(chemo_endo, left=2),
        "Late-start episodes: BPC regimen of the later class, and same-regimen matching on the drugs started by day 28. "
        "Separate BPC regimen: no received-window course of the class added after day 28 shares a BPC regimen number with the courses "
        "present at day 28. Drugs started by day 28: the episode's own courses started on or before line start + 28; every other eligible "
        "episode keeps its drug set, and the one-to-one rule and 14-day start tolerance are unchanged.",
        K.show(rematch),
        "Reason the unmatched QC-excluded episodes have no same-regimen BPC match (reconstruction attribution)",
        K.show(reasons),
        "Recalculate: python -m analyses.cohort_selection.bpc_agreement_pre_qc",
        f"Outputs: {OUT / 'bpc_agreement_pre_qc.csv'}, {OUT / 'bpc_agreement_pre_qc.txt'}"]) + "\n"
    (OUT / "bpc_agreement_pre_qc.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
