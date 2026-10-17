from pathlib import Path

import numpy as np
import pandas as pd

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
    return eligible.join(matched[["B_ID", "SAME_LABEL"]])


def agreement(p):
    m = p[p.B_ID.notna()]
    return {"Episodes": f"{len(p):,}",
            "Same regimen": K.frac(len(m), len(p)) if len(p) else "-",
            "Same event | same regimen": K.frac(m.SAME_LABEL.sum(), len(m)) if len(m) else "-"}


def by_site(p, label):
    return [{"Episode set": label, "Site": s, **agreement(p[p.SITE.eq(s)])} for s in SITES]


def episode_sets(p):
    sets = [("Retained episodes (Extended Data Table 19b)", p[p.STAGE.eq("final")]),
            ("All formed eligible episodes before QC", p)]
    return pd.DataFrame([row for label, frame in sets for row in by_site(frame, label)])


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
    flags, _ = class_presence(lines, courses)
    lines = lines.merge(flags, on=["PATIENT_ID", "LINE"], validate="one_to_one")
    rebuilt = lines.CHEMO_R & lines.HORMONE_R & lines.PFS_EVENT.ne(-1)
    assert (rebuilt == lines.QC_CHEMO_ENDO).all(), "received-window reconstruction disagrees with QC_CHEMO_ENDO"
    cat = categorize(lines, LATE + CO)
    lines["CATEGORY"] = cat
    lines["LATE_START"] = cat.isin(LATE)
    return lines


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


def main():
    d, courses = load()
    lines = classify_chemo_endo(d.lines, courses)
    sets = episode_sets(pairs_pre_qc(lines, d.ref))
    rematch = day28_rematch(lines, courses, d.courses, d.ref)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "bpc_agreement_pre_qc.csv"
    pd.concat([sets.assign(Table="agreement_by_episode_set"),
               rematch.assign(Table="late_start_day28_rematch")],
              ignore_index=True).to_csv(output, index=False)
    print(f"Wrote {output}.", flush=True)


if __name__ == "__main__":
    main()
