from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from analyses.common import cohort as C

OUT = Path("results/rebuttal/eligibility/late_start")
WINDOW = 28
LATE = {"Chemotherapy start, endocrine added", "Endocrine start, chemotherapy added", "Neither at start, both added"}
CO_NONGNRH = "Co-prescription: chemotherapy + non-GnRH endocrine at line start"
CO_GNRH_ONLY = "Co-prescription: GnRH agent only (no other endocrine agent in the line)"
CO_GNRH_LATE = "Co-prescription: GnRH agent at start, non-GnRH endocrine added after day 28"
ORDER = ["Chemotherapy start, endocrine added", "Endocrine start, chemotherapy added", "Neither at start, both added",
         CO_NONGNRH, CO_GNRH_ONLY, CO_GNRH_LATE]


def km_fit(d):
    return KaplanMeierFitter().fit(d.PFS_TIME_DAYS, d.PFS_EVENT)


def km_median(d):
    if len(d) == 0:
        return "-"
    m = km_fit(d).median_survival_time_
    return f"{m:.0f}" if np.isfinite(m) else "NR"


def pct(x):
    return f"{100 * x:.1f}%"


def most_common(s, n):
    return s.value_counts().sort_index().sort_values(ascending=False, kind="stable").head(n)


def line_ends():
    starts = pd.concat([C.load_breast_pfs(), C.load_dropped_pfs_lines()])[["PATIENT_ID", "LINE_START"]]
    starts = starts.drop_duplicates().sort_values(["PATIENT_ID", "LINE_START"])
    return starts.assign(NEXT_START=starts.groupby("PATIENT_ID").LINE_START.shift(-1))


def class_flags(lines, m):
    key = ["PATIENT_ID", "LINE"]
    flags = pd.DataFrame(index=pd.MultiIndex.from_frame(lines[key]))
    for suffix, mask in [("28", m.IN28), ("_R", m.INR)]:
        for cls in ["CHEMO", "GNRH", "NONGNRH"]:
            flags[f"{cls}{suffix}"] = m.loc[mask & m[cls]].groupby(key).size().reindex(flags.index).fillna(0).gt(0)
        flags[f"HORMONE{suffix}"] = flags[f"GNRH{suffix}"] | flags[f"NONGNRH{suffix}"]
    return flags.reset_index()


def categorize(lines, names):
    late_chemo, late_endo, late_neither, co_nongnrh, co_gnrh_only, co_gnrh_late = names
    f = lines.QC_CHEMO_ENDO
    cat = pd.Series("not flagged", index=lines.index, dtype=object)
    cat[f & lines.CHEMO28 & ~lines.HORMONE28] = late_chemo
    cat[f & ~lines.CHEMO28 & lines.HORMONE28] = late_endo
    cat[f & ~lines.CHEMO28 & ~lines.HORMONE28] = late_neither
    co = f & lines.CHEMO28 & lines.HORMONE28
    cat[co & lines.NONGNRH28] = co_nongnrh
    cat[co & ~lines.NONGNRH28 & ~lines.NONGNRH_R] = co_gnrh_only
    cat[co & ~lines.NONGNRH28 & lines.NONGNRH_R] = co_gnrh_late
    return cat


def eligible_lines():
    labels = pd.read_csv("data/BREAST_pfs.csv")
    dm = pd.read_csv("data/design_matrix.csv", usecols=["PATIENT_ID", "LINE", "HR", "HER2"]).assign(RETAINED=True)
    dropped = pd.read_csv("data/dropped_lines.csv", usecols=["PATIENT_ID", "LINE", "HR", "HER2", "QC_REASONS",
                          "PLANNED_TREATMENT_CHEMO", "PLANNED_TREATMENT_HORMONE",
                          "RECEIVED_TREATMENT_CHEMO", "RECEIVED_TREATMENT_HORMONE"]).assign(RETAINED=False)
    lines = labels.merge(pd.concat([dm, dropped], ignore_index=True), on=["PATIENT_ID", "LINE"], validate="one_to_one")
    assert len(lines) == len(labels) == 11085
    reasons = lines.QC_REASONS.fillna("")
    lines["QC_CHEMO_ENDO"] = reasons.str.contains("chemo+hormone in same LoT", regex=False)
    lines["QC_HRNEG"] = reasons.str.contains("HRneg_endocrine", regex=False)
    lines["SUBTYPE"] = C.subtype_label(lines.HR, lines.HER2)
    lines["FIRST_LINE"] = lines.LINE.eq(1)
    return lines.merge(line_ends().rename(columns={"NEXT_START": "NEXT_LINE_START"}), on=["PATIENT_ID", "LINE_START"],
                       validate="one_to_one")


def treatment_records(patient_ids):
    tx = pd.read_csv(C.MSK_DIR / "data_timeline_treatment.tsv", sep="\t", dtype=str)
    tx = tx[tx.PATIENT_ID.isin(set(patient_ids)) & tx.SUBTYPE.isin(["Chemo", "Hormone"])].copy()
    tx["START"] = tx.START_DATE.astype(int)
    tx["STOP"] = pd.to_numeric(tx.STOP_DATE, errors="coerce").fillna(tx.START + 1).astype(int)
    tx["AGENT"] = tx.AGENT.astype(str).str.upper().str.strip()
    tx["CHEMO"] = tx.SUBTYPE.eq("Chemo")
    tx["GNRH"] = tx.SUBTYPE.eq("Hormone") & tx.AGENT.isin(C.GNRH_AGENTS)
    tx["NONGNRH"] = tx.SUBTYPE.eq("Hormone") & ~tx.AGENT.isin(C.GNRH_AGENTS)
    return tx[["PATIENT_ID", "START", "STOP", "SUBTYPE", "AGENT", "CHEMO", "GNRH", "NONGNRH"]]


def classify_lines():
    lines = eligible_lines()
    tx = treatment_records(lines.PATIENT_ID)
    m = lines[["PATIENT_ID", "LINE", "LINE_START", "EVENT_DAY"]].merge(tx, on="PATIENT_ID")
    m = m[m.STOP.ge(m.LINE_START)]
    m["IN28"] = m.START.le(m.LINE_START + WINDOW)
    m["INR"] = m.START.le(m.EVENT_DAY - WINDOW)
    key = ["PATIENT_ID", "LINE"]
    lines = lines.merge(class_flags(lines, m), on=key, validate="one_to_one")
    rebuilt = lines.CHEMO_R & lines.HORMONE_R
    assert (rebuilt == lines.QC_CHEMO_ENDO).all(), "received-window reconstruction disagrees with the QC flag"
    d = lines[~lines.RETAINED]
    assert (d.CHEMO28 == d.PLANNED_TREATMENT_CHEMO.gt(0)).all() and (d.HORMONE28 == d.PLANNED_TREATMENT_HORMONE.gt(0)).all(), \
        "day-28 reconstruction disagrees with PLANNED_TREATMENT flags"
    cat = categorize(lines, ORDER)
    lines["CATEGORY"] = cat
    lines["LATE_START"] = cat.isin(LATE)
    r = m[m.INR]
    first = {}
    for cls in ["CHEMO", "HORMONE", "NONGNRH"]:
        sel = r.CHEMO if cls == "CHEMO" else (r.SUBTYPE.eq("Hormone") if cls == "HORMONE" else r.NONGNRH)
        g = r.loc[sel].groupby(key)
        first[cls] = (g.START.min() - g.LINE_START.first())
    idx = pd.MultiIndex.from_frame(lines[key])
    add = pd.Series(np.nan, index=lines.index)
    chemo_first, horm_first = first["CHEMO"].reindex(idx).to_numpy(), first["HORMONE"].reindex(idx).to_numpy()
    nong_first = first["NONGNRH"].reindex(idx).to_numpy()
    c = cat.to_numpy()
    sel = c == "Chemotherapy start, endocrine added"
    add[sel] = horm_first[sel]
    sel = c == "Endocrine start, chemotherapy added"
    add[sel] = chemo_first[sel]
    sel = c == "Neither at start, both added"
    add[sel] = np.fmin(chemo_first, horm_first)[sel]
    sel = c == CO_GNRH_LATE
    add[sel] = nong_first[sel]
    lines["ADD_DAY"] = add
    assert lines.loc[lines.LATE_START, "ADD_DAY"].gt(WINDOW).all()
    assert lines.loc[lines.LATE_START, "ADD_DAY"].le(lines.loc[lines.LATE_START, "EVENT_DAY"] - lines.loc[lines.LATE_START, "LINE_START"] - WINDOW).all()
    lines["LINE_GROUP"] = np.where(lines.FIRST_LINE, "First line", "Later line")
    lines["START_CLASS"] = np.select([lines.CHEMO28 & ~lines.HORMONE28, lines.HORMONE28 & ~lines.CHEMO28,
                                      lines.CHEMO28 & lines.HORMONE28], ["Chemotherapy", "Endocrine", "Both"], "Neither")
    return lines


def group_row(label, d):
    return {"Group": label, "Lines": f"{len(d):,}", "Patients": f"{d.PATIENT_ID.nunique():,}",
            "First lines": f"{int(d.FIRST_LINE.sum()):,}", "Events": pct(d.PFS_EVENT.mean()) if len(d) else "-",
            "Median PFS, d": km_median(d)}


def category_table(lines):
    f = lines[lines.QC_CHEMO_ENDO]
    rows = [group_row(c, f[f.CATEGORY.eq(c)]) for c in ORDER]
    rows.append(group_row("Late-start, all", f[f.LATE_START]))
    rows.append(group_row("Co-prescription at line start, all", f[~f.LATE_START]))
    rows.append(group_row("All flagged lines", f))
    rows.append(group_row("Retained (modeled) lines", lines[lines.RETAINED]))
    return pd.DataFrame(rows)


def km_table(lines):
    late = lines[lines.LATE_START & ~lines.QC_HRNEG]
    retained = lines[lines.RETAINED]
    rows = []
    for subtype in ("HR+/HER2-", "HR+/HER2+"):
        r = retained[retained.SUBTYPE.eq(subtype) & retained.FIRST_LINE]
        l = late[late.SUBTYPE.eq(subtype) & late.FIRST_LINE]
        both = pd.concat([r, l])
        rows.append({"Subtype": subtype, "Line group": "First line", "Retained n": f"{len(r):,}",
                     "Restored n": f"{len(l):,}", "Retained median": km_median(r),
                     "Retained + restored median": km_median(both)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = classify_lines()
    lines.to_csv(OUT / "line_classification.csv", index=False)
    category_table(lines).to_csv(OUT / "category.csv", index=False)
    km_table(lines).to_csv(OUT / "km_retained_vs_restored.csv", index=False)
    print(f"Wrote the classification and table results to {OUT}.", flush=True)


if __name__ == "__main__":
    main()
