from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from tabulate import tabulate

from config import CALIBRATION_HORIZONS as HORIZONS
from analyses.common import cohort as C

OUT = Path("results/rebuttal/eligibility/late_start")
FIG = OUT / "figures"
WINDOW = 28
SUBTYPES = ["HR+/HER2-", "HR+/HER2+", "HR-/HER2+", "TNBC"]
LATE = {"Chemotherapy start, endocrine added", "Endocrine start, chemotherapy added", "Neither at start, both added"}
CO_NONGNRH = "Co-prescription: chemotherapy + non-GnRH endocrine at line start"
CO_GNRH_ONLY = "Co-prescription: GnRH agent only (no other endocrine agent in the line)"
CO_GNRH_LATE = "Co-prescription: GnRH agent at start, non-GnRH endocrine added after day 28"
ORDER = ["Chemotherapy start, endocrine added", "Endocrine start, chemotherapy added", "Neither at start, both added",
         CO_NONGNRH, CO_GNRH_ONLY, CO_GNRH_LATE]


def fmt(frame):
    return tabulate(frame, headers="keys", showindex=False, tablefmt="psql", floatfmt=".3f", disable_numparse=True)


def km_fit(d):
    return KaplanMeierFitter().fit(d.PFS_TIME_DAYS, d.PFS_EVENT)


def km_median(d):
    if len(d) == 0:
        return "-"
    m = km_fit(d).median_survival_time_
    return f"{m:.0f}" if np.isfinite(m) else "NR"


def pct(x):
    return f"{100 * x:.1f}%"


def iqr(s):
    q = s.quantile([.25, .5, .75]).to_numpy()
    return f"{q[1]:.0f} ({q[0]:.0f}-{q[2]:.0f})"


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
        first[cls + "_AGENTS"] = r.loc[sel].sort_values(["START", "AGENT"]).groupby(key).apply(
            lambda x: ", ".join(dict.fromkeys(x.loc[x.START.eq(x.START.min()), "AGENT"].str.title())))
    idx = pd.MultiIndex.from_frame(lines[key])
    add = pd.Series(np.nan, index=lines.index)
    agents = pd.Series("", index=lines.index, dtype=object)
    chemo_first, horm_first = first["CHEMO"].reindex(idx).to_numpy(), first["HORMONE"].reindex(idx).to_numpy()
    nong_first = first["NONGNRH"].reindex(idx).to_numpy()
    c = cat.to_numpy()
    sel = c == "Chemotherapy start, endocrine added"
    add[sel], agents[sel] = horm_first[sel], first["HORMONE_AGENTS"].reindex(idx).to_numpy()[sel]
    sel = c == "Endocrine start, chemotherapy added"
    add[sel], agents[sel] = chemo_first[sel], first["CHEMO_AGENTS"].reindex(idx).to_numpy()[sel]
    sel = c == "Neither at start, both added"
    add[sel] = np.fmin(chemo_first, horm_first)[sel]
    agents[sel] = np.where(chemo_first[sel] <= horm_first[sel], first["CHEMO_AGENTS"].reindex(idx).to_numpy()[sel],
                           first["HORMONE_AGENTS"].reindex(idx).to_numpy()[sel])
    sel = c == CO_GNRH_LATE
    add[sel], agents[sel] = nong_first[sel], first["NONGNRH_AGENTS"].reindex(idx).to_numpy()[sel]
    lines["ADD_DAY"], lines["ADDED_AGENTS"] = add, agents
    assert lines.loc[lines.LATE_START, "ADD_DAY"].gt(WINDOW).all()
    assert lines.loc[lines.LATE_START, "ADD_DAY"].le(lines.loc[lines.LATE_START, "EVENT_DAY"] - lines.loc[lines.LATE_START, "LINE_START"] - WINDOW).all()
    s = m[m.IN28].sort_values(["START", "AGENT"])
    start_agents = s.groupby(key).AGENT.apply(lambda x: ", ".join(dict.fromkeys(x.str.title()))).reindex(idx)
    lines["START_CLASS_AGENTS"] = start_agents.to_numpy()
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


def hrneg_note(lines):
    late = lines[lines.LATE_START]
    hr = lines[lines.QC_HRNEG]
    return (f"{int(late.QC_HRNEG.sum())} late-start lines also fail the HR-negative endocrine rule and stay excluded, "
            f"so {int((~late.QC_HRNEG).sum())} are restorable. Of the {len(hr)} HR-negative endocrine exclusions, "
            f"{int((~hr.HORMONE28).sum())} have no endocrine agent in the first 28 days (the agent began later).")


def size_table(lines):
    late = lines[lines.LATE_START & ~lines.QC_HRNEG]
    rows = []
    for subtype in SUBTYPES + ["All"]:
        for grp in ["First line", "Later line", "All lines"]:
            sel = (lines.SUBTYPE.eq(subtype) if subtype != "All" else pd.Series(True, index=lines.index)) & \
                  (lines.LINE_GROUP.eq(grp) if grp != "All lines" else pd.Series(True, index=lines.index))
            e, l = lines[sel], late[sel.loc[late.index]]
            chemo_start = e[e.START_CLASS.eq("Chemotherapy")]
            endo_start = e[e.START_CLASS.eq("Endocrine")]
            lc, le = l[l.START_CLASS.eq("Chemotherapy")], l[l.START_CLASS.eq("Endocrine")]
            rows.append({"Subtype": subtype, "Line group": grp, "Eligible lines": f"{len(e):,}",
                         "Retained": f"{int(e.RETAINED.sum()):,}", "Late-start (restorable)": f"{len(l):,}",
                         "Share of eligible": pct(len(l) / len(e)) if len(e) else "-",
                         "Chemo-start lines (eligible)": f"{len(chemo_start):,}",
                         "Late-start among chemo-start": f"{len(lc):,} ({pct(len(lc) / len(chemo_start)) if len(chemo_start) else '-'})",
                         "Endocrine-start lines (eligible)": f"{len(endo_start):,}",
                         "Late-start among endocrine-start": f"{len(le):,} ({pct(len(le) / len(endo_start)) if len(endo_start) else '-'})"})
    return pd.DataFrame(rows)


def timing_table(lines):
    late = lines[lines.LATE_START]
    rows = []
    for label, sel in [("All late-start", late.index), *[(c, late.index[late.CATEGORY.eq(c)]) for c in ORDER[:3]],
                       *[(f"{c} / {s}", late.index[late.CATEGORY.eq(c) & late.SUBTYPE.eq(s)]) for c in ORDER[:2] for s in SUBTYPES]]:
        d = late.loc[sel]
        if len(d) == 0:
            continue
        a = d.ADD_DAY
        rows.append({"Group": label, "Lines": f"{len(d):,}", "Median (IQR) day of second-class start": iqr(a),
                     "10th-90th pct": f"{a.quantile(.1):.0f}-{a.quantile(.9):.0f}",
                     "Within 90 d": pct(a.le(90).mean()), "91-180 d": pct(a.between(91, 180).mean()),
                     "181-365 d": pct(a.between(181, 365).mean()), "After 365 d": pct(a.gt(365).mean()),
                     "Median PFS, d": km_median(d), "Residual PFS after addition, median d": km_median(
                         d.assign(PFS_TIME_DAYS=d.PFS_TIME_DAYS - d.ADD_DAY))})
    added = most_common(late.loc[late.CATEGORY.eq(ORDER[0]), "ADDED_AGENTS"].str.split(", ").explode(), 8)
    added2 = most_common(late.loc[late.CATEGORY.eq(ORDER[1]), "ADDED_AGENTS"].str.split(", ").explode(), 8)
    notes = [f"Endocrine agents added to chemotherapy-start lines (first added record): {', '.join(f'{k} {v}' for k, v in added.items())}.",
             f"Chemotherapy agents added to endocrine-start lines: {', '.join(f'{k} {v}' for k, v in added2.items())}."]
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, c in zip(axes, ORDER[:2]):
        d = late[late.CATEGORY.eq(c)]
        ax.hist(d.ADD_DAY.clip(upper=730), bins=np.arange(0, 760, 30), color="#4C72B0", edgecolor="white")
        ax.set_title(f"{c} (n={len(d)})", fontsize=10)
        ax.set_xlabel("Days from line start to second-class start"); ax.set_ylabel("Lines")
        ax.axvline(d.ADD_DAY.median(), color="k", ls="--", lw=1)
    fig.tight_layout(); fig.savefig(FIG / "second_class_start_timing.png", dpi=200); plt.close(fig)
    return pd.DataFrame(rows), notes


def km_tables_and_figures(lines):
    late = lines[lines.LATE_START & ~lines.QC_HRNEG]
    retained = lines[lines.RETAINED]
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for j, subtype in enumerate(SUBTYPES):
        for i, grp in enumerate(["First line", "Later line"]):
            r = retained[retained.SUBTYPE.eq(subtype) & retained.LINE_GROUP.eq(grp)]
            l = late[late.SUBTYPE.eq(subtype) & late.LINE_GROUP.eq(grp)]
            both = pd.concat([r, l])
            k_r, k_b = km_fit(r), km_fit(both)
            row = {"Subtype": subtype, "Line group": grp, "Retained n": f"{len(r):,}", "Restored n": f"{len(l):,}",
                   "Retained median": km_median(r), "Retained + restored median": km_median(both),
                   "Restored-only median": km_median(l) if len(l) else "-"}
            for h in HORIZONS:
                row[f"S({h}) retained"] = f"{k_r.predict(h):.3f}"
                row[f"S({h}) retained + restored"] = f"{k_b.predict(h):.3f}"
            rows.append(row)
            ax = axes[i, j]
            k_r.plot_survival_function(ax=ax, ci_show=False, color="#1f77b4", label=f"Retained (n={len(r)})")
            k_b.plot_survival_function(ax=ax, ci_show=False, color="#d62728", ls="--", label=f"Retained + restored (n={len(both)})")
            if len(l) >= 10:
                km_fit(l).plot_survival_function(ax=ax, ci_show=False, color="#ff7f0e", ls=":", label=f"Restored only (n={len(l)})")
            ax.set_title(f"{subtype}, {grp.lower()}", fontsize=10); ax.set_xlim(0, 730); ax.set_xlabel("Days from line start")
            ax.legend(fontsize=7); ax.set_ylabel("PFS probability" if j == 0 else "")
    fig.tight_layout(); fig.savefig(FIG / "km_retained_vs_restored_by_subtype_line.png", dpi=200); plt.close(fig)
    return pd.DataFrame(rows)


def cox_hr(d, group_col, adjust=True):
    frame = d[["PFS_TIME_DAYS", "PFS_EVENT", "PATIENT_ID", group_col]].copy()
    if adjust:
        for s in SUBTYPES[1:]:
            frame[f"S_{s}"] = d.SUBTYPE.eq(s).astype(int)
        frame["FIRST"] = d.FIRST_LINE.astype(int)
        frame = frame.loc[:, frame.nunique().gt(1) | frame.columns.isin(["PFS_TIME_DAYS", "PFS_EVENT", "PATIENT_ID", group_col])]
    cph = CoxPHFitter().fit(frame, "PFS_TIME_DAYS", "PFS_EVENT", cluster_col="PATIENT_ID")
    s = cph.summary.loc[group_col]
    return f"{s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f})"


def landmark_table(lines):
    late = lines[lines.LATE_START & ~lines.QC_HRNEG]
    retained = lines[lines.RETAINED]
    median_add = int(np.round(late.ADD_DAY.median() / 30) * 30)
    rows = []
    for L in sorted({90, median_add, 180, 365}):
        for label, r_sel, l_sel in [("All lines", retained.index, late.index),
                                    ("HR+ chemotherapy-start lines", retained.index[retained.HR.eq(1) & retained.START_CLASS.eq("Chemotherapy")],
                                     late.index[late.HR.eq(1) & late.START_CLASS.eq("Chemotherapy")]),
                                    ("HR+ endocrine-start lines", retained.index[retained.HR.eq(1) & retained.START_CLASS.eq("Endocrine")],
                                     late.index[late.HR.eq(1) & late.START_CLASS.eq("Endocrine")])]:
            r = retained.loc[r_sel]; r = r[r.PFS_TIME_DAYS.gt(L)]
            l = late.loc[l_sel]; l = l[l.PFS_TIME_DAYS.gt(L) & l.ADD_DAY.le(L)]
            if len(l) < 10:
                rows.append({"Landmark day": L, "Comparison": label, "Retained at risk": f"{len(r):,}", "Late-start at risk (class added by landmark)": f"{len(l):,}",
                             "Residual median, retained": "-", "Residual median, late-start": "-", "Residual S(365) retained": "-",
                             "Residual S(365) late-start": "-", "HR late-start vs retained (adjusted)": "n < 10"})
                continue
            both = pd.concat([r.assign(G=0), l.assign(G=1)])
            both["PFS_TIME_DAYS"] = both.PFS_TIME_DAYS - L
            kr, kl = km_fit(both[both.G.eq(0)]), km_fit(both[both.G.eq(1)])
            rows.append({"Landmark day": L, "Comparison": label, "Retained at risk": f"{len(r):,}",
                         "Late-start at risk (class added by landmark)": f"{len(l):,}",
                         "Residual median, retained": km_median(both[both.G.eq(0)]), "Residual median, late-start": km_median(both[both.G.eq(1)]),
                         "Residual S(365) retained": f"{kr.predict(365):.3f}", "Residual S(365) late-start": f"{kl.predict(365):.3f}",
                         "HR late-start vs retained (adjusted)": cox_hr(both, "G", adjust=(label == "All lines"))})
    return pd.DataFrame(rows), median_add


def coprescription_table(lines):
    co = lines[lines.QC_CHEMO_ENDO & ~lines.LATE_START & ~lines.QC_HRNEG]
    late = lines[lines.LATE_START & ~lines.QC_HRNEG]
    retained = lines[lines.RETAINED]
    rows = []
    for subtype in ["All"] + SUBTYPES:
        for grp in ["All lines", "First line", "Later line"]:
            def pick(d):
                s = d.SUBTYPE.eq(subtype) if subtype != "All" else pd.Series(True, index=d.index)
                g = d.LINE_GROUP.eq(grp) if grp != "All lines" else pd.Series(True, index=d.index)
                return d[s & g]
            r, c, l = pick(retained), pick(co), pick(late)
            row = {"Subtype": subtype, "Line group": grp, "Retained n": f"{len(r):,}", "Retained median": km_median(r),
                   "Co-prescription n": f"{len(c):,}", "Co-prescription median": km_median(c),
                   "Late-start n": f"{len(l):,}", "Late-start median": km_median(l)}
            for name, d in [("co-prescription", c), ("late-start", l)]:
                both = pd.concat([r.assign(G=0), d.assign(G=1)])
                row[f"HR {name} vs retained"] = cox_hr(both, "G", adjust=(subtype == "All")) if len(d) >= 20 else "n < 20"
            rows.append(row)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, grp in zip(axes, ["All lines", "First line"]):
        for d, label, color, ls in [(retained, "Retained", "#1f77b4", "-"), (co, "Co-prescription exclusions", "#2ca02c", "-."),
                                    (late, "Late-start exclusions", "#d62728", "--")]:
            dd = d if grp == "All lines" else d[d.FIRST_LINE]
            km_fit(dd).plot_survival_function(ax=ax, ci_show=False, color=color, ls=ls, label=f"{label} (n={len(dd):,})")
        ax.set_title(grp); ax.set_xlim(0, 730); ax.set_xlabel("Days from line start"); ax.legend(fontsize=8)
    axes[0].set_ylabel("PFS probability")
    fig.tight_layout(); fig.savefig(FIG / "km_retained_vs_coprescription_vs_late.png", dpi=200); plt.close(fig)
    return pd.DataFrame(rows)


def other_late_start_table(lines):
    other = lines[lines.LATE_START & ~lines.CATEGORY.eq(ORDER[0])]
    rows = []
    for c in ORDER[1:3]:
        for subtype in SUBTYPES:
            d = other[other.CATEGORY.eq(c) & other.SUBTYPE.eq(subtype)]
            if len(d) == 0:
                continue
            top = most_common(d.START_CLASS_AGENTS, 3)
            rows.append({"Category": c, "Subtype": subtype, "Lines": f"{len(d):,}", "First lines": f"{int(d.FIRST_LINE.sum()):,}",
                         "HR-neg rule also fails": f"{int(d.QC_HRNEG.sum()):,}", "Events": pct(d.PFS_EVENT.mean()),
                         "Median PFS, d": km_median(d), "Second class start, median (IQR) d": iqr(d.ADD_DAY),
                         "Most common line-start agents": "; ".join(f"{k} ({v})" for k, v in top.items())})
    rows.append({"Category": "All other late-start", "Subtype": "All", "Lines": f"{len(other):,}", "First lines": f"{int(other.FIRST_LINE.sum()):,}",
                 "HR-neg rule also fails": f"{int(other.QC_HRNEG.sum()):,}", "Events": pct(other.PFS_EVENT.mean()),
                 "Median PFS, d": km_median(other), "Second class start, median (IQR) d": iqr(other.ADD_DAY), "Most common line-start agents": ""})
    return pd.DataFrame(rows)


def first_line_subtype_table(lines):
    rows = []
    for subtype in SUBTYPES + ["All"]:
        s = lines.SUBTYPE.eq(subtype) if subtype != "All" else pd.Series(True, index=lines.index)
        r = lines[s & lines.RETAINED & lines.FIRST_LINE]
        l = lines[s & lines.LATE_START & ~lines.QC_HRNEG & lines.FIRST_LINE]
        rc = r[r.START_CLASS.eq("Chemotherapy")]; lc = l[l.START_CLASS.eq("Chemotherapy")]
        rows.append({"Subtype": subtype, "Retained first lines": f"{len(r):,}", "Median PFS (retained)": km_median(r),
                     "Late-start first lines": f"{len(l):,}", "Median PFS (retained + late-start)": km_median(pd.concat([r, l])),
                     "Chemo-start first lines, retained": f"{len(rc):,}", "Median (retained)": km_median(rc),
                     "Chemo-start first lines, late-start": f"{len(lc):,}", "Median (retained + late-start)": km_median(pd.concat([rc, lc]))})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    lines = classify_lines()
    lines.to_csv(OUT / "line_classification.csv", index=False)
    cat = category_table(lines); size = size_table(lines); timing, timing_notes = timing_table(lines)
    km = km_tables_and_figures(lines); landmark, median_add = landmark_table(lines)
    co = coprescription_table(lines); other = other_late_start_table(lines); first = first_line_subtype_table(lines)
    for name, frame in [("category", cat), ("size", size), ("timing", timing), ("km_retained_vs_restored", km),
                        ("landmark", landmark), ("coprescription_vs_retained", co), ("other_late_start", other),
                        ("first_line_medians", first)]:
        frame.to_csv(OUT / f"{name}.csv", index=False)
    cross = pd.crosstab(lines.loc[lines.QC_CHEMO_ENDO, "CATEGORY"], lines.loc[lines.QC_CHEMO_ENDO, "SUBTYPE"], margins=True)
    text = "\n\n".join([
        "Chemotherapy + endocrine QC exclusions classified by what was known at day 28.",
        "Chemo = MSK-CHORD SUBTYPE Chemo; endocrine = SUBTYPE Hormone (GnRH agents: leuprolide, goserelin, triptorelin, histrelin, "
        "degarelix, relugolix). Day-28 presence: a record overlapping [line start, line start + 28], the model's treatment-input window "
        "(PLANNED_TREATMENT_*). Received-window presence: a record overlapping [line start, event day - 28], the window that sets the QC flag. "
        "Late-start: a flagged line in which the two classes are not both present at day 28. Second-class start: first start of the absent "
        "class, in days from line start. PFS: pre-QC labels with the 730-day cap. Medians are Kaplan-Meier.",
        "Self-checks: the received-window reconstruction reproduces the QC flag for all 11,085 eligible lines, and the day-28 reconstruction "
        "reproduces PLANNED_TREATMENT_CHEMO and PLANNED_TREATMENT_HORMONE for all 2,294 QC-excluded lines.",
        "Flagged lines by category", fmt(cat), hrneg_note(lines),
        "Flagged lines by category and subtype", tabulate(cross, headers="keys", tablefmt="psql"),
        "Size of the late-start group (restorable lines, HR-negative rule retained)", fmt(size),
        "Timing of the second-class start", fmt(timing), "\n".join(timing_notes),
        f"Figure: {FIG / 'second_class_start_timing.png'}",
        "Kaplan-Meier PFS, retained versus retained plus late-start, by subtype and line group", fmt(km),
        f"Figure: {FIG / 'km_retained_vs_restored_by_subtype_line.png'}",
        f"Landmark comparison (median second-class start rounded to 30 days: {median_add}). Lines progression-free at the landmark; "
        "late-start lines must have added the second class by the landmark. Hazard ratios: Cox with patient-clustered standard errors; "
        "'All lines' adjusts for subtype and first-line status, strata rows are unadjusted. Residual S(365): KM survival 365 days after the landmark.",
        fmt(landmark),
        "Co-prescription exclusions (both classes in the first 28 days; HR-negative rule not failed) and late-start exclusions versus retained lines. "
        "Hazard ratios as above ('All' rows adjusted for subtype and first-line status).", fmt(co),
        f"Figure: {FIG / 'km_retained_vs_coprescription_vs_late.png'}",
        "Late-start lines other than the chemotherapy-start group", fmt(other),
        "First-line medians by subtype, retained versus retained plus late-start (Table 1 context)", fmt(first),
        "Recalculate: python -m analyses.cohort_selection.qc_late_start",
        "Inputs: saved MSK line records and labels; design matrices; MSK treatment timeline."]) + "\n"
    (OUT / "late_start.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
