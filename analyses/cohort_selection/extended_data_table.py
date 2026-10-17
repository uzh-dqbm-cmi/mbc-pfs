import re
from pathlib import Path

import pandas as pd

from analyses.reconstruction import days
from analyses.regimen_pfs import BOTH, REMOVED, RETAINED, qc_selection_km, qc_selection_lines

LATE = Path("results/rebuttal/eligibility/late_start")
EVAL = Path("results/rebuttal/qc_refit/eval")
ORIGINAL = "Original (exclusion kept)"
PRIMARY = "Late-start restored, full PFS (primary)"
CATEGORIES = {"Chemotherapy start, endocrine added": "Chemotherapy start; endocrine therapy added later",
              "Endocrine start, chemotherapy added": "Endocrine start; chemotherapy added later",
              "Neither at start, both added": "Neither class in the first 28 days; both added later",
              "Late-start, all": "Late-start lines, total",
              "Co-prescription at line start, all": "Both classes recorded within the first 28 days",
              "Retained (modeled) lines": "Original retained cohort"}
REFITS = {ORIGINAL: "Original analysis",
          PRIMARY: "505 late-start lines restored, full PFS",
          "Late-start restored, censored at class addition (secondary)": "505 late-start lines censored at addition",
          "Restated rule (late-start + GnRH groups restored)": "Late-start and HR-positive GnRH groups restored",
          "Chemo+endocrine rule removed": "Chemotherapy plus endocrine rule removed",
          "All QC exclusions restored": "All QC exclusions restored"}
STRATA = {"HR+/HER2+ first lines starting on chemotherapy": ("HR+/HER2+ first lines", "starting on chemotherapy"),
          "HR+/HER2- first lines starting on chemotherapy": ("HR+/HER2- first lines", "starting on chemotherapy"),
          "HR+/HER2+ first lines, all": ("All HR+/HER2+", "first lines"),
          "HR+ endocrine-start lines (incl. restored endocrine-start lines)": ("HR+ endocrine-start lines", "including restored"),
          "Unaffected: TNBC": ("Triple-negative", ""),
          "Unaffected: HR-/HER2+": ("HR-/HER2+", ""),
          "All lines in cohort": ("All lines", ""),
          "Restored lines, all": ("Restored late-start lines", "only")}
REGIMENS = [("HR+/HER2-", None, "HR+/HER2-: all first lines"),
            ("HR+/HER2-", "HR+/HER2-: letrozole + palbociclib", "– Letrozole + palbociclib"),
            ("HR+/HER2-", "HR+/HER2-: letrozole", "– Letrozole"),
            ("HR+/HER2+", None, "HR+/HER2+: all first lines"),
            ("HR+/HER2+", "HR+/HER2+: paclitaxel + pertuzumab + trastuzumab", "– Paclitaxel + pertuzumab + trastuzumab")]
SITES = ("DFCI", "VICC", "MSK", "All sites")


def read(path):
    return pd.read_csv(path, dtype=str).fillna("")


def percent(cell):
    return re.sub(r"\d\.\d{3}", lambda m: f"{100 * float(m.group()):.1f}", cell)


def panel_a():
    cat = read(LATE / "category.csv").set_index("Group")
    return pd.DataFrame([[label, *(cat.at[g, c] for c in ["Lines", "Patients", "First lines"]), cat.at[g, "Events"].rstrip("%"),
                          cat.at[g, "Median PFS, d"]] for g, label in CATEGORIES.items()],
                        columns=["Group", "Lines", "Patients", "First lines", "Events (%)", "Median PFS (days)"])


def panel_b():
    disc = read(EVAL / "discrimination.csv").set_index("Model")
    original = disc.at[ORIGINAL, "Full-cohort lines"]
    rows = [[label, disc.at[saved, "Full-cohort lines"], disc.at[saved, "C on 8,791 lines"],
             disc.at[saved, "Delta C vs original (paired 95% CI)"], disc.at[saved, "Delta IBS vs original (paired 95% CI)"]]
            for saved, label in REFITS.items()]
    return pd.DataFrame(rows, columns=["Refit cohort", "Lines", f"C on original {original} lines", "Paired ΔC (95% CI)",
                                       "Paired ΔIBS (95% CI)"])


def panel_c():
    cal = read(EVAL / "calibration_by_stratum.csv")
    cal = cal[cal.Cohort.eq(PRIMARY)].set_index(["Stratum", "Model"])
    rows = []
    for saved, (first, second) in STRATA.items():
        o, r = cal.loc[(saved, ORIGINAL)], cal.loc[(saved, PRIMARY)]
        for lead, model, x in [([first, o["Lines"], percent(o["365 d: KM S [95% CI]"])], "Original", o),
                               ([second, "", ""], "Refit", r)]:
            rows.append([*lead, model, percent(x["365 d: mean predicted S"]), x["365 d: predicted - KM (points)"],
                         x["365 d: slope"], x["365 d: ICI (points)"]])
    return pd.DataFrame(rows, columns=["Stratum", "Lines", "Observed one-year PFS, % (95% CI)", "Model", "Mean predicted, %",
                                       "Predicted - observed, points", "Slope", "ICI, points"])


def panel_d():
    ext = read(EVAL / "external.csv")
    rows = []
    for site in ["DFCI", "VICC"]:
        o, r = (ext[ext.Model.eq(m) & ext.Site.eq(site)].iloc[0] for m in [ORIGINAL, PRIMARY])
        rows.append([site, o["Regimens"], o["Patients"], *(f"{o[c]} / {r[c]}" for c in ["C", "IBS", "180 d: ICI (points)"])])
    return pd.DataFrame(rows, columns=["Site", "Regimens", "Patients", "C", "IBS", "180-day ICI, points"])


def panel_e():
    bpc = read(LATE / "bpc_agreement_pre_qc.csv")
    sets = bpc[bpc.Table.eq("agreement_by_episode_set")]
    retained = sets[sets["Episode set"].str.startswith("Retained")].set_index("Site")
    pre = sets[sets["Episode set"].eq("All formed eligible episodes before QC")].set_index("Site")
    rows = [[site, *(d.at[site, c] for c in ["Same regimen", "Same event | same regimen"] for d in (retained, pre))]
            for site in SITES[:3]]
    return pd.DataFrame(rows, columns=["Site", *(f"{c}: {d}" for c in ["Same regimen", "Same event, given same regimen"]
                                                 for d in ["Retained episodes", "QC filter removed"])])


def panel_f():
    bpc = read(LATE / "bpc_agreement_pre_qc.csv")
    rematch = bpc[bpc.Table.eq("late_start_day28_rematch")].set_index("Site")
    columns = ["Late-start episodes", "Later class in a separate BPC regimen", "Same regimen, full drug set",
               "Same regimen, drugs started by day 28"]
    return pd.DataFrame([[site, *(rematch.at[site, c] for c in columns)] for site in SITES],
                        columns=["Site", "Episodes", "Later class in a separate BPC regimen", "Same regimen, all drugs",
                                 "Same regimen, drugs started by day 28"])


def removed_are_restored():
    classes = pd.read_csv(LATE / "line_classification.csv", usecols=["PATIENT_ID", "LINE", "LATE_START", "QC_HRNEG"])
    restored = classes[classes.LATE_START & ~classes.QC_HRNEG.astype(bool)]
    lines = qc_selection_lines()
    for _, regimen, _ in REGIMENS:
        if regimen is not None:
            d = lines[regimen][REMOVED]
            assert len(d.merge(restored, on=["PATIENT_ID", "LINE"])) == len(d), regimen


def panel_g():
    removed_are_restored()
    km = read(LATE / "km_retained_vs_restored.csv")
    km = km[km["Line group"].eq("First line")].set_index("Subtype")
    exact = qc_selection_km()
    rows = []
    for subtype, regimen, label in REGIMENS:
        if regimen is None:
            k = km.loc[subtype]
            total = int(k["Retained n"].replace(",", "")) + int(k["Restored n"].replace(",", ""))
            cells = [k["Retained n"], k["Retained median"], f"{total:,}", k["Retained + restored median"]]
        else:
            r, b = exact[regimen, RETAINED], exact[regimen, BOTH]
            cells = [f"{r['n']:,}", days(r["median"]), f"{b['n']:,}", days(b["median"])]
        rows.append([label, *cells])
    return pd.DataFrame(rows, columns=["Subtype and exact initial drug set", "Retained cohort: lines",
                                       "Retained cohort: median (days)", "Late-start lines restored: lines",
                                       "Late-start lines restored: median (days)"])
