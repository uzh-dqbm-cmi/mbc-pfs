from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test

from config import MONTH_DAYS
from manuscript_utils import treatment_combinations
from genie_validation.regimen_compare import build_msk_line_regimens
from analyses.common import agreement as G, cohort as C
from analyses.reconstruction import km_cell, km_ci, load_bpc_endpoints, show

TRIPLET = frozenset({"PACLITAXEL", "PERTUZUMAB", "TRASTUZUMAB"})
MAINTENANCE = C.ENDOCRINE | C.GNRH_AGENTS | C.CDK46
ORDER = [("HR+/HER2-", "LETROZOLE + PALBOCICLIB"), ("HR+/HER2-", "LETROZOLE"),
         ("HR-/HER2+", "PACLITAXEL + PERTUZUMAB + TRASTUZUMAB"), ("TNBC", "CAPECITABINE"),
         ("TNBC", "PACLITAXEL"), ("HR+/HER2+", "PACLITAXEL + PERTUZUMAB + TRASTUZUMAB"),
         ("HR+/HER2+", "LETROZOLE")]
DEMICHELE = "DeMichele 2021, weighted rwPFS"
PERUSE = "PERUSE 2019, trial PFS"
LITERATURE = {ORDER[0]: dict(source=DEMICHELE, arm="Palbociclib plus letrozole", n=772, median=20.0, lo=17.5, hi=21.9),
              ORDER[1]: dict(source=DEMICHELE, arm="Letrozole alone", n=658, median=11.9, lo=10.5, hi=13.7),
              ORDER[2]: dict(source=PERUSE, median=24.2),
              ORDER[5]: dict(source=PERUSE, median=22.7)}
QC_SETS = [("HR+/HER2+", ["CHEMO_PACLITAXEL", "BIOLOGIC_PERTUZUMAB", "BIOLOGIC_TRASTUZUMAB"]),
           ("HR-/HER2+", ["CHEMO_PACLITAXEL", "BIOLOGIC_PERTUZUMAB", "BIOLOGIC_TRASTUZUMAB"]),
           ("HR+/HER2-", ["HORMONE_LETROZOLE", "TARGETED_PALBOCICLIB"]),
           ("HR+/HER2-", ["HORMONE_LETROZOLE"])]
RETAINED, REMOVED, BOTH = "Retained (modeled)", "Removed by the chemo + endocrine rule", "Both together"


@lru_cache(maxsize=1)
def first_lines():
    first = C.load_design_matrix().query("LINE == 1").copy()
    regimens = build_msk_line_regimens()[["PATIENT_ID", "LINE", "REGIMEN_SET"]]
    first = first.merge(regimens, on=["PATIENT_ID", "LINE"], validate="one_to_one")
    first["AGENTS"] = first.REGIMEN_SET.map(C.canonical_agents)
    return first


def regimen_groups():
    _, met = load_bpc_endpoints()
    bpc, msk = met[met.RANK.eq(1)], first_lines()
    return {(subtype, drugs): (msk[msk.SUBTYPE.eq(subtype) & msk.AGENTS.eq(key)],
                               bpc[bpc.SUBTYPE_BCA.eq(subtype) & bpc.AGENTS.eq(key)])
            for subtype, drugs in ORDER for key in [frozenset(drugs.split(" + "))]}


def regimen_medians():
    return pd.DataFrame([{"subtype": subtype, "drugs": drugs, "msk": C.km(a.PFS_TIME_DAYS, a.PFS_EVENT),
                          "bpc": C.km(b.mixed_time, b.mixed_event), "bpc_365": int(b.mixed_time.gt(365).sum()),
                          "p": logrank_test(a.PFS_TIME_DAYS, b.mixed_time, a.PFS_EVENT, b.mixed_event).p_value}
                         for (subtype, drugs), (a, b) in regimen_groups().items()])


def bpc_endpoint_definitions():
    _, met = load_bpc_endpoints()
    first = met[met.RANK.eq(1)]
    groups = [("All first regimens", first)] + [(s, first[first.SUBTYPE_BCA.eq(s)]) for s in C.SUBTYPES]
    groups += [(f"{s}: {d.lower()}", b) for (s, d), (_, b) in regimen_groups().items()]
    endpoints = [("PFS-M (med-onc note): days; n/events", "tt_pfs_m_g_days", "pfs_m_g_status"),
                 ("PFS-I (released): days; n/events", "tt_pfs_i_g_days", "pfs_i_g_status"),
                 ("PFS-I + Mixed: days; n/events", "mixed_time", "mixed_event")]
    rows = []
    for name, g in groups:
        cells = {}
        for label, time, event in endpoints:
            t, e = g[time].astype(float).to_numpy(), g[event].astype(float).to_numpy()
            ok = ~np.isnan(t) & ~np.isnan(e)
            cells[label] = km_cell(C.km(*G.censor_730(t[ok], e[ok])))
        rows.append({"BPC first regimens (curated subtype, three sites)": name, **cells})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def qc_selection_lines():
    labels = C.load_breast_pfs()[["PATIENT_ID", "LINE", "PFS_TIME_DAYS", "PFS_EVENT"]]
    retained = pd.read_csv("data/design_matrix.csv", low_memory=False)
    removed = pd.read_csv("data/dropped_lines.csv", low_memory=False)
    removed = removed[removed.QC_REASONS.astype(str).str.contains("chemo\\+hormone")]
    removed = removed.drop(columns=["PFS_TIME_DAYS", "PFS_EVENT"]).merge(labels, on=["PATIENT_ID", "LINE"])
    planned = [c for c in retained.columns if c.startswith("PLANNED_AGENT_") and "BONE TREATMENT" not in c]
    selection = {}
    for subtype, drugs in QC_SETS:
        cols = [f"PLANNED_AGENT_{a}" for a in drugs]
        others = [c for c in planned if c not in cols]
        parts = {}
        for label, frame in [(RETAINED, retained), (REMOVED, removed)]:
            d = frame[frame.LINE.eq(1) & frame.PFS_EVENT.isin([0, 1])]
            d = d[C.subtype_label(d.HR, d.HER2).eq(subtype).to_numpy()]
            d = d[(d[cols] > 0).all(axis=1) & (d[others].fillna(0) > 0).sum(axis=1).eq(0)]
            t, e = G.censor_730(d.PFS_TIME_DAYS, d.PFS_EVENT)
            parts[label] = d.assign(PFS_TIME_DAYS=t, PFS_EVENT=e)
        parts[BOTH] = pd.concat(parts.values())
        selection[f"{subtype}: {' + '.join(a.split('_', 1)[1].lower() for a in drugs)}"] = parts
    return selection


def qc_selection_km():
    return {(name, label): C.km(d.PFS_TIME_DAYS, d.PFS_EVENT)
            for name, parts in qc_selection_lines().items() for label, d in parts.items()}


def qc_selection_by_regimen():
    return pd.DataFrame([{"Subtype and exact initial drug set": name, "Lines": label, "Median PFS: days; n/events": km_cell(k)}
                         for (name, label), k in qc_selection_km().items()])


def generate():
    medians = regimen_medians()
    by = {(r.subtype, r.drugs): r for r in medians.itertuples()}
    msk = first_lines()
    grouped_treatments = StringIO()
    with redirect_stdout(grouped_treatments):
        treatment_combinations(pd.read_csv("data/design_matrix.csv").query("LINE == 1"), "AGENT")
    records = pd.DataFrame([{"Subtype and exact drug set": f"{r.subtype}: {r.drugs.lower()}",
                             "MSK-CHORD: days; patients/events": km_cell(r.msk),
                             "BPC PFS-I + Mixed: days; patients/events": km_cell(r.bpc)} for r in medians.itertuples()])
    intervals = pd.DataFrame([{"Subtype and exact drug set": f"{r.subtype}: {r.drugs.lower()}",
                               "MSK-CHORD median (95% CI), days": km_ci(r.msk),
                               "BPC PFS-I + Mixed median (95% CI), days": km_ci(r.bpc),
                               "Log-rank p": f"{r.p:.3f}"} for r in medians.itertuples()])
    t = by[ORDER[5]]
    triplet = pd.DataFrame([{"BPC HR+/HER2+ triplet": "First metastatic regimen",
                            "Patients": t.bpc["n"], "Events": t.bpc["events"],
                            "Median (95% CI), days": km_ci(t.bpc),
                            "At risk after 365 days": t.bpc_365}])
    subtype = []
    for label in C.SUBTYPES:
        g = msk[msk.SUBTYPE.eq(label)]
        median = C.km(g.PFS_TIME_DAYS, g.PFS_EVENT)["median"]
        triplet_n = int(g.AGENTS.eq(TRIPLET).sum())
        subtype.append({"Subtype": label, "First-line median PFS, days": median,
                        "Median PFS, months": f"{median / MONTH_DAYS:.1f}",
                        "Triplet first lines, n (%)": f"{triplet_n}/{len(g)} ({triplet_n / len(g):.1%})"})
    months = pd.DataFrame([{"Subtype and drugs": f"{subtype}: {drugs.lower()}",
                            "MSK median, months": f"{m / MONTH_DAYS:.1f}",
                            "Published median, months": f"{v['median']:.1f}",
                            "Reduction from published median": f"{1 - m / MONTH_DAYS / v['median']:.1%}",
                            "Published source": v["source"]}
                           for (subtype, drugs), v in LITERATURE.items() for m in [by[subtype, drugs].msk["median"]]])
    return "\n\n".join([
        "Regimen-level PFS. BPC PFS-I + Mixed.",
        "KM medians. MSK first lines use the study HR/HER2 fields. BPC first non-GnRH metastatic regimens use the curated subtype; sites are pooled.",
        show(records),
        "Medians with 95% CIs (lifelines KM, log-log transform) and the two-sample log-rank test, same groups",
        show(intervals),
        "First-line PFS by subtype", show(pd.DataFrame(subtype)),
        "Regimen selection: manuscript input groups, in frequency order",
        grouped_treatments.getvalue().strip(),
        "These input groups include Other. They differ from the exact drug sets in the benchmark table.",
        f"Published comparators (kept out of the main table; quoted in prose). Month conversions use {MONTH_DAYS} days per month",
        show(months),
        "BPC endpoint definitions in the same first regimens (730-day cap): clinician-note versus imaging PFS",
        show(bpc_endpoint_definitions()),
        "First lines by QC status, for lines whose initial regimen (agents started within 28 days of line start) is exactly the drug set. Removed lines use the pre-QC labels; 730-day cap. The retained rows match the benchmark table above.",
        show(qc_selection_by_regimen(), left=2),
        "BPC triplet summary", show(triplet),
        "Published sources: https://link.springer.com/article/10.1186/s13058-021-01409-8 ; https://iris.uniupo.it/retrieve/handle/11579/112846/77792/PERUSE%20ANN%20ONCOL%202019.pdf .",
    ]) + "\n"
