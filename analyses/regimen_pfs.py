from functools import lru_cache

import pandas as pd

from genie_validation.regimen_compare import build_msk_line_regimens
from analyses.common import agreement as G, cohort as C
from analyses.reconstruction import load_bpc_endpoints

ORDER = [("HR+/HER2-", "LETROZOLE + PALBOCICLIB"), ("HR+/HER2-", "LETROZOLE"),
         ("HR-/HER2+", "PACLITAXEL + PERTUZUMAB + TRASTUZUMAB"), ("TNBC", "CAPECITABINE"),
         ("TNBC", "PACLITAXEL"), ("HR+/HER2+", "PACLITAXEL + PERTUZUMAB + TRASTUZUMAB"),
         ("HR+/HER2+", "LETROZOLE")]
DEMICHELE = "DeMichele 2021, weighted rwPFS"
LITERATURE = {ORDER[0]: dict(source=DEMICHELE, arm="Palbociclib plus letrozole", n=772, median=20.0, lo=17.5, hi=21.9),
              ORDER[1]: dict(source=DEMICHELE, arm="Letrozole alone", n=658, median=11.9, lo=10.5, hi=13.7)}
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
                          "bpc": C.km(b.mixed_time, b.mixed_event)}
                         for (subtype, drugs), (a, b) in regimen_groups().items()])


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
