from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tabulate import tabulate

from config import RADIOLOGY_REPORT_WINDOW_DAYS
from genie_validation import paths as P
from analyses.common import agreement as G, bpc_adapter as A, bpc_data as B, cohort as C, reconstruction as R

SITE_GROUPS = ("DFCI", "VICC", "MSK")
REGIMEN_ROWS = ["Same regimen", "Different: treatment changed without counted progression",
                "Different: investigational drug masked in BPC", "Different: other recording differences",
                "Different: not attributed"]
REGIMEN_GROUPS = {"treatment changed without counted progression": REGIMEN_ROWS[1],
                  "investigational drug in the BPC regimen": REGIMEN_ROWS[2],
                  "other grouping difference": REGIMEN_ROWS[4]}
ENDPOINT_ROWS = ["Same event indicator and exact PFS duration",
                 "BPC event is a progression report in the first 28 days; our rule does not count it",
                 "Progression reported within 28 days after a regimen switch; BPC censors at the switch",
                 "Both censored; censoring-date rules differ", "Start dates differ", "Other"]


def named(drugs):
    return bool(drugs) and not any("OTHER" in a or "INVESTIGATIVE" in a for a in drugs)


@lru_cache(maxsize=1)
def load_bpc_endpoints():
    reg = B.add_next_start(B.load_regimens())
    rows = B.mixed_endpoints(reg)
    met = reg.sort_values(["record_id", "regimen_number", "pfs_i_g_status", "ca_seq"],
                          na_position="last", kind="mergesort").drop_duplicates(["record_id", "regimen_number"])
    met = met[met.pfs_eligible & met.ca_type.eq("Breast Cancer")].copy()
    fields = ["record_id", "regimen_number", "ca_seq"]
    met = met.merge(rows[fields + ["mixed_event", "mixed_time"]], on=fields, validate="one_to_one")
    met["AGENTS"] = met.regimen_drugs.map(lambda s: C.canonical_agents(t for t in str(s).split(",") if t.strip()))
    met["SUBTYPE_BCA"] = met.bca_subtype.map(C.BCA_SUBTYPE_MAP).fillna("unknown")
    met = met.sort_values(["record_id", "start", "regimen_number"], kind="mergesort")
    met = met[~met.AGENTS.map(lambda s: bool(s) and s <= C.GNRH_AGENTS)]
    met["RANK"] = met.groupby("record_id").cumcount() + 1
    return rows, met


def show(frame, left=1):
    return tabulate(frame, headers="keys", tablefmt="psql", showindex=False, disable_numparse=True,
                    colalign=["left"] * left + ["right"] * (frame.shape[1] - left))


def frac(n, d):
    return f"{int(n):,}/{int(d):,} ({n / d:.1%})"


def days(v):
    return "NR" if not np.isfinite(v) else f"{v:g}"


def km_cell(r):
    return f"{days(r['median'])}; {r['n']}/{r['events']}"


def km_ci(r, dash="-"):
    return f"{days(r['median'])} ({days(r['lo'])}{dash}{days(r['hi'])})"


def pair_rows(ours, ref, time, event, tol=14, retained_only=True):
    ours = (ours[ours.STAGE.eq("final")] if retained_only else ours).reset_index(drop=True).copy()
    ref = ref.copy()
    for frame, column, separator in [(ours, "AGENTS", "+"), (ref, "regimen_drugs", ",")]:
        frame["DRUG_SET"] = frame[column].fillna("").map(
            lambda s: C.strip_gnrh(C.canonical_agents(s.split(separator))))
        frame["NAMED"] = frame.DRUG_SET.map(named)
        frame["MATCH_KEY"] = frame.PATIENT_ID + "|" + frame.DRUG_SET.map(lambda s: " + ".join(sorted(s)))
    ours["A_ID"], ours["A_START"] = ours.index, ours.LINE_START
    ref["B_START"] = ref.START
    a = ours.loc[ours.NAMED, ["A_ID", "A_START", "MATCH_KEY"]].rename(columns={"MATCH_KEY": "PATIENT_ID"})
    b = ref.loc[ref.NAMED, ["B_ID", "B_START", "MATCH_KEY"]].rename(columns={"MATCH_KEY": "PATIENT_ID"})
    links = G.greedy_match(a, b, tol).drop(columns="PATIENT_ID")
    ref["BPC_PFS_DAYS"], ref["BPC_EVENT"] = G.censor_730(ref[time], ref[event])
    ours["OUR_PFS_DAYS"], ours["OUR_EVENT"] = G.censor_730(ours.RAW_PFS_TIME_DAYS, ours.RAW_PFS_EVENT)
    return ours.merge(links, on="A_ID", how="left").merge(
        ref[["B_ID", "B_START", "DRUG_SET", "NEXT_NONLHRH_START", "BPC_PFS_DAYS", "BPC_EVENT"]].rename(
            columns={"DRUG_SET": "BPC_DRUG_SET"}), on="B_ID", how="left")


def reference_regimens(rows, regimens):
    ref = G.reference_regimens(regimens)
    derived = rows.rename(columns={"record_id": "PATIENT_ID"})[
        ["PATIENT_ID", "ca_seq", "regimen_number", "mixed_time", "mixed_event"]]
    return ref.merge(derived, on=["PATIENT_ID", "ca_seq", "regimen_number"],
                     how="left", validate="one_to_one")


def same_patients(ref, lines):
    return ref[ref.PATIENT_ID.isin(set(lines.PATIENT_ID))]


def site_columns(values):
    return {site: values(site) for site in SITE_GROUPS}


def positive_days(g):
    imaging = A.build_imaging(g)
    return {pid: set(d.START_DATE) for pid, d in
            imaging[imaging.LABEL.isin(["Progressing", "Mixed"])].groupby("PATIENT_ID")}


def matched_pairs(mixed, ref):
    r = pair_rows(mixed, ref, "mixed_time", "mixed_event")
    r = r[r.B_ID.notna()].copy()
    r["DIFF"] = r.OUR_PFS_DAYS - r.BPC_PFS_DAYS
    r["SAME_LABEL"] = r.OUR_EVENT.eq(r.BPC_EVENT)
    r["EXACT"] = r.SAME_LABEL & r.DIFF.eq(0)
    return r


def same_regimen_pairs(mixed, ref, positive):
    r = matched_pairs(mixed, ref)
    end_bpc = r.B_START + r.BPC_PFS_DAYS
    r["EARLY_BPC_EVENT"] = r.BPC_EVENT.eq(1) & r.BPC_PFS_DAYS.lt(28) & pd.Series(
        [e in positive.get(pid, set()) for pid, e in zip(r.PATIENT_ID, end_bpc)], index=r.index)
    r["BOTH_CENSORED"] = r.OUR_EVENT.eq(0) & r.BPC_EVENT.eq(0)
    return r


@lru_cache(maxsize=1)
def reconstruction():
    rows, _ = load_bpc_endpoints()
    g = A.load()
    regimens = A.build_regimens(g)
    ref = reference_regimens(rows, regimens)
    rec = R.run(g)
    return SimpleNamespace(rows=rows, g=g, regimens=regimens, ref=ref, lines=rec.lines,
                           courses=rec.courses_assigned,
                           pairs=same_regimen_pairs(rec.lines, ref, positive_days(g)))


def retained_flow(lines, g, ref):
    meta = A.build_meta(g)
    ref = same_patients(ref, lines)
    pre, qc, final = (lines.STAGE.eq(s) for s in ("excluded_pre_qc", "excluded_qc", "final"))
    gate = f"no_radiology_within_{RADIOLOGY_REPORT_WINDOW_DAYS}_prior"

    def count(frame, mask=True):
        return lambda s: f"{(frame.SITE.eq(s) & mask).sum():,}"

    def patients(mask):
        return lambda s: f"{lines.loc[lines.SITE.eq(s) & mask, 'PATIENT_ID'].nunique():,}"

    steps = [
        ("GENIE BPC breast cancer patients", lambda s: f"{g.patients.institution.eq(s).sum():,}"),
        ("Index cancer is breast carcinoma (breast sarcoma excluded)", count(meta)),
        ("Metastatic disease recorded (stage IV, or a distant-metastasis date)",
         count(meta, meta.STAGE_IV | meta.MET_ONSET.notna())),
        ("At least one metastatic-setting drug record and one imaging assessment", patients(True)),
        ("Treatment episodes formed by our algorithm", count(lines)),
        (f"Excluded: no imaging report in the {RADIOLOGY_REPORT_WINDOW_DAYS} days before the episode start",
         count(lines, pre & lines.LINE_SOURCE.eq(gate))),
        ("Excluded: no imaging assessment or death after the episode start",
         count(lines, pre & lines.LINE_SOURCE.eq("no_radiology_follow_up"))),
        ("Excluded: PFS shorter than 28 days",
         count(lines, pre & lines.LINE_SOURCE.eq("line_too_short_for_effectiveness"))),
        ("Excluded by QC: chemotherapy and endocrine therapy in the same episode",
         count(lines, qc & lines.QC_CHEMO_ENDO)),
        ("Excluded by QC: endocrine therapy in HR-negative disease",
         count(lines, qc & lines.QC_HRNEG_ENDO & ~lines.QC_CHEMO_ENDO)),
        ("Retained episodes (patients)", lambda s: f"{count(lines, final)(s)} ({patients(final)(s)})"),
        ("For comparison: non-LHRH BPC regimens with PFS labels in the same patients", count(ref))]
    return pd.DataFrame([{"Step": name, **site_columns(f)} for name, f in steps])


def conditional_agreement(lines, pairs):
    retained = lines[lines.STAGE.eq("final")]
    return pd.DataFrame([{"SITE": s, "RETAINED": retained.SITE.eq(s).sum(), "PAIRS": len(p),
                          "SAME_LABEL": p.SAME_LABEL.sum(), "EXACT": p.EXACT.sum(),
                          "KAPPA": G.kappa(p.OUR_EVENT, p.BPC_EVENT)}
                         for s in SITE_GROUPS for p in [pairs[pairs.SITE.eq(s)]]])


def event_concordance(agreement):
    return pd.DataFrame([{"BPC site": r.SITE, "Same-regimen pairs": f"{r.PAIRS:,}",
                          "Same event indicator": frac(r.SAME_LABEL, r.PAIRS),
                          "Cohen's kappa": f"{r.KAPPA:.3f}"} for r in agreement.itertuples()])


def duration_agreement(pairs):
    same = pairs[pairs.SAME_LABEL]
    return pd.DataFrame([{"BPC site": s, "Median PFS difference, days": d.DIFF.median(),
                          "Lin's CCC": f"{G.lin_ccc(d.OUR_PFS_DAYS, d.BPC_PFS_DAYS):.3f}"}
                         for s in SITE_GROUPS for d in [same[same.SITE.eq(s)]]])


def lines_per_patient(lines, ref):
    out = []
    for name, frame in [("Episodes formed by our algorithm", lines),
                        ("Non-LHRH BPC regimens with PFS labels", same_patients(ref, lines))]:
        def cell(site, frame=frame):
            patients = lines.loc[lines.SITE.eq(site), "PATIENT_ID"].unique()
            n = frame[frame.SITE.eq(site)].groupby("PATIENT_ID").size().reindex(patients, fill_value=0)
            return f"{n.mean():.1f}; {n.median():g} ({n.quantile(.25):g}-{n.quantile(.75):g})"
        out.append({"Per patient: mean; median (IQR)": name, **site_columns(cell)})
    return pd.DataFrame(out)


def boundary_agreement(lines, ref):
    ref = same_patients(ref, lines)
    delta = G.nearest_abs(ref, "START", lines, "LINE_START")
    return pd.DataFrame([{"BPC regimen start near a reconstructed start": f"Within {tol} days",
                          **site_columns(lambda s: frac((delta[ref.SITE.eq(s).to_numpy()] <= tol).sum(),
                                                       ref.SITE.eq(s).sum()))}
                         for tol in (14, 28)])


def regimen_fate(mixed, ref):
    ref = same_patients(ref, mixed)
    r = pair_rows(mixed, ref, "mixed_time", "mixed_event", retained_only=False)
    qc = r[r.B_ID.notna() & r.STAGE.eq("excluded_qc")]
    return pd.DataFrame([{"QC-excluded matching BPC regimens": "Same drugs; starts within 14 days",
                          **site_columns(lambda s: frac(qc.SITE.eq(s).sum(), ref.SITE.eq(s).sum()))}])


def unmatched_causes(mixed, courses, regimens, ref):
    r = pair_rows(mixed, ref, "mixed_time", "mixed_event")
    unmatched = r[r.B_ID.isna()]
    own = courses[~courses.AGENT.isin(R.LHRH_AGENTS)]
    own = {k: g for k, g in own.groupby(["PATIENT_ID", "LINE"])}
    regimens = regimens.set_index(["PATIENT_ID", "regimen_number"])

    def cause(e):
        if not e.NAMED:
            return "no named drug"
        c = own[(e.PATIENT_ID, e.LINE)]
        if (c.START_DATE < e.LINE_START).any():
            return "drug carried over from an earlier episode"
        source = regimens.loc[[(e.PATIENT_ID, n) for n in sorted(set(c.REGIMEN_NUMBER))]]
        if len(source) > 1:
            return ("BPC records drugs started together as separate regimens"
                    if (source.START - e.LINE_START).abs().le(14).sum() > 1
                    else "treatment changed without counted progression")
        x = source.iloc[0]
        if not x.LABELLED:
            return "BPC regimen has no PFS label"
        if abs(x.START - e.LINE_START) > 14:
            first = str(e.FIRST_AGENTS) if pd.notna(e.FIRST_AGENTS) else ""
            return ("GnRH agonist started first" if first and set(first.split("+")) <= R.LHRH_AGENTS
                    else "other grouping difference")
        return "investigational drug in the BPC regimen" if x.IS_INVESTIGATIONAL else "other grouping difference"
    causes = unmatched.apply(cause, axis=1)
    return r, causes


def regimen_category(mixed, courses, regimens, ref):
    r, causes = unmatched_causes(mixed, courses, regimens, ref)
    category = pd.Series(REGIMEN_ROWS[0], index=r.index)
    category[causes.index] = causes.map(lambda c: REGIMEN_GROUPS.get(c, REGIMEN_ROWS[3]))
    return r.assign(CATEGORY=category)


def endpoint_category(pairs):
    end_ours = pairs.LINE_START + pairs.OUR_PFS_DAYS
    end_bpc = pairs.B_START + pairs.BPC_PFS_DAYS
    after_switch = (pairs.OUR_EVENT.eq(1) & pairs.BPC_EVENT.eq(0) & end_bpc.eq(pairs.NEXT_NONLHRH_START)
                    & (end_ours - end_bpc).between(1, 27))
    unequal = pairs.SAME_LABEL & pairs.DIFF.ne(0)
    return pd.Series(np.select(
        [pairs.EXACT, pairs.EARLY_BPC_EVENT, after_switch, unequal & pairs.BOTH_CENSORED,
         unequal & pairs.DELTA_B_MINUS_A.ne(0)], ENDPOINT_ROWS[:5], default=ENDPOINT_ROWS[5]), index=pairs.index)


def breakdown(category, site, rows, unit, total):
    n = site.value_counts()
    label, members = total

    def cell(mask, s):
        k = int((mask & site.eq(s)).sum())
        return f"{k:,} ({k / n[s]:.1%})"
    masks = [(row, category.eq(row)) for row in rows] + [(label, category.isin(members))]
    return pd.DataFrame([{"": row, **{f"{s} ({n[s]:,} {unit})": cell(mask, s) for s in SITE_GROUPS}}
                         for row, mask in masks])


def regimen_breakdown(r):
    return breakdown(r.CATEGORY, r.SITE, REGIMEN_ROWS[:3], "episodes",
                     ("Same regimen or a listed grouping difference", REGIMEN_ROWS[:3]))


def endpoint_breakdown(pairs):
    exact = "Same label and exact PFS duration"
    censoring = "Censoring rules differ: BPC censors at the next regimen start or last contact"
    category = endpoint_category(pairs).replace(
        {ENDPOINT_ROWS[0]: exact, ENDPOINT_ROWS[2]: censoring, ENDPOINT_ROWS[3]: censoring})
    rows = [exact, censoring, ENDPOINT_ROWS[1]]
    return breakdown(category, pairs.SITE, rows, "pairs", ("Exact agreement or a listed endpoint-rule difference", rows))


def harmonized_endpoint(r, positive, negative):
    stop = np.inf if r.is_lhrh or np.isnan(r.next_start) else r.next_start
    days = positive.get(r.record_id, B.EMPTY)
    ok = (days >= r.start + 28) & (days < stop + 28)
    death = r.hybrid_death_int if pd.notna(r.hybrid_death_int) and r.hybrid_death_int <= stop else np.nan
    ends = [t for t in (days[ok].min() if ok.any() else np.nan, death) if not np.isnan(t)]
    if ends:
        return 1, min(ends) - r.start
    end = min(stop, r.dob_lastalive_int)
    n = negative.get(r.record_id, B.EMPTY)
    n = n[(n > r.start) & (n <= end)]
    end = n.max() if len(n) else end
    return 0, end - r.start


def rule_agreement(mixed, regimens, rows):
    imaging = B.load_imaging()
    positive = B.days_by_patient(imaging, [B.PD, B.MIXED])
    negative = B.days_by_patient(imaging, list(B.NON_PD))
    rows = rows.copy()
    rows[["mixed_event", "mixed_time"]] = [harmonized_endpoint(r, positive, negative) for r in rows.itertuples()]
    pairs = matched_pairs(mixed, reference_regimens(rows, regimens))
    return pd.DataFrame([{"BPC site": s, "Exact agreement after three endpoint rule changes":
                          frac(pairs.EXACT[pairs.SITE.eq(s)].sum(), pairs.SITE.eq(s).sum())}
                         for s in SITE_GROUPS])


def external_counts(rows):
    design = B.validation_labels(pd.read_csv(P.DESIGN_MATRIX_OUT, usecols=["PATIENT_ID", "LINE"]), rows)
    return pd.DataFrame([{"External validation site": site, "Regimens": len(d), "Patients": d.PATIENT_ID.nunique()}
                         for site in ("DFCI", "VICC") for d in [design[design.PATIENT_ID.str.contains(f"-{site}-")]]])


def generate():
    d = reconstruction()
    parts = ["Reconstruction agreement. BPC PFS-I + Mixed.",
             "A match requires the same patient, the same named drug set without GnRH agents, and starts within 14 days.",
             "The study algorithm uses a 28-day grace period, a 90-day prior-imaging window, and a 730-day cap.",
             "Subtotals are category sums. They are not agreement after a rule change.",
             "Same regimen or a listed grouping difference",
             show(regimen_breakdown(regimen_category(d.lines, d.courses, d.regimens, d.ref))),
             "Exact agreement or a listed endpoint-rule difference", show(endpoint_breakdown(d.pairs)),
             "Event-indicator agreement", show(event_concordance(conditional_agreement(d.lines, d.pairs))),
             "PFS duration, among pairs with the same label", show(duration_agreement(d.pairs)),
             "Cohort flow", show(retained_flow(d.lines, d.g, d.ref)),
             "Lines per patient", show(lines_per_patient(d.lines, d.ref)),
             "QC exclusions, as a share of labelled BPC regimens", show(regimen_fate(d.lines, d.ref)),
             "Nearest-start counts; no drug condition", show(boundary_agreement(d.lines, d.ref)),
             "Exact agreement after three endpoint rule changes",
             show(rule_agreement(d.lines, d.regimens, d.rows)),
             "External validation cohort", show(external_counts(d.rows)),
             "Source rules: BPC starts a new regimen after a break of 8 weeks.",
             "References 15 and 16 are external source identifiers."]
    return "\n\n".join(parts) + "\n"
