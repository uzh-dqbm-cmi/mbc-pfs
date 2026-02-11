import pandas as pd

HER2_TARGET_GROUPS = {"HER2_MAB", "HER2_ADC", "HER2_TKI"}
CHEMO_HORMONE_OVERLAP_GRACE_DAYS = 0


def agent_to_equiv_group(agent: str) -> str:
    agent = str(agent).upper().strip()

    # Endocrine
    if agent in {"LETROZOLE", "ANASTROZOLE", "EXEMESTANE"}:
        return "ENDO_AI"
    if agent in {"FULVESTRANT"}:
        return "ENDO_SERD"
    if agent in {"TAMOXIFEN", "TOREMIFENE", "RALOXIFENE"}:
        return "ENDO_SERM"
    if agent in {"GOSERELIN", "LEUPROLIDE"}:
        return "ENDO_GNRH"
    if agent in {"MEGESTROL"}:
        return "ENDO_PROGESTIN"

    # CDK4/6
    if agent in {"PALBOCICLIB", "RIBOCICLIB", "ABEMACICLIB"}:
        return "CDK46I"

    # PI3K / AKT / mTOR
    if agent in {"ALPELISIB"}:
        return "PI3KI"
    if agent in {"CAPIVASERTIB"}:
        return "AKTI"
    if agent in {"EVEROLIMUS"}:
        return "MTORI"

    # PARP
    if agent in {"OLAPARIB", "TALAZOPARIB"}:
        return "PARPI"

    # HER2
    if agent in {
        "TRASTUZUMAB",
        "PERTUZUMAB",
        "HYALURONIDASE/PERTUZUMAB/TRASTUZUMAB",
        "HYALURONIDASE-TRASTUZUMAB",
        "MARGETUXIMAB",
    }:
        return "HER2_MAB"
    if agent in {
        "ADO-TRASTUZUMAB EMTANSINE",
        "FAM-TRASTUZUMAB DERUXTECAN",
        "TRASTUZUMAB DERUXTECAN",
        "T-DM1",
        "T-DXD",
    }:
        return "HER2_ADC"
    if agent in {"LAPATINIB", "NERATINIB", "TUCATINIB"}:
        return "HER2_TKI"

    # Immunotherapy
    if agent in {"PEMBROLIZUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "DURVALUMAB"}:
        return "IMMUNO_IO"

    # Chemo (common buckets)
    if agent in {"PACLITAXEL", "DOCETAXEL", "PACLITAXEL PROTEIN-BOUND"}:
        return "CHEMO_TAXANE"
    if agent in {"CARBOPLATIN", "CISPLATIN"}:
        return "CHEMO_PLATINUM"
    if agent in {"CAPECITABINE", "FLUOROPYRIMIDINE", "FLUOROURACIL"}:
        return "CHEMO_FLUOROPYRIMIDINE"
    if agent in {"DOXORUBICIN", "DOXORUBICIN LIPOSOMAL", "EPIRUBICIN"}:
        return "CHEMO_ANTHRACYCLINE"
    if agent in {"GEMCITABINE"}:
        return "CHEMO_GEMCITABINE"
    if agent in {"VINORELBINE"}:
        return "CHEMO_VINORELBINE"
    if agent in {"ERIBULIN"}:
        return "CHEMO_ERIBULIN"
    if agent in {"IXABEPILONE"}:
        return "CHEMO_IXABEPILONE"
    if agent in {"CYCLOPHOSPHAMIDE"}:
        return "CHEMO_CYCLOPHOSPHAMIDE"
    if agent in {"METHOTREXATE"}:
        return "CHEMO_MTX"
    if agent in {"IRINOTECAN"}:
        return "CHEMO_IRINOTECAN"
    if agent in {"ETOPOSIDE"}:
        return "CHEMO_ETOPOSIDE"

    # Angiogenesis/other biologics
    if agent in {"BEVACIZUMAB"}:
        return "BIO_ANGIO"

    if "INVESTIGATIVE" in agent:
        return "INVESTIGATIVE"
    return "OTHER"


def line_overlap_flags(
    df: pd.DataFrame, grace_days: int, line_start_map: dict | None = None
) -> pd.DataFrame:
    """Flag lines where endocrine+chemo or endocrine+immuno overlap beyond a grace window."""

    def any_interval_overlap(
        starts_a: list[int],
        stops_a: list[int],
        starts_b: list[int],
        stops_b: list[int],
    ) -> bool:
        for sa, ea in zip(starts_a, stops_a):
            for sb, eb in zip(starts_b, stops_b):
                overlap_start = max(sa, sb)
                overlap_end = min(ea, eb)
                if (overlap_end - overlap_start) > grace_days:
                    return True
        return False

    def line_sort_key(val: object) -> float | str:
        try:
            return float(val)  # type: ignore[return-value]
        except Exception:
            return str(val)

    def overlapping_lines(a: pd.DataFrame, b: pd.DataFrame, starts: dict) -> set:
        a = a.dropna(subset=["START_DATE", "STOP_DATE"])
        b = b.dropna(subset=["START_DATE", "STOP_DATE"])
        if len(a) == 0 or len(b) == 0:
            return set()

        if starts:
            start_by_line = starts
        else:
            start_by_line = (
                pd.concat([a, b])
                .groupby("LINE", observed=True)["START_DATE"]
                .min()
                .to_dict()
            )

        overlap_lines = set()
        for line_a, ga in a.groupby("LINE", observed=True):
            starts_a = ga["START_DATE"].tolist()
            stops_a = ga["STOP_DATE"].tolist()
            for line_b, gb in b.groupby("LINE", observed=True):
                starts_b = gb["START_DATE"].tolist()
                stops_b = gb["STOP_DATE"].tolist()
                if any_interval_overlap(starts_a, stops_a, starts_b, stops_b):
                    if line_a == line_b:
                        overlap_lines.add(line_a)
                    else:
                        start_a = start_by_line.get(line_a)
                        start_b = start_by_line.get(line_b)
                        if (start_a is not None) and (start_b is not None):
                            later_line = line_a if start_a >= start_b else line_b
                        else:
                            later_line = (
                                line_a
                                if line_sort_key(line_a) >= line_sort_key(line_b)
                                else line_b
                            )
                        overlap_lines.add(later_line)
        return overlap_lines

    rows = []
    for pid, grp in df.groupby("PATIENT_ID", observed=True):
        endo = grp[grp["IS_ENDOCRINE"] | grp["SUBTYPE"].eq("HORMONE")]
        chemo = grp[grp["IS_CHEMO"] | grp["SUBTYPE"].eq("CHEMO")]
        immuno = grp[grp["IS_IMMUNO"] | grp["SUBTYPE"].eq("IMMUNO")]

        patient_starts = (
            line_start_map.get(pid, {}) if line_start_map is not None else {}
        )

        endo_chemo_overlap = overlapping_lines(endo, chemo, patient_starts)
        endo_immuno_overlap = overlapping_lines(endo, immuno, patient_starts)

        for line in sorted(grp["LINE"].unique(), key=line_sort_key):
            rows.append(
                {
                    "PATIENT_ID": pid,
                    "LINE": line,
                    "ENDO_CHEMO_OVERLAP": line in endo_chemo_overlap,
                    "ENDO_IMMUNO_OVERLAP": line in endo_immuno_overlap,
                }
            )
    return pd.DataFrame(rows)


def summarize_lines_for_qc(
    df: pd.DataFrame, line_start_map: dict | None = None
) -> pd.DataFrame:
    """Collapse treatment rows to one record per patient-line with QC helpers."""

    grouped = df.groupby(["PATIENT_ID", "LINE"], as_index=False, observed=True)
    summary = grouped.agg(
        COHORT=("COHORT", "first"),
        LINE_START=("START_DATE", "min"),
        LINE_END=("STOP_DATE", "max"),
        SUBTYPES=("SUBTYPE", lambda x: tuple(sorted(pd.unique(x)))),
        AGENTS=("AGENT_NORM", lambda x: tuple(sorted(pd.unique(x)))),
        EQUIV_GROUPS=("EQUIV_GROUP", lambda x: tuple(sorted(pd.unique(x)))),
        HAS_ENDOCRINE=("IS_ENDOCRINE", "any"),
        HAS_CHEMO=("IS_CHEMO", "any"),
        HAS_HER2_TARGET=("IS_HER2_TARGET", "any"),
        HAS_IMMUNO=("IS_IMMUNO", "any"),
    )
    if line_start_map:

        def _override_line_start(row: pd.Series) -> float:
            return line_start_map.get(row["PATIENT_ID"], {}).get(
                row["LINE"], row["LINE_START"]
            )

        summary["LINE_START"] = summary.apply(_override_line_start, axis=1)
    return summary


def _cohort_label(hr_value: object, her2_value: object) -> str | None:
    try:
        hr = int(hr_value)
        her2 = int(her2_value)
    except (TypeError, ValueError):
        return None
    hr_label = "HRpos" if hr == 1 else "HRneg"
    her2_label = "HER2pos" if her2 == 1 else "HER2neg"
    return f"{hr_label}_{her2_label}"


def build_qc_table(
    treatment_df: pd.DataFrame, line_starts: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Replicate mbc_lot_qc flags to drive hard drops in the pipeline."""

    work = treatment_df.copy()
    work["AGENT_NORM"] = work["AGENT"].astype(str).str.upper().str.strip()
    work["EQUIV_GROUP"] = work["AGENT_NORM"].map(agent_to_equiv_group)
    work["IS_ENDOCRINE"] = work["EQUIV_GROUP"].str.startswith("ENDO_")
    work["IS_CHEMO"] = work["EQUIV_GROUP"].str.startswith("CHEMO_")
    work["IS_HER2_TARGET"] = work["EQUIV_GROUP"].isin(HER2_TARGET_GROUPS)
    work["IS_IMMUNO"] = work["EQUIV_GROUP"].eq("IMMUNO_IO")
    work["COHORT"] = work.apply(
        lambda row: _cohort_label(row.get("HR"), row.get("HER2")), axis=1
    )
    work["SUBTYPE"] = work["SUBTYPE"].astype(str).str.upper()
    work["STOP_DATE"] = work["STOP_DATE"].fillna(work["START_DATE"])

    line_start_map: dict | None = None
    if line_starts is not None and not line_starts.empty:
        starts = (
            line_starts[["PATIENT_ID", "LINE", "LINE_START"]]
            .dropna(subset=["PATIENT_ID", "LINE", "LINE_START"])
            .copy()
        )
        if not starts.empty:
            starts["LINE"] = starts["LINE"].astype(work["LINE"].dtype, copy=False)
            starts["LINE_START"] = starts["LINE_START"].astype(float)
            line_start_map = {
                pid: {
                    int(l): float(ls)
                    for l, ls in grp[["LINE", "LINE_START"]].to_numpy()
                }
                for pid, grp in starts.groupby("PATIENT_ID", observed=True)
            }

    overlap_flags = line_overlap_flags(
        work, CHEMO_HORMONE_OVERLAP_GRACE_DAYS, line_start_map=line_start_map
    )
    line_table = summarize_lines_for_qc(work, line_start_map=line_start_map)
    line_table = line_table.merge(
        overlap_flags, on=["PATIENT_ID", "LINE"], how="left", validate="one_to_one"
    )

    cohort_series = line_table["COHORT"].fillna("")
    line_table["HR_NEG_ENDOCRINE"] = (
        cohort_series.str.startswith("HRneg") & line_table["HAS_ENDOCRINE"]
    )
    line_table["HER2_NEG_HER2_TARGET"] = (
        cohort_series.str.endswith("HER2neg") & line_table["HAS_HER2_TARGET"]
    )
    line_table["IMMUNO_NO_CHEMO"] = line_table["HAS_IMMUNO"] & (
        ~line_table["HAS_CHEMO"]
    )
    line_table["CHEMO_AND_ENDOCRINE_SAME_LINE"] = (
        line_table["HAS_CHEMO"] & line_table["HAS_ENDOCRINE"]
    )
    line_table["QC_HARD_FAIL"] = (
        line_table["HR_NEG_ENDOCRINE"]
        | line_table["ENDO_CHEMO_OVERLAP"].fillna(False)
        | line_table["CHEMO_AND_ENDOCRINE_SAME_LINE"]
    )

    def _qc_reasons(row: pd.Series) -> tuple:
        reasons: list[str] = []
        if row["CHEMO_AND_ENDOCRINE_SAME_LINE"]:
            reasons.append("chemo+hormone in same LoT")
        if row["HR_NEG_ENDOCRINE"]:
            reasons.append("HRneg_endocrine")
        if row["ENDO_CHEMO_OVERLAP"]:
            reasons.append("endo+chemo_overlap_any_line")
        if row["HER2_NEG_HER2_TARGET"]:
            reasons.append("HER2neg_HER2target_review")
        if row["IMMUNO_NO_CHEMO"]:
            reasons.append("IO_without_chemo_review")
        return tuple(reasons)

    line_table["QC_REASONS"] = line_table.apply(_qc_reasons, axis=1)
    return line_table


def qc_from_received_flags(
    df: pd.DataFrame,
    hr_col: str = "HR",
    her2_col: str = "HER2",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Flag QC hard failures using binary RECEIVED_* features instead of raw timelines.

    Existing reasons:
      * HRneg_endocrine
      * chemo+hormone in same LoT (also tagged as endo+chemo_overlap_any_line)
      * HER2neg_HER2target_review
      * IO_without_chemo_review
    """

    work = df.copy()

    def _flag(col: str) -> pd.Series:
        if col not in work.columns:
            return pd.Series(False, index=work.index)
        series = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)
        return series > 0

    has_hormone = _flag("RECEIVED_TREATMENT_HORMONE")
    has_chemo = _flag("RECEIVED_TREATMENT_CHEMO")
    has_immuno = _flag("RECEIVED_TREATMENT_IMMUNO")

    her2_target_cols = [
        "RECEIVED_AGENT_BIOLOGIC_TRASTUZUMAB",
        "RECEIVED_AGENT_BIOLOGIC_PERTUZUMAB",
        "RECEIVED_AGENT_BIOLOGIC_ADO-TRASTUZUMAB EMTANSINE",
        "RECEIVED_AGENT_BIOLOGIC_FAM-TRASTUZUMAB DERUXTECAN",
    ]
    has_her2_target = pd.Series(False, index=work.index)
    for col in her2_target_cols:
        has_her2_target = has_her2_target | _flag(col)

    hr_series = pd.to_numeric(work.get(hr_col, 1), errors="coerce")
    her2_series = pd.to_numeric(work.get(her2_col, 1), errors="coerce")
    hr_neg = hr_series.fillna(1).astype(int) == 0
    her2_neg = her2_series.fillna(1).astype(int) == 0

    chemo_and_endocrine = has_chemo & has_hormone
    endo_chemo_overlap = chemo_and_endocrine
    hr_neg_endocrine = hr_neg & has_hormone
    her2_neg_target = her2_neg & has_her2_target
    immuno_no_chemo = has_immuno & (~has_chemo)

    qc_hard_fail = hr_neg_endocrine | chemo_and_endocrine

    def _reasons(idx: int) -> tuple[str, ...]:
        reasons: list[str] = []
        if chemo_and_endocrine.iloc[idx]:
            reasons.append("chemo+hormone in same LoT")
        if hr_neg_endocrine.iloc[idx]:
            reasons.append("HRneg_endocrine")
        if endo_chemo_overlap.iloc[idx]:
            reasons.append("endo+chemo_overlap_any_line")
        if her2_neg_target.iloc[idx]:
            reasons.append("HER2neg_HER2target_review")
        if immuno_no_chemo.iloc[idx]:
            reasons.append("IO_without_chemo_review")
        return tuple(reasons)

    work["QC_HARD_FAIL"] = qc_hard_fail
    work["QC_REASONS"] = [_reasons(i) for i in range(len(work))]
    # set PFS_EVENT to -1 for QC hard fails
    work.loc[qc_hard_fail, "PFS_EVENT"] = -1

    if verbose:
        summary = {
            "chemo+hormone in same LoT": int(chemo_and_endocrine.sum()),
            "HRneg_endocrine": int(hr_neg_endocrine.sum()),
            "endo+chemo_overlap_any_line": int(endo_chemo_overlap.sum()),
            "HER2neg_HER2target_review": int(her2_neg_target.sum()),
            "IO_without_chemo_review": int(immuno_no_chemo.sum()),
            "total_hard_fail": int(qc_hard_fail.sum()),
        }
        print("QC (received flags) summary:")
        for key, val in summary.items():
            print(f"  {key}: {val}")

    return work
