import pandas as pd


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
