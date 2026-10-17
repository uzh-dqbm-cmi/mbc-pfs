from dataclasses import dataclass

import numpy as np
import pandas as pd

import build_pfs
from build_pfs import apply_censor, assign_patient_lines_and_pfs, flag_short_lines
from config import DAY_SPLIT, LINE_START_WINDOW_DAYS
from qc import qc_from_received_flags
from analyses.common import bpc_adapter as A

LHRH_AGENTS = frozenset({"LEUPROLIDE", "GOSERELIN", "HISTRELIN", "TRIPTORELIN"})


@dataclass
class Reconstruction:
    lines: pd.DataFrame
    courses_assigned: pd.DataFrame


def run(g: A.Genie) -> Reconstruction:
    meta = A.build_meta(g).set_index("PATIENT_ID")
    courses_raw = A.build_drug_courses(g)
    courses_raw = courses_raw[courses_raw.PATIENT_ID.isin(meta.index)]
    tx = courses_raw[courses_raw.AGENT.ne(A.INVESTIGATIVE_AGENT)]
    prog = A.build_progression(g)
    prog = prog[prog.PATIENT_ID.isin(meta.index)]

    def post_onset(df):
        stage4 = df.PATIENT_ID.map(meta.STAGE_IV).astype(bool)
        onset = df.PATIENT_ID.map(meta.MET_ONSET).fillna(np.inf)
        return df[stage4 | (df.START_DATE >= onset)]

    tx, prog = post_onset(tx), post_onset(prog)
    pids = set(tx.PATIENT_ID) & set(prog.PATIENT_ID)
    tx = tx[tx.PATIENT_ID.isin(pids)]
    prog = prog[prog.PATIENT_ID.isin(pids)]
    prog_by = {pid: grp[["START_DATE", "PROGRESSION"]].reset_index(drop=True)
               for pid, grp in prog.groupby("PATIENT_ID")}
    rows, assigned = [], []
    algo_cols = ["PATIENT_ID", "START_DATE", "STOP_DATE", "AGENT", "SUBTYPE", "REGIMEN_NUMBER", "DRUG_K"]
    build_pfs.prior_measurement_map = A.imaging_days(g)
    try:
        for pid, ptx in tx.groupby("PATIENT_ID"):
            death = meta.at[pid, "DEATH_DAY"]
            death = None if pd.isna(death) else int(death)
            try:
                lines_tx, labels = assign_patient_lines_and_pfs(ptx[algo_cols], prog_by[pid], DAY_SPLIT,
                                                                LINE_START_WINDOW_DAYS, death)
            except ValueError:
                lines_tx, labels = assign_patient_lines_and_pfs(ptx[algo_cols], prog_by[pid], DAY_SPLIT,
                                                                LINE_START_WINDOW_DAYS, None)
            rows.extend(labels)
            assigned.append(lines_tx)
    finally:
        build_pfs.prior_measurement_map = None
    lines = pd.DataFrame(rows).sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)
    courses_assigned = pd.concat(assigned, ignore_index=True)
    lines["RAW_PFS_TIME_DAYS"] = lines.PFS_TIME_DAYS.astype(float)
    lines["RAW_PFS_EVENT"] = lines.PFS_EVENT
    lines = flag_short_lines(lines)
    lines["PFS_TIME_DAYS"] = lines.PFS_TIME_DAYS.astype(float)
    lines = apply_censor(lines)
    lines = lines.merge(meta[["SITE", "HR"]], left_on="PATIENT_ID", right_index=True, how="left")
    lines = _apply_qc(lines, courses_raw)
    lines["STAGE"] = np.where(lines.PFS_EVENT.eq(-1), "excluded_pre_qc",
                              np.where(lines.QC_FAIL, "excluded_qc", "final"))
    agents = courses_assigned.groupby(["PATIENT_ID", "LINE"]).AGENT.agg(lambda s: "+".join(sorted(set(s))))
    first = courses_assigned.merge(lines[["PATIENT_ID", "LINE", "LINE_START"]], on=["PATIENT_ID", "LINE"])
    first = first[first.START_DATE.eq(first.LINE_START)]
    first_agents = first.groupby(["PATIENT_ID", "LINE"]).AGENT.agg(lambda s: "+".join(sorted(set(s))))
    lines = lines.merge(agents.rename("AGENTS"), left_on=["PATIENT_ID", "LINE"], right_index=True, how="left")
    lines = lines.merge(first_agents.rename("FIRST_AGENTS"), left_on=["PATIENT_ID", "LINE"], right_index=True, how="left")
    return Reconstruction(lines, courses_assigned)


def _apply_qc(lines, courses):
    by_pid = dict(tuple(courses.groupby("PATIENT_ID")))
    received = []
    for row in lines.itertuples():
        tx = by_pid[row.PATIENT_ID]
        stop = tx.STOP_DATE.fillna(tx.START_DATE + 1)
        hit = tx[(tx.START_DATE <= row.EVENT_DAY - LINE_START_WINDOW_DAYS) & (stop >= row.LINE_START)]
        classes = set(hit.SUBTYPE) if row.PFS_EVENT != -1 else set()
        received.append({"RECEIVED_TREATMENT_HORMONE": "HORMONE" in classes,
                         "RECEIVED_TREATMENT_CHEMO": "CHEMO" in classes})
    flags = pd.DataFrame(received, index=lines.index).assign(HR=lines.HR, HER2=1, PFS_EVENT=lines.PFS_EVENT)
    flags = qc_from_received_flags(flags, verbose=False)
    lines["QC_FAIL"] = flags.QC_HARD_FAIL
    lines["QC_CHEMO_ENDO"] = flags.QC_REASONS.map(lambda s: "chemo+hormone in same LoT" in s)
    lines["QC_HRNEG_ENDO"] = flags.QC_REASONS.map(lambda s: "HRneg_endocrine" in s)
    return lines
