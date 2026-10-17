from __future__ import annotations

import pandas as pd

from ..utils import filter_and_sort_input, iter_measurements_before_line
from .base import ModalityFrame
from config import LINE_COL, PATIENT_ID_COL, START_DATE_COL
from typing import Dict, List

PDL1_FILE = "data_timeline_pdl1.tsv"
MMR_FILE = "data_timeline_mmr.tsv"


def make_tri_state_from_last_prior(
    events_df: pd.DataFrame,
    patient_lot_info: pd.DataFrame,
    value_col: str,
    pos_name: str,
    neg_name: str,
    unk_name: str,
) -> pd.DataFrame:
    """Produce mutually exclusive POS/NEG/UNK flags based on the latest prior result."""

    events = events_df[[PATIENT_ID_COL, START_DATE_COL, value_col]].dropna().copy()
    events[START_DATE_COL] = events[START_DATE_COL].astype(int)
    events[value_col] = events[value_col].astype(int)

    rows: List[Dict[str, int]] = []

    for pid, line_id, _, history in iter_measurements_before_line(
        events,
        patient_lot_info,
    ):
        row = {
            PATIENT_ID_COL: pid,
            LINE_COL: line_id,
            pos_name: 0,
            neg_name: 0,
            unk_name: 1,
        }

        if not history.empty:
            last_value = int(history[value_col].iloc[-1])
            if last_value == 1:
                row[pos_name] = 1
                row[neg_name] = 0
                row[unk_name] = 0
            elif last_value == 0:
                row[pos_name] = 0
                row[neg_name] = 1
                row[unk_name] = 0

        rows.append(row)

    out = pd.DataFrame(rows)
    keys = patient_lot_info[[PATIENT_ID_COL, LINE_COL]].drop_duplicates()
    out = keys.merge(out, on=[PATIENT_ID_COL, LINE_COL], how="left")
    return out


def build_pdl1_features(ctx) -> ModalityFrame:
    raw = pd.read_csv(ctx.data_dir / PDL1_FILE, sep="\t")
    pdl1 = filter_and_sort_input(raw, ctx.type_specific_patients)
    pdl1 = pdl1.sort_values(["PATIENT_ID", "START_DATE"]).reset_index(drop=True)
    pdl1["PDL1_POSITIVE"] = pdl1["PDL1_POSITIVE"].apply(
        lambda x: 1 if (isinstance(x, str) and x.upper() == "YES") or x == 1 else 0
    )

    tri = make_tri_state_from_last_prior(
        events_df=pdl1,
        patient_lot_info=ctx.patient_lot_info,
        value_col="PDL1_POSITIVE",
        pos_name="PDL1_POS",
        neg_name="PDL1_NEG",
        unk_name="PDL1_UNKNOWN",
    )
    return ModalityFrame(name="PDL1", frame=tri, temporal=True)


def build_mmr_features(ctx) -> ModalityFrame:
    raw = pd.read_csv(ctx.data_dir / MMR_FILE, sep="\t")
    mmr = filter_and_sort_input(raw, ctx.type_specific_patients)
    mmr = mmr.sort_values(["PATIENT_ID", "START_DATE"]).reset_index(drop=True)
    mmr["MMR_ABSENT"] = mmr["MMR_ABSENT"].apply(
        lambda x: 1 if (isinstance(x, str) and x.upper() == "YES") or x == 1 else 0
    )

    tri = make_tri_state_from_last_prior(
        events_df=mmr,
        patient_lot_info=ctx.patient_lot_info,
        value_col="MMR_ABSENT",
        pos_name="MMR_NEG",
        neg_name="MMR_POS",
        unk_name="MMR_UNKNOWN",
    )
    return ModalityFrame(name="MMR", frame=tri, temporal=True)
