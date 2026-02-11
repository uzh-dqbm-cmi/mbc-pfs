from __future__ import annotations

import pandas as pd

from .base import ModalityFrame
from ..utils import filter_and_sort_input

LINE_START_WINDOW_DAYS = 28
LINE_RECEIVED_GRACE_DAYS = 28


def _normalize_token(series: pd.Series) -> pd.Series:
    tokens = series.astype(str).str.strip().str.upper()
    return tokens


def _build_top_agent_lookup(treatment: pd.DataFrame) -> dict[str, set[str]]:
    allowed_mask = treatment["AGENT"].notna()
    agent_candidates = treatment.loc[allowed_mask, ["TREATMENT", "AGENT"]]
    if agent_candidates.empty:
        return {}

    counts = (
        agent_candidates.groupby(["TREATMENT", "AGENT"], observed=True)
        .size()
        .reset_index(name="COUNT")
    )
    top_pairs = (
        counts.sort_values(
            ["TREATMENT", "COUNT", "AGENT"], ascending=[True, False, True]
        )
        .groupby("TREATMENT", observed=True)
        .head(5)
    )
    return (
        top_pairs.groupby("TREATMENT", observed=True)["AGENT"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )


def _overlaps(
    start: int, stop: int, interval_start: int, interval_end: int | None
) -> bool:
    if interval_end is None:
        return stop >= interval_start
    return (start <= interval_end) and (stop >= interval_start)


def build_treatment_features(ctx) -> ModalityFrame:
    treatment = pd.read_csv(
        ctx.data_dir / "data_timeline_treatment.tsv", sep="\t", dtype=str
    )
    treatment = filter_and_sort_input(
        treatment, patient_ids=ctx.patient_lot_info["PATIENT_ID"].unique()
    )
    treatment["TREATMENT"] = _normalize_token(treatment["SUBTYPE"])
    treatment["AGENT"] = _normalize_token(treatment["AGENT"])
    treatment["START_DATE"] = treatment["START_DATE"].astype(int)
    treatment["STOP_DATE"] = pd.to_numeric(treatment.get("STOP_DATE"), errors="coerce")
    treatment["STOP_DATE"] = treatment["STOP_DATE"].fillna(treatment["START_DATE"] + 1)
    treatment["STOP_DATE"] = treatment["STOP_DATE"].astype(int)

    top_agents = _build_top_agent_lookup(treatment)

    lot_info = ctx.patient_lot_info[
        ["PATIENT_ID", "LINE", "LINE_START", "EVENT_DAY"]
    ].copy()
    lot_info = lot_info.sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)
    lot_info["EVENT_DAY"] = lot_info["EVENT_DAY"].astype("Int64")

    records: list[dict] = []
    for pid, lines in lot_info.groupby("PATIENT_ID", sort=False):
        patient_treatments = treatment[treatment["PATIENT_ID"] == pid]
        for _, line_row in lines.iterrows():
            line_start = int(line_row["LINE_START"])
            event_day_int = int(line_row.get("EVENT_DAY"))
            planned_end = line_start + LINE_START_WINDOW_DAYS
            # if it's received shortly before line ends, does not count it
            received_end = event_day_int - LINE_START_WINDOW_DAYS

            features: dict[str, int | str] = {
                "PATIENT_ID": pid,
                "LINE": line_row["LINE"],
            }

            for tx in patient_treatments.itertuples(index=False):
                start = int(tx.START_DATE)
                stop = int(tx.STOP_DATE)
                treat_name = tx.TREATMENT
                agent_name = tx.AGENT

                planned_hit = _overlaps(start, stop, line_start, planned_end)
                received_hit = _overlaps(start, stop, line_start, received_end)

                if planned_hit:
                    features[f"PLANNED_TREATMENT_{treat_name}"] = 1
                if received_hit:
                    features[f"RECEIVED_TREATMENT_{treat_name}"] = 1

                if pd.isna(agent_name) or agent_name is None:
                    continue
                in_top = agent_name in top_agents.get(treat_name, set())
                if planned_hit:
                    key = (
                        f"PLANNED_AGENT_{treat_name}_{agent_name}"
                        if in_top
                        else f"PLANNED_AGENT_{treat_name}_OTHER"
                    )
                    features[key] = 1
                if received_hit:
                    key = (
                        f"RECEIVED_AGENT_{treat_name}_{agent_name}"
                        if in_top
                        else f"RECEIVED_AGENT_{treat_name}_OTHER"
                    )
                    features[key] = 1

            records.append(features)

    merged = pd.DataFrame.from_records(records)
    feature_cols = [c for c in merged.columns if c not in {"PATIENT_ID", "LINE"}]
    merged[feature_cols] = merged[feature_cols].fillna(0).astype("int8")
    merged = merged.sort_values(["PATIENT_ID", "LINE"]).reset_index(drop=True)
    return ModalityFrame(name="TREATMENT", frame=merged, temporal=True)
