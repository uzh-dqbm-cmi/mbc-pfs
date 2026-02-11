from __future__ import annotations

from .base import ModalityFrame


def build_prior_lines(ctx) -> ModalityFrame:
    df = ctx.patient_lot_info[["PATIENT_ID", "LINE"]].copy()
    df["NUM_PRIOR_LINES"] = df.groupby("PATIENT_ID")["LINE"].rank(method="first") - 1
    df["NUM_PRIOR_LINES"] = df["NUM_PRIOR_LINES"].astype(int)
    return ModalityFrame(name="PRIOR_LINES", frame=df, temporal=True)
