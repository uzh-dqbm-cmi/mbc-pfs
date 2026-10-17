from pathlib import Path
import re

import pandas as pd


def generate(context=None):
    checklist = Path("data/rebuttal/TRIPOD_AI_checklist.txt")
    if not checklist.exists():
        return "TRIPOD+AI checklist is not distributed with this repository.\n"
    source = checklist.read_text()
    entries = re.findall(r"^\|\s*(\d+[a-z]?)\s*\|", source, flags=re.MULTILINE)
    dm = pd.read_csv("data/design_matrix.csv", usecols=["PATIENT_ID"])
    ext = pd.read_csv("data/rebuttal/calibration/horizon_predictions.csv")
    rows = ["TRIPOD+AI reporting checklist", f"Checklist entries: {len(entries)}.",
            f"MSK: {len(dm):,} lines; {dm.PATIENT_ID.nunique():,} patients."]
    for site in ("DFCI", "VICC"):
        d = ext.loc[ext.cohort.eq(site)]
        rows.append(f"{site}: {len(d):,} regimens; {d.PATIENT_ID.nunique():,} patients.")
    rows.extend(["No clinical net-benefit estimate or prospective sample-size calculation is reported.", source])
    return "\n\n".join(rows) + "\n"
