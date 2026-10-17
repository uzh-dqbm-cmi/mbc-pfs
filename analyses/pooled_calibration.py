from pathlib import Path

import numpy as np
import pandas as pd

from analyses.calibration import HORIZONS, diagnostic_fits, ipcw, risk_basis

PREDICTIONS = Path("results/rebuttal/external/recalibration_predictions_cv.csv")
OUTPUT = Path("results/rebuttal/external/pooled_ici.csv")


def pooled_ici():
    data = pd.read_csv(PREDICTIONS)
    counts = np.ones(len(data), dtype=int)
    rows = []
    for horizon in HORIZONS:
        y, weights, _ = ipcw(data.PFS_TIME_DAYS.to_numpy(), data.PFS_EVENT.to_numpy(), horizon, counts)
        for variant, prefix in [("Original", "original_risk_"), ("Site-specific CV update", "risk_")]:
            prob, z, spline, _ = risk_basis(data[f"{prefix}{horizon}"].to_numpy())
            metrics, _ = diagnostic_fits(y, weights, z, np.asarray(spline), prob, counts)
            rows.append(dict(cohort="DFCI + VICC", lines=len(data), patients=data.PATIENT_ID.nunique(),
                             horizon_days=horizon, variant=variant, ICI_percentage_points=100 * metrics[3]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = pooled_ici()
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
