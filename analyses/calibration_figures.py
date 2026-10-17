from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("manuscript/figures/extended_data")


def calibration_curves():
    curves = pd.read_csv("results/rebuttal/calibration/curves.csv")
    fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, site in enumerate(("MSK_OOF", "DFCI", "VICC")):
        for col, horizon in enumerate((180, 365, 730)):
            ax = axes[row, col]
            d = curves.loc[curves.cohort.eq(site) & curves.horizon.eq(horizon)]
            ax.plot([0, 100], [0, 100], "--", color="0.5", lw=1, label="Ideal")
            ax.fill_between(100*d.predicted_risk, 100*d.smooth_lo, 100*d.smooth_hi,
                            color="#0072B2", alpha=.17, label="95% interval")
            ax.plot(100*d.predicted_risk, 100*d.observed_risk_smooth,
                    color="#0072B2", lw=1.7, label="Calibration curve")
            ax.set(title=f'{site.replace("MSK_OOF", "MSK-CHORD")} | {horizon} days',
                   xlim=(0, 100), ylim=(0, 100))
            if row == 2: ax.set_xlabel("Predicted progression risk (%)")
            if col == 0: ax.set_ylabel("Estimated observed risk (%)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.savefig(OUT / "calibration_curves.png")
    plt.close(fig)


def external_tertiles():
    tertiles = pd.read_csv("results/rebuttal/external/recalibration_tertiles.csv")
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, site in enumerate(("DFCI", "VICC")):
        for col, horizon in enumerate((180, 365, 730)):
            ax = axes[row, col]
            d = tertiles.loc[tertiles.cohort.eq(site) & tertiles.horizon.eq(horizon)].set_index("risk_tertile").loc[["Low", "Middle", "High"]]
            for key, label, color, marker in [("observed_KM_survival", "Observed KM", "0.2", "o"), ("predicted_survival", "Original", "#0072B2", "s"), ("recalibrated_survival", "Local CV update", "#D55E00", "D")]:
                ax.plot(d.index, 100*d[key], marker=marker, label=label, color=color, lw=1.3)
            ax.set(title=f"{site} | {horizon} days", ylim=(0, 100))
            if row == 1: ax.set_xlabel("Original predicted-risk tertile")
            if col == 0: ax.set_ylabel("PFS probability (%)")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUT / "external_tertiles.png")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    OUT.mkdir(parents=True, exist_ok=True)
    calibration_curves()
    external_tertiles()


if __name__ == "__main__":
    main()
