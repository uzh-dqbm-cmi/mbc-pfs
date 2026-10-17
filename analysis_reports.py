import argparse
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
import os
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mbc_analysis_plots")

ANALYSES = [
    ('reconstruction', False),
    ('regimen_pfs', False),
    ('progression_definition', False),
    ('repeated_lines', True),
    ('clinical_comparator', True),
    ('care_process', True),
    ('calibration', False),
    ('cohort_selection.imaging_eligibility', True),
    ('treatment_window', True),
    ('ablation_inference', False),
    ('clinical_utility', False),
    ('model_diagnostics', False),
    ('site_count', True),
]

METHODS = """
METHODS USED THROUGHOUT (stated once here; the per-analysis sections below give only the tables)

Cohorts. MSK-CHORD development cohort: 8,791 modeled lines from 2,881 patients (first captured
lines, mLoT1: 2,342; later lines: 6,449 from 2,149 patients). Endpoint: first MSK-CHORD NLP
progression report at least 28 days after line start, or death; censored at the last
non-progression report; lines with PFS below 28 days excluded; administrative censoring at 730 days.

GENIE BPC. Regimens are curator-abstracted drug groupings (a new regimen after
a treatment break of 8 weeks or more); they are not clinical lines of therapy. BPC PFS-I + Mixed is
derived from the curated imaging assessments: first Progressing/Worsening/Enlarging or Mixed
assessment after regimen start, or death; censored at the next regimen start or last contact; the
28-day minimum and the 730-day cap are reapplied in the external validation cohorts, whereas the
regimen-level BPC Kaplan-Meier medians (Extended Data Table 20) use uncapped times
without the 28-day minimum. External validation cohorts: DFCI 1,394 regimens
from 317 patients; VICC 695 regimens from 152 patients. The reconstruction-agreement cohorts (our
algorithm run on BPC records) are separate: DFCI 749 episodes from 270 patients; VICC 381 from 130;
MSK 1,040 from 295.

Estimates. "Pooled" metrics are computed once on all held-out (out-of-fold) predictions. "Mean +/- SD"
values average the five outer test folds; the SD across folds is descriptive, not a confidence
interval. External metrics apply the five fixed fold models to the external cohort and average them.

Confidence intervals. Unless stated otherwise, 95% intervals are percentile intervals from 1,000
patient-level bootstrap resamples (seed 42): each sampled patient contributes all of their lines,
saved predictions are held fixed, and no model is refitted. Paired comparisons (model A minus
model B, or cohort variants) use the same 1,000 patient draws for both arms, so the interval is for
the paired difference. IBS intervals recalculate the censoring Kaplan-Meier estimate within each
draw. These intervals reflect evaluation-sample uncertainty only, not model-training uncertainty.

Calibration. Evaluated at 180, 365 and 730 days with inverse-probability-of-censoring weights.
CITL: logistic intercept on the log-odds scale with the slope fixed at 1 (positive = average
under-estimation of progression risk). Slope: logistic calibration slope with a free intercept
(ideal 1). ICI, E50, E90: mean, median and 90th percentile of the absolute difference between a
three-knot natural cubic spline calibration curve and the predicted risk, in percentage points.
Internal probabilities are out-of-fold; external probabilities are the mean of the five fixed fold
models.

Clinical reference model. Cox model (alpha = 1) with the first-line indicator, HR, HER2, latest
ECOG, liver involvement in the prior 90 days, and latest log1p CA15-3; trained on the same outer
folds with training-fold preprocessing.

Fixed-model sensitivities (180-day imaging window, day-28 landmark, endpoint redefinitions,
midpoint imputation) re-score the saved GBSA predictions on the modified cohort or labels without
refitting.
"""


def run_bootstrap():
    from analyses.common.ibs_bootstrap import main as paired_ibs
    from analyses.common.primary_bootstrap import calculate as primary_metrics
    from analyses.ablation_inference import calculate_metrics

    with redirect_stdout(sys.stderr):
        paired_ibs()
        primary_metrics()
        calculate_metrics()


def main(bootstrap=False):
    os.chdir(Path(__file__).resolve().parent)
    if bootstrap:
        run_bootstrap()
    parts = ["ANALYSIS NUMBERS", "Run: python analysis_reports.py", METHODS]
    context = None
    report_dir = Path("results/analysis_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, uses_predictions in ANALYSES:
        print(f"Calculate {name}", file=sys.stderr, flush=True)
        with redirect_stdout(sys.stderr):
            module = import_module("analyses." + name)
            if uses_predictions and context is None:
                from analyses.common.survival import SurvivalContext
                context = SurvivalContext()
            report = module.generate(context) if uses_predictions else module.generate()
        report = report.rstrip() + "\n"
        (report_dir / f"{name}.txt").write_text(report)
        parts.extend(["\n" + "=" * 80, name, "=" * 80, report])
    primary = Path("results/rebuttal/bootstrap/primary_metrics.csv")
    if primary.exists():
        import pandas as pd
        if len(pd.read_csv(primary)) == 25:
            from tabulate import tabulate
            from analyses.common.primary_bootstrap import summary
            table = tabulate(summary(), headers="keys", showindex=False, tablefmt="psql", disable_numparse=True)
            parts.extend(["\n" + "=" * 80, "Primary model intervals: 1,000 patient resamples", "=" * 80, table])
    output = "\n".join(parts) + "\n"
    Path("analysis_data.txt").write_text(output)
    print(output, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true",
                        help="Rebuild the added intervals with 1,000 patient resamples from saved predictions.")
    main(bootstrap=parser.parse_args().bootstrap)
