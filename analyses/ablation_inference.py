import argparse
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from tabulate import tabulate
from manuscript_utils import BASE_MODELS
from analyses.common.primary_bootstrap import OUTPUT as BOOTSTRAP, metrics
from analyses.common.survival import ci, load_curves, patient_bootstrap, patient_draws

PAIRED_C = Path("results/rebuttal/ablation/paired_C.csv")
INTERVALS = BOOTSTRAP / "treatment_masking_intervals.csv"


def summarize_metric_intervals():
    primary = metrics()
    masked = pd.read_csv(BOOTSTRAP / "primary_metrics_samples.csv")
    unmasked = pd.read_csv(BOOTSTRAP / "treatment_masking_samples.csv")
    points = pd.read_csv(BOOTSTRAP / "treatment_masking_points.csv").set_index("model")
    rows = []
    for model in BASE_MODELS:
        first = masked.loc[masked.model.eq(model)].set_index("draw")
        second = unmasked.loc[unmasked.model.eq(model)].set_index("draw")
        for metric in ("IBS", "mean_auc"):
            a = float(primary.loc[(model, metric), "point"])
            b = float(points.loc[model, metric])
            delta = second[metric] - first[metric]
            lo, hi = ci(delta)
            rows.append(dict(model=model, metric=metric, masked_point=a, unmasked_point=b,
                delta_unmasked_minus_masked=b-a, CI_low=lo, CI_high=hi,
                n_bootstrap=len(delta), seed=42))
    result = pd.DataFrame(rows)
    result.to_csv(INTERVALS, index=False)
    print(result.to_string(index=False), flush=True)


def calculate_metrics():
    from analyses.common.auc_bootstrap import bootstrap_auc
    from analyses.common.ibs_bootstrap import bootstrap_ibs

    dm = pd.read_csv("data/design_matrix.csv")
    _, _, weights = patient_draws(dm)
    BOOTSTRAP.mkdir(parents=True, exist_ok=True)
    samples, points = [], []
    for model in BASE_MODELS:
        started = monotonic()
        sf, grid = load_curves(dm, model, folder="leakage", metric_grid=True)
        ibs, ibs_samples = bootstrap_ibs(sf.to_numpy(), grid, dm, weights)
        auc, auc_samples = bootstrap_auc(sf.to_numpy(), grid, dm, weights)
        samples.append(pd.DataFrame(dict(model=model, draw=np.arange(1000), IBS=ibs_samples,
                                         mean_auc=auc_samples[:, 0])))
        points.append(dict(model=model, IBS=ibs, mean_auc=auc["mean_auc"]))
        pd.concat(samples, ignore_index=True).to_csv(BOOTSTRAP / "treatment_masking_samples.csv", index=False)
        pd.DataFrame(points).to_csv(BOOTSTRAP / "treatment_masking_points.csv", index=False)
        print(f"{model}: 1,000 fixed-model patient draws in {monotonic()-started:.1f} seconds.", flush=True)
    primary_paths = [BOOTSTRAP / name for name in ("primary_metrics_samples.csv", "primary_metrics.csv")]
    if all(path.exists() and set(pd.read_csv(path, usecols=["model"]).model) == set(BASE_MODELS)
           for path in primary_paths):
        summarize_metric_intervals()
    else:
        print("Masked primary bootstrap files are pending. Use --summarize-metrics after they are complete.", flush=True)


def main():
    start = monotonic()
    PAIRED_C.parent.mkdir(parents=True, exist_ok=True)
    dm = pd.read_csv('data/design_matrix.csv')
    _, ids, weights = patient_draws(dm)
    denominator = None
    rows = []
    for model in BASE_MODELS:
        pair = []
        for folder in ['results', 'leakage']:
            sf, grid = load_curves(dm, model, folder=folder, metric_grid=True)
            result = patient_bootstrap(sf.to_numpy(), grid, dm, weights=weights, den=denominator)
            denominator = result["den"]
            pair.append(result)
        lo, hi = ci(pair[1]["samples"] - pair[0]["samples"])
        rows.append(dict(model=model, C_masked_pooled_OOF=pair[0]["C"],
                         C_unmasked_pooled_OOF=pair[1]["C"], delta_C_unmasked_minus_masked=pair[1]["C"]-pair[0]["C"],
                         CI_low=lo, CI_high=hi, n_bootstrap=1000, seed=42,
                         n_patients=len(ids), n_lines=len(dm)))
        pd.DataFrame(rows).to_csv(PAIRED_C, index=False)
        print(rows[-1], flush=True)
    print(f'Completed all five pairs in {monotonic() - start:.1f} seconds.', flush=True)


def mean_fold_effects():
    from manuscript_utils import ManuscriptContext
    ctx = ManuscriptContext()
    return {"Treatment unmasked minus masked": ctx.leakage_deltas,
            "Age removed minus full": ctx.age_ablation_deltas}


def generate(context=None):
    parts = ["Table C10a. Unmasked minus masked treatment: pooled held-out C.",
             "Use 1,000 shared patient resamples. Keep all lines for each sampled patient.",
             "Keep the five saved models fixed. These intervals exclude training uncertainty.",
             tabulate(pd.read_csv(PAIRED_C), headers="keys", showindex=False, tablefmt="psql", floatfmt=".5f")]
    for label, table in mean_fold_effects().items():
        rows = [{"Model": r.model, "Change in C": r.delta_C, "Change in IBS": r.delta_IBS, "Change in mean_auc": r.delta_AUC}
                for r in table.itertuples()]
        parts.extend(["Table C10b. " + label + ": descriptive mean-fold effects.",
                      tabulate(rows, headers="keys", tablefmt="psql", floatfmt="+.6f")])
    if INTERVALS.exists():
        parts.extend(["Table C10c. Unmasked minus masked treatment: paired pooled IBS and mean AUC.",
                      "Use the same 1,000 patient draws as the primary-model intervals. Keep models fixed.",
                      "Lower IBS and higher mean AUC indicate better performance. These values differ from mean-fold effects.",
                      tabulate(pd.read_csv(INTERVALS), headers="keys", showindex=False, tablefmt="psql", floatfmt=".6f"),
                      "Recalculate: python -m analyses.ablation_inference --metrics"])
    parts.extend(["Five-fold means differ from pooled C. No p-value is reported.",
                  "Individual age-ablation predictions are absent. No paired age interval is available.",
                  "Recalculate pooled effects: python -m analyses.ablation_inference",
                  "Inputs: data/design_matrix.csv; results and leakage survival predictions; saved fold metrics."])
    return "\n\n".join(parts) + "\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="store_true", help="Calculate 1,000 IBS and mean-AUC samples; retain completed C intervals.")
    parser.add_argument("--summarize-metrics", action="store_true", help="Pair saved masked and unmasked bootstrap samples.")
    args = parser.parse_args()
    if args.metrics:
        calculate_metrics()
    elif args.summarize_metrics:
        summarize_metric_intervals()
    else:
        main()
