import argparse

import numpy as np
import pandas as pd
from manuscript_utils import BASE_MODELS, DEFAULT_MODEL_NAME_MAP, overall_performance_metric_rows
from analyses.common.survival import ci, ibs_evaluator, load_curves, patient_bootstrap, patient_draws
from analyses.common.ibs_bootstrap import OUTPUT, bootstrap_ibs

METRICS = ["C", "mean_auc", "auc_at_1y", "auc_at_2y", "IBS"]


def metrics():
    return pd.read_csv(OUTPUT / "primary_metrics.csv").set_index(["model", "metric"])


def summary():
    table = metrics()
    cell = "{0.point:.3f} ({0.low:.3f}, {0.high:.3f})".format
    return pd.DataFrame([{"Metric": label, **{DEFAULT_MODEL_NAME_MAP[model]: cell(table.loc[(model, key)])
                                              for model in BASE_MODELS}}
                         for label, key, _ in overall_performance_metric_rows()])


def calculate(force=False):
    from analyses.common.auc_bootstrap import bootstrap_auc

    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample_path = OUTPUT / "primary_metrics_samples.csv"
    summary_path = OUTPUT / "primary_metrics.csv"
    old_samples = pd.read_csv(sample_path) if sample_path.exists() and not force else pd.DataFrame()
    old_summary = pd.read_csv(summary_path) if summary_path.exists() and not force else pd.DataFrame()
    dm = pd.read_csv("data/design_matrix.csv", usecols=["PATIENT_ID", "PFS_TIME_DAYS", "PFS_EVENT"])
    _, ids, weights = patient_draws(dm)
    ibs_cache_path = OUTPUT / "IBS_samples.csv"
    ibs_cache = pd.read_csv(ibs_cache_path) if ibs_cache_path.exists() else pd.DataFrame()
    denominator = None
    samples, summary = [], []
    for model in BASE_MODELS:
        if not old_samples.empty and not old_summary.empty:
            cached_samples = old_samples[old_samples.model.eq(model)]
            cached_summary = old_summary[old_summary.model.eq(model)]
            if len(cached_samples) == 1000 and len(cached_summary) == len(METRICS):
                samples.append(cached_samples)
                summary.append(cached_summary)
                print(f"Reuse {model}: 1,000 completed patient draws.", flush=True)
                continue
        sf, grid = load_curves(dm, model, metric_grid=True)
        curves = sf.to_numpy()
        concordance = patient_bootstrap(curves, grid, dm, weights=weights, den=denominator)
        denominator = concordance["den"]
        name = f"full_{model}"
        if name in ibs_cache and len(ibs_cache) == 1000 and model != "deephit":
            ibs_samples = ibs_cache[name].to_numpy()
            ibs_point = float(ibs_evaluator(curves, grid, dm).integrated_brier_score())
        else:
            ibs_point, ibs_samples = bootstrap_ibs(curves, grid, dm, weights)
        auc_point, auc_samples = bootstrap_auc(curves, grid, dm, weights)
        values = pd.DataFrame(dict(model=model, draw=np.arange(1000), C=concordance["samples"],
                                   mean_auc=auc_samples[:, 0], auc_at_1y=auc_samples[:, 1],
                                   auc_at_2y=auc_samples[:, 2], IBS=ibs_samples))
        points = dict(C=concordance["C"], IBS=ibs_point, **auc_point)
        rows = []
        for metric in METRICS:
            low, high = ci(values[metric])
            rows.append(dict(model=model, metric=metric, point=points[metric], low=low, high=high,
                             n_boot=1000, seed=42, lines=len(dm), patients=len(ids)))
        samples.append(values)
        summary.append(pd.DataFrame(rows))
        pd.concat(samples, ignore_index=True).to_csv(sample_path, index=False)
        pd.concat(summary, ignore_index=True).to_csv(summary_path, index=False)
        print(f"{model}: completed 1,000 patient draws for C, mean AUC, AUC1y, AUC2y, IBS.", flush=True)
    result = pd.concat(summary, ignore_index=True)
    result.to_csv(summary_path, index=False)
    pd.concat(samples, ignore_index=True).to_csv(sample_path, index=False)
    print(result.to_string(index=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    calculate(force=parser.parse_args().force)
