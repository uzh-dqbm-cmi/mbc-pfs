from __future__ import annotations

import argparse
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd

from config import AUC, C_INDEX, IBS
from cv_config import C_SELECTION_EPS, RANDOM_STATE
from deephit import train_deephit
from deepsurv import train_deepsurv
from dryrun import train_dryrun
from nested_cv_splits import (
    add_run_args,
    model_configs,
    prepare_run,
    select_best_config,
    write_metrics,
)
from utils import save_eval_metrics

MODEL_RUNNERS = {
    "deepsurv": train_deepsurv,
    "deephit": train_deephit,
    "dryrun": train_dryrun,
}


def _outer_split(manifest: Dict[str, object], outer_num: int):
    split_path = manifest[f"outer{outer_num}_inner0"]["path"]
    split = pd.read_csv(split_path)
    if "outer_es" not in split.columns:
        raise ValueError(f"Split file {split_path} missing `outer_es` column.")
    non_test_mask = split["split"] != "test"
    outer_es_mask = split["outer_es"].astype(bool)
    train_idx = split.loc[non_test_mask & ~outer_es_mask, "idx"].to_numpy(dtype=int)
    early_stop_idx = split.loc[non_test_mask & outer_es_mask, "idx"].to_numpy(dtype=int)
    test_idx = split[split["split"] == "test"]["idx"].to_numpy(dtype=int)
    if early_stop_idx.size == 0:
        raise ValueError(f"Split file {split_path} produced empty `outer_es` set.")
    if test_idx.size == 0:
        raise ValueError(f"Split file {split_path} produced empty outer `test` set.")
    return train_idx, early_stop_idx, test_idx


def main():
    parser = argparse.ArgumentParser(
        description="Nested CV orchestrator that calls in-process trainers."
    )
    add_run_args(parser, default_models="deepsurv")
    args = parser.parse_args()
    ignore_prefix = "PLANNED_" if args.no_treatment_masking else None
    dm, manifest, results_root, ablation_configs = prepare_run(args)
    run_ablation = ablation_configs is not None

    for raw_model in args.models.split(","):
        try:
            model = raw_model.strip()
            configs_dict = model_configs(model, ablation_configs)
            model_metrics: Dict[str, object] = {
                "model": model,
                "outer_folds": {},
            }
            if run_ablation:
                config_name, config = next(iter(configs_dict.items()))
                for outer_num in range(manifest["k_outer"]):
                    print(f"OUTER FOLD {outer_num}: {model}--{config_name}")
                    outer_key = f"outer{outer_num}"
                    outer_entry: Dict[str, object] = {
                        "outer": outer_num,
                        "inner_skipped": True,
                    }
                    train_idx, early_stop_idx, test_idx = _outer_split(manifest, outer_num)
                    final_config = config.copy()
                    final_config["outer_fold"] = outer_num
                    final_config["ignore_prefix"] = ignore_prefix
                    outer_outdir = results_root / model / f"outer{outer_num}"
                    try:
                        final_metrics = MODEL_RUNNERS[model](
                            train_idx=train_idx,
                            val_idx=early_stop_idx,
                            test_idx=test_idx,
                            design_matrix=dm,
                            config=final_config,
                            outer=True,
                            outdir=outer_outdir / config_name,
                            run_ablation=run_ablation,
                        )
                        print(f"  FINAL EVAL {C_INDEX}: {final_metrics.get(C_INDEX)}")
                        print(f"  FINAL EVAL {AUC}: {final_metrics.get(AUC)}")
                        print(f"  FINAL EVAL {IBS}: {final_metrics.get(IBS)}")
                    except Exception as exc:
                        print(f"[ERROR] {model} outer{outer_num} final: {exc}")
                        traceback.print_exc()
                        final_metrics = {"error": str(exc)}
                    best_model_record = {
                        "split_path": manifest[f"outer{outer_num}_inner0"]["path"],
                        "results_path": str(outer_outdir / config_name),
                        "config": config_name,
                        "metrics": final_metrics,
                        "name": config_name,
                    }
                    outer_entry["best_model"] = best_model_record
                    model_metrics["outer_folds"][outer_key] = best_model_record
                    write_metrics(outer_outdir / "fold_metrics.json", outer_entry)
                write_metrics(results_root / model / "eval_metrics.json", model_metrics)
                print(f"Wrote metrics to {results_root / model / 'eval_metrics.json'}")
                continue

            for outer_num in range(manifest["k_outer"]):
                print(f"OUTER FOLD {outer_num}: {model}")
                outer_key = f"outer{outer_num}"
                outer_entry: Dict[str, object] = {"outer": outer_num}
                model_metrics["outer_folds"][
                    outer_key
                ] = None  # placeholder; fill after best_model
                config_avgs: Dict[str, Dict[str, float]] = {}
                inner_data: Dict[int, Dict[str, object]] = {
                    i: {} for i in range(manifest["k_inner"])
                }

                for config_name, config in configs_dict.items():
                    c_vals: List[float] = []
                    auc_vals: List[float] = []
                    ibs_vals: List[float] = []
                    for inner_num in range(manifest["k_inner"]):
                        fold_label = f"outer{outer_num}_inner{inner_num}"
                        split = pd.read_csv(manifest[fold_label]["path"])
                        train_mask = split["split"] == "train"
                        inner_es_mask = split["inner_es"].astype(bool)
                        train_idx = split.loc[
                            train_mask & ~inner_es_mask, "idx"
                        ].to_numpy()
                        early_stop_idx = split.loc[
                            train_mask & inner_es_mask, "idx"
                        ].to_numpy()
                        eval_idx = split.loc[split["split"] == "val", "idx"].to_numpy()
                        print(f"  INNER FOLD {inner_num}: {model}--{config_name}")
                        config_ctx = dict(config)
                        config_ctx.setdefault("seed", RANDOM_STATE)
                        config_ctx["outer_fold"] = outer_num
                        config_ctx["inner_fold"] = inner_num
                        config_ctx["ignore_prefix"] = ignore_prefix
                        inner_outdir = (
                            results_root
                            / model
                            / f"outer{outer_num}"
                            / f"inner{inner_num}"
                            / config_name
                        )
                        inner_outdir_str = str(inner_outdir)
                        if early_stop_idx.size == 0:
                            raise ValueError(
                                f"Split file {manifest[fold_label]['path']} produced empty `inner_es` set."
                            )
                        if eval_idx.size == 0:
                            raise ValueError(
                                f"Split file {manifest[fold_label]['path']} produced empty inner `val` set."
                            )
                        try:
                            metrics = MODEL_RUNNERS[model](
                                train_idx=train_idx.astype(int),
                                val_idx=early_stop_idx.astype(int),
                                test_idx=eval_idx.astype(
                                    int
                                ),  # report inner performance on held-out eval fold
                                design_matrix=dm,
                                config=config_ctx,
                                outer=False,
                                outdir=inner_outdir,
                                run_ablation=run_ablation,
                            )
                        except Exception as exc:
                            print(f"[ERROR] {model} {config_name} {fold_label}: {exc}")
                            traceback.print_exc()
                            metrics = {"error": str(exc)}
                            save_eval_metrics(
                                outdir=inner_outdir,
                                metrics=metrics,
                            )
                        inner_data[inner_num][config_name] = {
                            "results_path": inner_outdir_str,
                            "metrics": metrics,
                        }
                        c_vals.append(float(metrics[C_INDEX]))
                        auc_vals.append(float(metrics[AUC]))
                        ibs_vals.append(float(metrics[IBS]))
                        print(
                            f"    {C_INDEX}: {metrics.get(C_INDEX)}, {AUC}: {metrics.get(AUC)}, {IBS}: {metrics.get(IBS)}\n"
                        )

                    config_avgs[config_name] = {
                        C_INDEX: np.mean(c_vals),
                        AUC: np.mean(auc_vals),
                        IBS: np.mean(ibs_vals),
                    }

                # select the best model, fit on the outer fold. Save models, metrics, surv outputs, and SHAP values.
                rank_of = {
                    name: cfg["internal_ranking"] for name, cfg in configs_dict.items()
                }
                best_config_name, best_c, c_tie_names = select_best_config(
                    config_avgs,
                    rank_of,
                )
                print(
                    "  Tie-band selection:"
                    f" best_C={best_c:.6f},"
                    f" eps={C_SELECTION_EPS:.3f},"
                    f" candidates={c_tie_names}"
                )
                # use the inner0 split to get train/test indices for final eval from outer fold
                train_idx, early_stop_idx, test_idx = _outer_split(manifest, outer_num)
                final_config = configs_dict[best_config_name].copy()
                final_config["outer_fold"] = outer_num
                final_config["ignore_prefix"] = ignore_prefix

                print(f"  OUTER FOLD {outer_num} BEST CONFIG: {best_config_name}")
                print("  Average scores:")
                print(f"    {C_INDEX}: {config_avgs[best_config_name][C_INDEX]}")
                print(f"    {AUC}: {config_avgs[best_config_name][AUC]}")
                print(f"    {IBS}: {config_avgs[best_config_name][IBS]}")
                outer_outdir = results_root / model / f"outer{outer_num}"
                try:
                    final_metrics = MODEL_RUNNERS[model](
                        train_idx=train_idx,
                        val_idx=early_stop_idx,
                        test_idx=test_idx,
                        design_matrix=dm,
                        config=final_config,
                        outer=True,
                        outdir=outer_outdir / best_config_name,
                        run_ablation=run_ablation,
                    )
                    print(f"  FINAL EVAL {C_INDEX}: {final_metrics.get(C_INDEX)}")
                    print(f"  FINAL EVAL {AUC}: {final_metrics.get(AUC)}")
                    print(f"  FINAL EVAL {IBS}: {final_metrics.get(IBS)}")
                except Exception as exc:
                    print(f"[ERROR] {model} outer{outer_num} final: {exc}")
                    traceback.print_exc()
                    final_metrics = {"error": str(exc)}
                best_model_record = {
                    "split_path": manifest[f"outer{outer_num}_inner0"]["path"],
                    "results_path": str(outer_outdir / best_config_name),
                    "config": best_config_name,
                    "metrics": final_metrics,
                }
                for inner_num in range(manifest["k_inner"]):
                    outer_entry[f"inner_{inner_num}"] = inner_data[inner_num]
                for cfg_name, avg in config_avgs.items():
                    outer_entry[f"{cfg_name}_avg"] = avg
                best_model_record["name"] = best_config_name
                outer_entry["best_model"] = best_model_record
                model_metrics["outer_folds"][outer_key] = best_model_record
                write_metrics(outer_outdir / "fold_metrics.json", outer_entry)

            write_metrics(results_root / model / "eval_metrics.json", model_metrics)
            print(f"Wrote metrics to {results_root / model / 'eval_metrics.json'}")
        except Exception as exc:
            print(f"[ERROR] {model} overall: {exc}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
