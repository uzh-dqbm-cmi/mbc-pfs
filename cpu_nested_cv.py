from __future__ import annotations

import os

# Prevent OpenMP/BLAS oversubscription and fork-safety issues when using multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd

from config import AUC, C_INDEX, IBS
from coxph import train_coxph
from cv_config import C_SELECTION_EPS, RANDOM_STATE
from dryrun import train_dryrun
from gbsa import train_gbsa
from nested_cv_splits import (
    add_run_args,
    model_configs,
    prepare_run,
    select_best_config,
    write_metrics,
)
from rsf import train_rsf

MODEL_RUNNERS = {
    "coxph": train_coxph,
    "gbsa": train_gbsa,
    "rsf": train_rsf,
    "dryrun": train_dryrun,
}
PARALLEL_MODELS = {"coxph", "gbsa"}
CONFIGS_PER_BATCH = 2
MAX_WORKERS = 30


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main():
    mp_context = mp.get_context("spawn")

    parser = argparse.ArgumentParser(
        description="Nested CV orchestrator that calls in-process trainers."
    )
    add_run_args(parser, default_models="coxph,gbsa,rsf")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Max process pool workers for parallel models.",
    )
    args = parser.parse_args()
    ignore_prefix = "PLANNED_" if args.no_treatment_masking else None
    dm, manifest, results_root, ablation_configs = prepare_run(args)
    run_ablation = ablation_configs is not None
    k_outer = manifest["k_outer"]
    k_inner = manifest["k_inner"]

    split_cache: Dict[tuple[int, int], Dict[str, object]] = {}
    outer_train_val_indices: Dict[int, np.ndarray] = {}
    outer_test_indices: Dict[int, np.ndarray] = {}
    for outer_num in range(k_outer):
        for inner_num in range(k_inner):
            fold_label = f"outer{outer_num}_inner{inner_num}"
            split = pd.read_csv(manifest[fold_label]["path"])
            train_idx = split[split["split"] == "train"]["idx"].to_numpy()
            val_idx = split[split["split"] == "val"]["idx"].to_numpy()
            split_cache[(outer_num, inner_num)] = {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "split_path": manifest[fold_label]["path"],
            }
            if inner_num == 0:
                outer_train_val_indices[outer_num] = split[split["split"] != "test"][
                    "idx"
                ].to_numpy()
                outer_test_indices[outer_num] = split[split["split"] == "test"][
                    "idx"
                ].to_numpy()

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
                for outer_num in range(k_outer):
                    print(f"OUTER FOLD {outer_num}: {model}--{config_name}")
                    outer_key = f"outer{outer_num}"
                    outer_entry: Dict[str, object] = {
                        "outer": outer_num,
                        "inner_skipped": True,
                    }
                    train_idx = outer_train_val_indices[outer_num]
                    test_idx = outer_test_indices[outer_num]
                    final_config = config.copy()
                    final_config.setdefault("seed", RANDOM_STATE)
                    final_config["outer_fold"] = outer_num
                    final_config["ignore_prefix"] = ignore_prefix
                    outer_outdir = results_root / model / f"outer{outer_num}"
                    try:
                        final_metrics = MODEL_RUNNERS[model](
                            train_idx=train_idx,
                            val_idx=test_idx,
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
                        "name": config_name,
                        "split_path": manifest[f"outer{outer_num}_inner0"]["path"],
                        "results_path": str(outer_outdir / config_name),
                        "metrics": final_metrics,
                    }
                    outer_entry["best_model"] = best_model_record
                    model_metrics["outer_folds"][outer_key] = best_model_record
                    fold_metrics_path = (
                        results_root / model / f"outer{outer_num}" / "fold_metrics.json"
                    )
                    write_metrics(fold_metrics_path, outer_entry)
                final_metrics_path = results_root / model / "eval_metrics.json"
                write_metrics(final_metrics_path, model_metrics)
                print(f"Wrote metrics to {final_metrics_path}")
                continue
            inner_data: Dict[int, Dict[int, Dict[str, object]]] = {
                outer_num: {inner_num: {} for inner_num in range(k_inner)}
                for outer_num in range(k_outer)
            }
            config_avgs: Dict[int, Dict[str, Dict[str, float]]] = {
                outer_num: {} for outer_num in range(k_outer)
            }

            if model in PARALLEL_MODELS:
                config_scores: Dict[int, Dict[str, Dict[str, List[float]]]] = {
                    outer_num: {
                        cfg_name: {C_INDEX: [], AUC: [], IBS: []}
                        for cfg_name in configs_dict
                    }
                    for outer_num in range(k_outer)
                }
                config_items = list(configs_dict.items())

                for config_batch in _chunked(config_items, CONFIGS_PER_BATCH):
                    jobs = {}
                    with ProcessPoolExecutor(
                        max_workers=args.max_workers, mp_context=mp_context
                    ) as executor:
                        for config_name, config in config_batch:
                            for outer_num in range(k_outer):
                                for inner_num in range(k_inner):
                                    split_info = split_cache[(outer_num, inner_num)]
                                    print(
                                        f"  OUTER {outer_num} INNER {inner_num}: {model}--{config_name}"
                                    )
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
                                    jobs[
                                        executor.submit(
                                            MODEL_RUNNERS[model],
                                            train_idx=split_info["train_idx"],
                                            val_idx=split_info["val_idx"],
                                            test_idx=split_info["val_idx"],
                                            design_matrix=dm,
                                            config=config_ctx,
                                            outer=False,
                                            outdir=inner_outdir,
                                            run_ablation=run_ablation,
                                        )
                                    ] = (
                                        outer_num,
                                        inner_num,
                                        config_name,
                                        split_info["split_path"],
                                        str(inner_outdir),
                                    )
                    for future in as_completed(jobs):
                        outer_num, inner_num, config_name, split_path, outdir_str = (
                            jobs[future]
                        )
                        try:
                            metrics = future.result()
                        except Exception as exc:
                            print(
                                f"[ERROR] {model} {config_name} outer{outer_num}_inner{inner_num}: {exc}"
                            )
                            traceback.print_exc()
                            metrics = {"error": str(exc)}
                        inner_data[outer_num][inner_num][config_name] = {
                            "results_path": outdir_str,
                            "metrics": metrics,
                            "split_path": split_path,
                        }
                        config_scores[outer_num][config_name][C_INDEX].append(
                            float(metrics[C_INDEX])
                        )
                        config_scores[outer_num][config_name][AUC].append(
                            float(metrics[AUC])
                        )
                        config_scores[outer_num][config_name][IBS].append(
                            float(metrics[IBS])
                        )
                        print(
                            f"    {C_INDEX}: {metrics.get(C_INDEX)}, {AUC}: {metrics.get(AUC)}, {IBS}: {metrics.get(IBS)}\n"
                        )
                for outer_num in range(k_outer):
                    for cfg_name, score_lists in config_scores[outer_num].items():
                        config_avgs[outer_num][cfg_name] = {
                            C_INDEX: np.mean(score_lists[C_INDEX]),
                            AUC: np.mean(score_lists[AUC]),
                            IBS: np.mean(score_lists[IBS]),
                        }

            for outer_num in range(k_outer):
                print(f"OUTER FOLD {outer_num}: {model}")
                outer_key = f"outer{outer_num}"
                outer_entry: Dict[str, object] = {"outer": outer_num}
                model_metrics["outer_folds"][outer_key] = None
                if model not in PARALLEL_MODELS:
                    for config_name, config in configs_dict.items():
                        c_vals: List[float] = []
                        auc_vals: List[float] = []
                        ibs_vals: List[float] = []
                        for inner_num in range(k_inner):
                            fold_label = f"outer{outer_num}_inner{inner_num}"
                            split_info = split_cache[(outer_num, inner_num)]
                            print(
                                f"    INNER: {model.upper()}--{config_name.upper()}--{fold_label.upper()}"
                            )
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
                            try:
                                metrics = MODEL_RUNNERS[model](
                                    train_idx=split_info["train_idx"],
                                    val_idx=split_info["val_idx"],
                                    test_idx=split_info["val_idx"],
                                    design_matrix=dm,
                                    config=config_ctx,
                                    outer=False,
                                    outdir=inner_outdir,
                                    run_ablation=run_ablation,
                                )
                            except Exception as exc:
                                print(
                                    f"[ERROR] {model} {config_name} {fold_label}: {exc}"
                                )
                                traceback.print_exc()
                                metrics = {"error": str(exc)}
                            inner_data[outer_num][inner_num][config_name] = {
                                "results_path": str(inner_outdir),
                                "metrics": metrics,
                                "split_path": split_info["split_path"],
                            }
                            c_vals.append(float(metrics[C_INDEX]))
                            auc_vals.append(float(metrics[AUC]))
                            ibs_vals.append(float(metrics[IBS]))
                            print(
                                f"    {C_INDEX}: {metrics.get(C_INDEX)}, {AUC}: {metrics.get(AUC)}, {IBS}: {metrics.get(IBS)}\n"
                            )

                        config_avgs[outer_num][config_name] = {
                            C_INDEX: np.mean(c_vals),
                            AUC: np.mean(auc_vals),
                            IBS: np.mean(ibs_vals),
                        }

                rank_of = {
                    name: cfg["internal_ranking"] for name, cfg in configs_dict.items()
                }
                print(f"{rank_of=}")
                print(f"{config_avgs[outer_num]=}")
                best_config_name, best_c, c_tie_names = select_best_config(
                    config_avgs[outer_num],
                    rank_of,
                )
                print(
                    "  Tie-band selection:"
                    f" best_C={best_c:.6f},"
                    f" eps={C_SELECTION_EPS:.3f},"
                    f" candidates={c_tie_names}"
                )
                train_idx = outer_train_val_indices[outer_num]
                test_idx = outer_test_indices[outer_num]
                final_config = configs_dict[best_config_name].copy()
                final_config.setdefault("seed", RANDOM_STATE)
                final_config["outer_fold"] = outer_num
                final_config["ignore_prefix"] = ignore_prefix
                print(f"  OUTER FOLD {outer_num} BEST CONFIG: {best_config_name}")
                print("  Average scores:")
                print(
                    f"    {C_INDEX}: {config_avgs[outer_num][best_config_name][C_INDEX]}"
                )
                print(f"    {AUC}: {config_avgs[outer_num][best_config_name][AUC]}")
                print(f"    {IBS}: {config_avgs[outer_num][best_config_name][IBS]}")

                outer_outdir = results_root / model / f"outer{outer_num}"
                try:
                    final_metrics = MODEL_RUNNERS[model](
                        train_idx=train_idx,
                        val_idx=test_idx,
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

                for inner_num in range(k_inner):
                    outer_entry[f"inner_{inner_num}"] = inner_data[outer_num][inner_num]
                for cfg_name, avg in config_avgs[outer_num].items():
                    outer_entry[f"{cfg_name}_avg"] = avg
                best_model_record = {
                    "name": best_config_name,
                    "split_path": manifest[f"outer{outer_num}_inner0"]["path"],
                    "results_path": str(outer_outdir / best_config_name),
                    "metrics": final_metrics,
                }
                outer_entry["best_model"] = best_model_record
                model_metrics["outer_folds"][outer_key] = best_model_record

                fold_metrics_path = (
                    results_root / model / f"outer{outer_num}" / "fold_metrics.json"
                )
                write_metrics(fold_metrics_path, outer_entry)

            final_metrics_path = results_root / model / "eval_metrics.json"
            write_metrics(final_metrics_path, model_metrics)
            print(f"Wrote metrics to {final_metrics_path}")
        except Exception as exc:
            print(f"[ERROR] {model} overall: {exc}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
