from __future__ import annotations

import os

# Prevent OpenMP/BLAS oversubscription and fork-safety issues when using multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    AUC,
    C_INDEX,
    DESIGN_MATRIX_PATH,
    EVENT_COL,
    FEATURES_DICT_PATH,
    IBS,
    LINE_COL,
    MODALITY_GROUPS,
    PATIENT_ID_COL,
    RESULTS_PATH,
    TIME_COL,
)
from coxph import train_coxph
from cv_config import MODEL_CONFIGS, RANDOM_STATE, SPLITS_OUTPUT_DIR
from dryrun import train_dryrun
from gbsa import train_gbsa
from model_setup import EVENT_COL
from nested_cv_splits import generate_nested_cv_indices
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
DRY_RUN_METRIC = 0.5
# Treat configs within this C-index margin of the best score as ties.
C_SELECTION_EPS = 0.003


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _write_metrics_artifact(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fout:
        json.dump(payload, fout, indent=2)


def _select_best_config(
    config_scores: Dict[str, Dict[str, float]],
    rank_of: Dict[str, int],
) -> tuple[str, float, list[str]]:
    best_c = max(float(metrics[C_INDEX]) for metrics in config_scores.values())
    tied_names = sorted(
        (
            name
            for name, metrics in config_scores.items()
            if float(metrics[C_INDEX]) >= (best_c - C_SELECTION_EPS)
        ),
        key=lambda name: rank_of[name],
    )
    return tied_names[0], best_c, tied_names


def main():
    mp_context = mp.get_context("spawn")

    parser = argparse.ArgumentParser(
        description="Nested CV orchestrator that calls in-process trainers."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="coxph,gbsa,rsf",
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--exclude-modality",
        type=str,
        default=None,
        help="Name of modality group to exclude.",
    )
    parser.add_argument(
        "--only-modality",
        type=str,
        default=None,
        help="Name of modality group to use exclusively.",
    )
    parser.add_argument(
        "--drop-age",
        action="store_true",
        help="Drop AGE column from design matrix.",
    )
    parser.add_argument(
        "--drop-age-less-regularized",
        action="store_true",
        help=(
            "Drop AGE and use the less-regularized age ablation config/output path "
            "(gbsa-mid-400)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Max process pool workers for parallel models.",
    )
    parser.add_argument(
        "--no-treatment-masking",
        action="store_true",
        help="Disable treatment masking by dropping PLANNED_ instead of default prefixes.",
    )
    args = parser.parse_args()
    ignore_prefix = "PLANNED_" if args.no_treatment_masking else None

    dm = pd.read_csv(DESIGN_MATRIX_PATH)
    dm = dm[dm[EVENT_COL] != -1].reset_index(drop=True)

    manifest, _ = generate_nested_cv_indices(
        design_matrix=dm, seed=RANDOM_STATE, output_dir=Path(SPLITS_OUTPUT_DIR)
    )

    run_ablation = bool(
        args.exclude_modality
        or args.only_modality
        or args.drop_age
        or args.drop_age_less_regularized
    )
    if args.no_treatment_masking:
        results_root = Path("leakage")
    else:
        results_root = Path("results_ablation") if run_ablation else RESULTS_PATH
    ablation_cfg_names: Dict[str, List[str]] = {}
    ablation_cfg_path = Path("data/ablation_configs.json")
    if args.drop_age_less_regularized:
        ablation_cfg_path = Path("data/age_ablation_configs.json")
    if run_ablation:
        if not ablation_cfg_path.exists():
            raise FileNotFoundError(
                f"Missing ablation config file: {ablation_cfg_path}"
            )
        try:
            payload = json.load(open(ablation_cfg_path))
            ablation_cfg_names = {
                str(model): [str(name) for name in names]
                for model, names in payload.items()
                if isinstance(names, list)
            }
            print(f"Loaded ablation config names from {ablation_cfg_path}.")
        except Exception as exc:
            raise ValueError(f"Failed to load {ablation_cfg_path}: {exc}") from exc

    if run_ablation:
        FEATURES = json.load(open(FEATURES_DICT_PATH))
        # only one of exclude_modality, only_modality, drop_age variants should be set
        assert (
            sum(
                (
                    args.exclude_modality is not None,
                    args.only_modality is not None,
                    args.drop_age,
                    args.drop_age_less_regularized,
                )
            )
            <= 1
        ), (
            "Only one of --exclude-modality, --only-modality, "
            "--drop-age, --drop-age-less-regularized can be set."
        )

        if args.exclude_modality:
            if args.exclude_modality == "Surveillance Patterns":
                exclude_cols = [
                    c
                    for c in dm.columns
                    if c in MODALITY_GROUPS["Surveillance Patterns"]
                ]
            else:
                exclude_groups = MODALITY_GROUPS[args.exclude_modality]
                exclude_cols = sorted(
                    {
                        col
                        for group in exclude_groups
                        for col in FEATURES[group]
                        if col in dm.columns
                    }
                )
            dm = dm.drop(columns=exclude_cols)
            results_root = results_root / f"exclude_{args.exclude_modality.lower()}"
            # dump a text line saying which columns were dropped to results_root/excluded_columns.txt
            excluded_cols_path = results_root / "excluded_columns.txt"
            excluded_cols_path.parent.mkdir(parents=True, exist_ok=True)
            with open(excluded_cols_path, "w") as fout:
                fout.write(", ".join(exclude_cols) + "\n")

        elif args.only_modality:
            if args.only_modality == "Surveillance Patterns":
                include_cols = [
                    c
                    for c in dm.columns
                    if c in MODALITY_GROUPS["Surveillance Patterns"]
                ]
            else:
                include_groups = MODALITY_GROUPS[args.only_modality]
                include_cols = sorted(
                    {
                        col
                        for group in include_groups
                        for col in FEATURES[group]
                        if col in dm.columns
                    }
                )

            include_cols += [
                EVENT_COL,
                TIME_COL,
                PATIENT_ID_COL,
                LINE_COL,
            ]
            dm = dm[include_cols]
            print(f"Kept only cols {include_cols} for modality {args.only_modality}")
            results_root = results_root / f"only_{args.only_modality.lower()}"
            # dump a text line saying which columns were dropped to results_root/excluded_columns.txt
            included_cols_path = results_root / "included_columns.txt"
            included_cols_path.parent.mkdir(parents=True, exist_ok=True)
            with open(included_cols_path, "w") as fout:
                fout.write(", ".join(include_cols) + "\n")
        elif args.drop_age or args.drop_age_less_regularized:
            dm = dm.drop(columns=["AGE"])
            print("Dropped AGE column from design matrix.")
            if args.drop_age_less_regularized:
                results_root = results_root / "drop_age_less_regularized"
            else:
                results_root = results_root / "drop_age"
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
            configs = MODEL_CONFIGS[model]
            if run_ablation:
                allowed = set(ablation_cfg_names.get(model, []))
                if not allowed:
                    raise ValueError(
                        f"No ablation config names listed for model {model} in {ablation_cfg_path}."
                    )
                configs = [cfg for cfg in configs if cfg.get("name") in allowed]
                if not configs:
                    raise ValueError(
                        f"No configs left for model {model} after ablation filter."
                    )
                print(
                    f"Using {len(configs)} filtered configs for {model} from {ablation_cfg_path}."
                )
            configs_dict = {cfg["name"]: cfg for cfg in configs}
            model_metrics: Dict[str, object] = {
                "model": model,
                "outer_folds": {},
            }
            if run_ablation:
                if len(configs_dict) != 1:
                    raise ValueError(
                        f"Expected exactly one ablation config for model {model}, got {len(configs_dict)}."
                    )
                config_name, config = next(iter(configs_dict.items()))
                print(
                    f"Using single ablation config {config_name} for {model}; skipping inner folds."
                )
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
                    _write_metrics_artifact(fold_metrics_path, outer_entry)
                final_metrics_path = results_root / model / "eval_metrics.json"
                _write_metrics_artifact(final_metrics_path, model_metrics)
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
                best_config_name, best_c, c_tie_names = _select_best_config(
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
                _write_metrics_artifact(fold_metrics_path, outer_entry)

            final_metrics_path = results_root / model / "eval_metrics.json"
            _write_metrics_artifact(final_metrics_path, model_metrics)
            print(f"Wrote metrics to {final_metrics_path}")
        except Exception as exc:
            print(f"[ERROR] {model} overall: {exc}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
