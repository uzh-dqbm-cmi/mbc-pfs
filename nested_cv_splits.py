from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedGroupKFold,
                                     StratifiedShuffleSplit)

from cv_config import K_INNER, K_OUTER
from model_setup import EVENT_COL, GROUP_COL, TIME_COL
from preprocess import _hr_her2_admin_strata


def _min_groups_per_stratum(strata: np.ndarray, groups: np.ndarray) -> int:
    counts: Dict[str, set] = {}
    for s, g in zip(strata, groups):
        counts.setdefault(str(s), set()).add(str(g))
    if not counts:
        return 0
    return min(len(v) for v in counts.values())


def generate_early_stopping_split(
    design_matrix: pd.DataFrame,
    seed: int,
    val_frac: float,
    base_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Create a single stratified-group split for early stopping using stratified shuffle.

    This uses the same strata definition as the nested CV and carves out roughly
    ``val_frac`` (default 10%) of the provided rows for early stopping. If
    ``base_indices`` is provided, the split is restricted to that subset of the
    design matrix (e.g., an outer-fold train set) and returned indices are in
    the original design-matrix index space.
    """

    work_df = design_matrix.iloc[base_indices]
    groups_all = work_df[GROUP_COL].astype(str).to_numpy()
    strata = _hr_her2_admin_strata(
        work_df,
        group_col=GROUP_COL,
        time_col=TIME_COL,
        event_col=EVENT_COL,
    )
    if len(groups_all) != len(strata):
        raise ValueError("Strata and design matrix length mismatch.")

    min_per_stratum = _min_groups_per_stratum(strata, groups_all)
    if min_per_stratum < 2:
        raise ValueError(
            "Insufficient groups per stratum for stratified split: "
            f"need at least 2, found {min_per_stratum}."
        )

    # Build a group-level table for stratified shuffle splitting.
    group_labels = []
    strata_labels = []
    for g, s in zip(groups_all, strata):
        if g not in group_labels:
            group_labels.append(g)
            strata_labels.append(s)
    group_labels_np = np.asarray(group_labels, dtype=str)
    strata_labels_np = np.asarray(strata_labels, dtype=str)
    if group_labels_np.size < 2:
        raise ValueError(
            "Insufficient groups for early-stopping split: need at least 2."
        )

    holdout_n = int(round(float(val_frac) * float(group_labels_np.size)))
    holdout_n = max(1, min(int(group_labels_np.size - 1), holdout_n))
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=holdout_n, random_state=seed
    )
    _, group_holdout_local = next(splitter.split(group_labels_np, strata_labels_np))
    holdout_groups = set(group_labels_np[group_holdout_local])

    mask_holdout = work_df[GROUP_COL].astype(str).isin(holdout_groups).to_numpy()
    holdout_idx = base_indices[mask_holdout]
    train_idx = base_indices[~mask_holdout]

    return {
        "train_idx": np.asarray(train_idx, dtype=int),
        "early_stop_idx": np.asarray(holdout_idx, dtype=int),
    }


def generate_nested_cv_indices(
    design_matrix: pd.DataFrame,
    seed: int,
    output_dir: Path,
) -> Tuple[Path, List[Path]]:

    if design_matrix.empty:
        raise ValueError("Design matrix is empty; cannot build nested CV splits.")

    required_cols = {GROUP_COL, TIME_COL, EVENT_COL, "LINE"}
    missing = [col for col in required_cols if col not in design_matrix.columns]
    if missing:
        raise KeyError(f"Design matrix missing required columns: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "seed": int(seed),
        "k_outer": int(K_OUTER),
        "k_inner": int(K_INNER),
        "splits": [],
    }
    split_paths: List[Path] = []

    groups_all = design_matrix[GROUP_COL].astype(str).to_numpy()
    strata = _hr_her2_admin_strata(
        design_matrix,
        group_col=GROUP_COL,
        time_col=TIME_COL,
        event_col=EVENT_COL,
    )

    if len(groups_all) != len(strata):
        raise ValueError("Strata and design matrix length mismatch.")
    # assert and print the strata numbers: should be 8 in total
    unique, counts = np.unique(strata, return_counts=True)
    # print(f"Strata counts: {strata_counts}")
    assert len(unique) == 8, f"Expected 8 strata, found {len(unique)}."

    min_per_stratum = _min_groups_per_stratum(strata, groups_all)
    needed = max(K_OUTER, K_INNER)
    if min_per_stratum < needed:
        raise ValueError(
            "Insufficient groups per stratum for requested folds: "
            f"need at least {needed}, found {min_per_stratum}."
        )

    outer_cv = StratifiedGroupKFold(n_splits=K_OUTER, shuffle=True, random_state=seed)

    for outer_idx, (train_val_idx, test_idx) in enumerate(
        outer_cv.split(design_matrix, strata, groups=groups_all)
    ):
        train_val_idx = np.asarray(train_val_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        outer_es_split = generate_early_stopping_split(
            design_matrix=design_matrix,
            seed=seed,
            val_frac=0.10,
            base_indices=train_val_idx,
        )
        outer_es_idx = outer_es_split["early_stop_idx"]
        y_outer = strata[train_val_idx]
        groups_outer = groups_all[train_val_idx]
        X_outer = design_matrix.iloc[train_val_idx]

        inner_cv = StratifiedGroupKFold(
            n_splits=K_INNER,
            shuffle=True,
            random_state=seed,
        )

        for inner_idx, (inner_train_local, inner_val_local) in enumerate(
            inner_cv.split(X_outer, y_outer, groups=groups_outer)
        ):
            idx_train = train_val_idx[np.asarray(inner_train_local, dtype=int)]
            idx_val = train_val_idx[np.asarray(inner_val_local, dtype=int)]
            idx_test = test_idx.copy()
            inner_es_split = generate_early_stopping_split(
                design_matrix=design_matrix,
                seed=seed,
                val_frac=0.10,
                base_indices=idx_train,
            )
            inner_es_idx = inner_es_split["early_stop_idx"]

            def _rows(name: str, idxs: np.ndarray) -> pd.DataFrame:
                idxs = np.asarray(idxs, dtype=int)
                subset = design_matrix.iloc[idxs]
                inner_es_mask = np.isin(idxs, inner_es_idx)
                outer_es_mask = np.isin(idxs, outer_es_idx)
                return pd.DataFrame(
                    {
                        "split": name,
                        "idx": idxs,
                        "PATIENT_ID": subset[GROUP_COL].astype(str).to_numpy(),
                        "LINE": subset["LINE"].astype(int).to_numpy(),
                        "outer_fold": int(outer_idx),
                        "inner_fold": int(inner_idx),
                        "inner_es": inner_es_mask.astype(int),
                        "outer_es": outer_es_mask.astype(int),
                    }
                )

            split_df = pd.concat(
                [
                    _rows("train", idx_train),
                    _rows("val", idx_val),
                    _rows("test", idx_test),
                ],
                ignore_index=True,
            )

            split_filename = (
                f"nestedcv_seed{seed}_outer{outer_idx}_inner{inner_idx}.csv"
            )
            split_path = output_dir / split_filename
            # If a split exists, ensure the core split assignment is unchanged; allow
            # upgrading older files that predate early-stopping columns.
            if split_path.exists():
                if not split_df.equals(pd.read_csv(split_path)):
                    raise ValueError(
                        f"Existing split file {split_path} differs from newly generated split."
                    )
            else:
                split_df.to_csv(split_path, index=False)
            split_paths.append(split_path)

            manifest[f"outer{outer_idx}_inner{inner_idx}"] = {
                "outer_fold": int(outer_idx),
                "inner_fold": int(inner_idx),
                "path": split_path.as_posix(),
                "n_train": int(idx_train.size),
                "n_val": int(idx_val.size),
                "n_test": int(idx_test.size),
            }

        # check that the groups are disjoint between all three fold
        test_groups = set(
            design_matrix.iloc[idx_test][GROUP_COL].astype(str).to_numpy()
        )
        val_groups = set(design_matrix.iloc[idx_val][GROUP_COL].astype(str).to_numpy())
        train_groups = set(
            design_matrix.iloc[idx_train][GROUP_COL].astype(str).to_numpy()
        )
        assert (
            test_groups.isdisjoint(val_groups)
            and test_groups.isdisjoint(train_groups)
            and val_groups.isdisjoint(train_groups)
        ), "Groups are not disjoint between train, val, and test sets."

    manifest_path = output_dir / f"nestedcv_manifest_seed{seed}.json"
    with manifest_path.open("w") as fout:
        json.dump(manifest, fout, indent=2)

    return manifest, split_paths
