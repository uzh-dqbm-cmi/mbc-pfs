# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import os
import warnings
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycox.models import DeepHitSingle
from pycox.models.data import pair_rank_mat
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from torch.utils.data import DataLoader

from config import ADMIN_CENSOR_DAYS, DEVICE, EVAL_HORIZONS, PATIENCE
from model_setup import _seed_reproducibly
from preprocess import preprocess_df
from utils import (
    eval_model,
    evaluate_lot_and_stage_metrics,
    save_eval_metrics,
    save_full_survival_curves,
)


def train_deephit(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    design_matrix: pd.DataFrame,
    config: dict,
    outer: bool,
    outdir: Path,
    run_ablation: bool,
) -> dict:
    if outdir is None:
        raise ValueError("outdir must be provided for train_deephit.")
    os.makedirs(outdir, exist_ok=True)

    _seed_reproducibly(int(config["seed"]))

    splits = preprocess_df(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        design_matrix=design_matrix,
        ignore_prefix=config.get("ignore_prefix"),
    )
    with warnings.catch_warnings():
        # PyCox emits a noisy warning when quantile cuts are duplicated; swallow it.
        warnings.filterwarnings(
            "ignore",
            message=r"cuts are not unique, continue with .*",
            category=UserWarning,
        )
        label_transform = LabTransDiscreteTime(
            config["num_time_steps"], scheme="quantiles"
        )
        time_train_discrete_np, event_train_discrete_np = label_transform.fit_transform(
            splits["time_train_arr"], splits["event_train_arr"]
        )
    time_val_discrete_np, event_val_discrete_np = label_transform.transform(
        splits["time_val_arr"], splits["event_val_arr"]
    )
    time_grid_train_np = label_transform.cuts

    output_num_time_steps = len(time_grid_train_np)

    X_train = torch.tensor(splits["X_train_np"], dtype=torch.float32, device=DEVICE)
    time_train = torch.tensor(time_train_discrete_np, dtype=torch.int64, device=DEVICE)
    event_train = torch.tensor(
        event_train_discrete_np, dtype=torch.int32, device=DEVICE
    )
    X_val = torch.tensor(splits["X_val_np"], dtype=torch.float32, device=DEVICE)
    time_val = torch.tensor(time_val_discrete_np, dtype=torch.int64, device=DEVICE)
    event_val = torch.tensor(event_val_discrete_np, dtype=torch.int32, device=DEVICE)
    train_data = list(zip(X_train, time_train, event_train))
    val_data = list(zip(X_val, time_val, event_val))

    num_input_features = X_train.size(1)

    def build_head(
        in_dim: int, hidden_dims, dropout: float, out_dim: int
    ) -> nn.Sequential:
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    batch_size = config["batch_size"]

    base_neural_net = build_head(
        in_dim=num_input_features,
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        out_dim=output_num_time_steps,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        base_neural_net.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    deephit_model = DeepHitSingle(
        base_neural_net,
        alpha=config.get("alpha", 1.0),
        device=DEVICE,
        duration_index=time_grid_train_np,
    )
    deephit_loss = deephit_model.loss

    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_data,
        batch_size,
        shuffle=True,
        generator=torch_generator,
        num_workers=0,
    )
    val_loader = DataLoader(val_data, batch_size, shuffle=False, num_workers=0)
    train_epoch_losses: list[float] = []
    val_epoch_losses: list[float] = []
    best_epoch_index = -1
    best_val_loss = float("inf")
    best_params = None
    epochs_without_improve = 0

    for epoch_index in range(config["num_epochs"]):
        base_neural_net.train()
        for X_batch, Y_batch, D_batch in train_loader:
            neural_net_output = base_neural_net(X_batch)
            rank_mat = pair_rank_mat(Y_batch.cpu().numpy(), D_batch.cpu().numpy())
            rank_mat = torch.tensor(rank_mat, dtype=torch.int, device=DEVICE)
            loss_batch = deephit_loss(neural_net_output, Y_batch, D_batch, rank_mat)
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        base_neural_net.eval()
        with torch.no_grad():
            train_loss = torch.tensor(0.0, dtype=torch.float, device=DEVICE)
            num_points = 0
            for X_batch, Y_batch, D_batch in train_loader:
                batch_num_points = X_batch.size(0)
                neural_net_output = base_neural_net(X_batch)
                rank_mat = pair_rank_mat(Y_batch.cpu().numpy(), D_batch.cpu().numpy())
                rank_mat = torch.tensor(rank_mat, dtype=torch.int, device=DEVICE)
                train_loss += (
                    deephit_loss(neural_net_output, Y_batch, D_batch, rank_mat)
                    * batch_num_points
                )
                num_points += batch_num_points
            train_loss = float(train_loss / num_points)
            train_epoch_losses.append(train_loss)

        with torch.no_grad():
            val_loss = torch.tensor(0.0, dtype=torch.float, device=DEVICE)
            num_points = 0
            for X_batch, Y_batch, D_batch in val_loader:
                batch_num_points = X_batch.size(0)
                neural_net_output = base_neural_net(X_batch)
                rank_mat = pair_rank_mat(Y_batch.cpu().numpy(), D_batch.cpu().numpy())
                rank_mat = torch.tensor(rank_mat, dtype=torch.int, device=DEVICE)
                val_loss += (
                    deephit_loss(neural_net_output, Y_batch, D_batch, rank_mat)
                    * batch_num_points
                )
                num_points += batch_num_points
        val_loss = float(val_loss / num_points)
        val_epoch_losses.append(val_loss)

        if not np.isfinite(val_loss):
            print(
                "    [WARNING] Validation loss is non-finite; using current model weights."
            )
            if best_params is None:
                best_params = deepcopy(base_neural_net.state_dict())
                best_epoch_index = epoch_index
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_index = epoch_index
            best_params = deepcopy(base_neural_net.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                break

    if best_params is None:
        print(
            "    [WARNING] No valid validation improvement; using final model weights."
        )
        best_params = deepcopy(base_neural_net.state_dict())
        best_epoch_index = max(best_epoch_index, 0)

    print(f"    Best epoch at {best_epoch_index + 1} with val loss {best_val_loss}")
    base_neural_net.load_state_dict(best_params)
    base_neural_net.eval()

    surv_test_np = deephit_model.predict_surv(
        splits["X_test_np"].astype("float32"),
        batch_size=batch_size,
        to_cpu=True,
        numpy=True,
    )
    interpolation_factor = 10
    surv_test_interp_df = deephit_model.interpolate(
        interpolation_factor
    ).predict_surv_df(splits["X_test_np"].astype("float32"), batch_size=batch_size)
    time_grid_train_interp_np = surv_test_interp_df.index.to_numpy()
    horizon_mask = time_grid_train_interp_np < float(ADMIN_CENSOR_DAYS)
    if horizon_mask.any():
        surv_test_interp_df = surv_test_interp_df.iloc[horizon_mask]
        time_grid_train_interp_np = surv_test_interp_df.index.to_numpy()
    surv_test_interp_np = surv_test_interp_df.to_numpy().T

    metrics = eval_model(
        time_grid_train_np=time_grid_train_interp_np,
        surv_test_np=surv_test_interp_np,
        surv_test_df=surv_test_interp_df,
        splits=splits,
    )
    resolved_times = metrics.get("time")
    if resolved_times:
        metrics["resolved_time"] = [float(t) for t in resolved_times]
        metrics["time"] = [
            float(t) for t in metrics.get("eval_horizons", resolved_times)
        ]

    metrics["config_name"] = config["name"]
    metrics["best_epoch_index"] = int(best_epoch_index)

    if not run_ablation:
        plt.figure(figsize=(8, 5))
        epoch_indices = range(1, len(train_epoch_losses) + 1)
        plt.plot(epoch_indices, train_epoch_losses, label="Training")
        if val_epoch_losses:
            plt.plot(epoch_indices, val_epoch_losses, "--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outdir}/training_curve.png", dpi=200)
        plt.close()
    if not outer:
        return metrics
    else:
        if not run_ablation:
            lot_metrics, hr_her2_metrics = evaluate_lot_and_stage_metrics(
                design_matrix=design_matrix,
                splits=splits,
                surv_test_df=surv_test_interp_df,
                time_grid_train_np=time_grid_train_interp_np,
                eval_horizons=EVAL_HORIZONS,
            )
            metrics["lot_metrics"] = lot_metrics
            metrics["hr_her2_metrics"] = hr_her2_metrics

            save_eval_metrics(
                metrics=metrics,
                outdir=outdir,
            )

            save_full_survival_curves(
                outdir=outdir,
                time_grid=time_grid_train_np,
                surv_test_np=surv_test_np,
                idx_test_array=test_idx,
                filename="surv_test.npz",
            )

            torch.save(
                {
                    "model_state_dict": base_neural_net.state_dict(),
                    "config": config,
                    "best_epoch_index": best_epoch_index,
                    "train_epoch_losses": train_epoch_losses,
                    "val_epoch_losses": val_epoch_losses,
                    "numeric_cols": splits["numeric_cols"],
                    "binary_cols": splits["binary_cols"],
                    "feature_cols": splits["feature_cols"],
                    "time_grid": time_grid_train_np,
                    "interpolation_factor": interpolation_factor,
                },
                os.path.join(outdir, "model.pt"),
            )
            joblib.dump(
                {
                    "model_type": "deephit",
                    "preprocessor": splits["preprocessor"],
                    "numeric_cols": splits["numeric_cols"],
                    "binary_cols": splits["binary_cols"],
                    "feature_cols": splits["feature_cols"],
                    "config": config,
                    "model_pt": "model.pt",
                },
                os.path.join(outdir, "preprocess.joblib"),
            )

        return metrics
