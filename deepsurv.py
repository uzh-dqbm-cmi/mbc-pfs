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
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ADMIN_CENSOR_DAYS, DEVICE, EVAL_HORIZONS, PATIENCE
from dl_lib.pycox.models import CoxPH
from model_setup import _seed_reproducibly
from preprocess import preprocess_df
from utils import (
    eval_model,
    evaluate_lot_and_stage_metrics,
    save_eval_metrics,
    save_full_survival_curves,
)

MODEL_NAME = "DeepSurv"


def train_deepsurv(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    design_matrix: pd.DataFrame,
    config: dict,
    outer: bool,
    outdir: Path,
    run_ablation: bool,
) -> dict:
    """
    Trains DeepSurv model and evaluates it.

    Args:
        train_idx (np.ndarray): Indices for training data.
        test_idx (np.ndarray): Indices for testing data.
        design_matrix (pd.DataFrame): The design matrix containing features.
        config (dict): Configuration dictionary.

    Returns:
        dict: Evaluation metrics.
    """
    if outdir is None:
        raise ValueError("outdir must be provided for train_deepsurv.")

    _seed_reproducibly(int(config["seed"]))
    os.makedirs(outdir, exist_ok=True)
    splits = preprocess_df(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        design_matrix=design_matrix,
        ignore_prefix=config.get("ignore_prefix"),
    )

    X_train = torch.tensor(splits["X_train_np"], dtype=torch.float32, device=DEVICE)
    time_train = torch.tensor(
        splits["time_train_arr"], dtype=torch.float32, device=DEVICE
    )
    event_train = torch.tensor(
        splits["event_train_arr"], dtype=torch.int32, device=DEVICE
    )
    X_val = torch.tensor(splits["X_val_np"], dtype=torch.float32, device=DEVICE)
    time_val = torch.tensor(splits["time_val_arr"], dtype=torch.float32, device=DEVICE)
    event_val = torch.tensor(splits["event_val_arr"], dtype=torch.int32, device=DEVICE)
    train_data = list(zip(X_train, time_train, event_train))
    val_data = list(zip(X_val, time_val, event_val))

    num_input_features = X_train.size(1)

    def _build_mlp(input_dim: int, hidden_sizes, dropout: float) -> nn.Sequential:
        layers = []
        prev = input_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    batch_size = int(config["batch_size"])
    learning_rate = float(config["lr"])
    weight_decay = float(config["weight_decay"])

    base_neural_net = _build_mlp(
        input_dim=num_input_features,
        hidden_sizes=list(config["hidden_sizes"]),
        dropout=float(config["dropout"]),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        base_neural_net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    deepsurv_model = CoxPH(base_neural_net, device=DEVICE)
    deepsurv_loss = deepsurv_model.loss

    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_data,
        batch_size,
        shuffle=True,
        generator=torch_generator,
        num_workers=0,
    )  # shuffling for minibatch gradient descent
    val_loader = DataLoader(val_data, batch_size, shuffle=False, num_workers=0)
    train_epoch_losses: list[float] = []
    val_epoch_losses: list[float] = []
    best_epoch_index = -1
    best_val_loss = float("inf")
    best_params = None
    epochs_without_improve = 0

    for epoch_index in range(config["num_epochs"]):
        base_neural_net.train()
        for X_batch, time_batch, event_batch in train_loader:
            neural_net_output = base_neural_net(X_batch)
            loss_batch = deepsurv_loss(neural_net_output, time_batch, event_batch)
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        base_neural_net.eval()
        with torch.no_grad():
            train_loss = torch.tensor(0.0, dtype=torch.float, device=DEVICE)
            num_points = 0
            for X_batch, time_batch, event_batch in train_loader:
                batch_num_points = X_batch.size(0)
                neural_net_output = base_neural_net(X_batch)
                train_loss += (
                    deepsurv_loss(neural_net_output, time_batch, event_batch)
                    * batch_num_points
                )
                num_points += batch_num_points
            train_loss = float(train_loss / num_points)
            train_epoch_losses.append(train_loss)

        with torch.no_grad():
            val_loss = torch.tensor(0.0, dtype=torch.float, device=DEVICE)
            num_points = 0
            for X_batch, time_batch, event_batch in val_loader:
                batch_num_points = X_batch.size(0)
                neural_net_output = base_neural_net(X_batch)
                val_loss += (
                    deepsurv_loss(neural_net_output, time_batch, event_batch)
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

    deepsurv_model.compute_baseline_hazards(
        input=splits["X_train_np"].astype("float32"),
        target=(
            splits["time_train_arr"].astype("float32"),
            splits["event_train_arr"].astype("int32"),
        ),
    )

    surv_test_df = deepsurv_model.predict_surv_df(
        splits["X_test_np"].astype("float32"),
        max_duration=ADMIN_CENSOR_DAYS - 1,
        batch_size=batch_size,
    )
    surv_test_np = surv_test_df.to_numpy().T
    time_grid_train_np = surv_test_df.index.to_numpy()

    metrics = eval_model(
        time_grid_train_np=time_grid_train_np,
        surv_test_np=surv_test_np,
        surv_test_df=surv_test_df,
        splits=splits,
    )

    metrics["config_name"] = config["name"]
    metrics["best_epoch_index"] = int(best_epoch_index)

    if not run_ablation:
        plt.figure(figsize=(8, 5))
        epoch_indices = range(1, len(train_epoch_losses) + 1)
        plt.plot(epoch_indices, train_epoch_losses, label="Training")
        if val_epoch_losses:
            plt.plot(
                range(1, len(val_epoch_losses) + 1),
                val_epoch_losses,
                "--",
                label="Validation",
            )
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
                surv_test_df=surv_test_df,
                time_grid_train_np=time_grid_train_np,
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
                },
                os.path.join(outdir, "model.pt"),
            )

        return metrics
