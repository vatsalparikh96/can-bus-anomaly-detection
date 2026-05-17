"""Training loop with per-epoch validation and early stopping."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """One pass through `loader`. If `optimizer` is provided, trains; else evaluates.

    Returns the per-sample average loss.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_n = 0
    grad_ctx = torch.enable_grad() if is_training else torch.no_grad()

    with grad_ctx:
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            pred = model(batch)
            loss = loss_fn(pred, batch)        # autoencoder: target = input
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.size(0)
            total_n += batch.size(0)

    return total_loss / total_n


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train with Adam optimizer, MSE loss, and val-based early stopping.

    On stop, restores the model weights from the epoch with the best val loss.
    Returns the training history dict with keys 'train_loss' and 'val_loss'.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = run_one_epoch(model, val_loader, loss_fn, optimizer=None, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            flag = " *" if improved else ""
            print(
                f"epoch {epoch:2d}  train={train_loss:.6f}  val={val_loss:.6f}  "
                f"({time.time() - t0:.1f}s){flag}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping — no val improvement for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Restored best model weights (val={best_val:.6f})")

    return history
