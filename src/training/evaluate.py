"""Reconstruction-error computation and threshold-based metric evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_recon_errors(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 4096,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Per-frame MSE reconstruction error. Returns shape (len(X),) float32 array."""
    model.eval()
    errors = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            pred = model(batch)
            mse = ((pred - batch) ** 2).mean(dim=1)
            errors[i:i + batch_size] = mse.cpu().numpy()
    return errors


def pick_threshold(errors_normal: np.ndarray, percentile: float = 99.0) -> float:
    """Return the threshold value at the given percentile of a normal-only error array."""
    return float(np.quantile(errors_normal, percentile / 100.0))


def evaluate_at_threshold(
    err_negative: np.ndarray,
    err_positive: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute precision / recall / F1 at a given threshold, plus PR-AUC (threshold-independent)."""
    errors = np.concatenate([err_negative, err_positive])
    labels = np.concatenate([
        np.zeros(len(err_negative)),
        np.ones(len(err_positive)),
    ])
    preds = (errors > threshold).astype(int)
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "pr_auc":    float(average_precision_score(labels, errors)),
    }
