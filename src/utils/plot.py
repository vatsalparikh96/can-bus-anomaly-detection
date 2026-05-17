"""Reusable plotting helpers for training and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_loss_curves(history: dict, save_path: Path | str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Autoencoder training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    return fig


def plot_recon_error_hist(
    err_normal: np.ndarray,
    err_attacks: dict[str, np.ndarray],
    threshold: float,
    save_path: Path | str | None = None,
    title: str = "Reconstruction error distributions",
) -> plt.Figure:
    """One log-scale histogram per attack, normal in blue, attack in red, threshold dashed."""
    n = len(err_attacks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, err) in zip(axes, err_attacks.items()):
        max_err = max(err_normal.max(), err.max())
        bins = np.linspace(0, max_err * 1.05, 80)
        ax.hist(err_normal, bins=bins, alpha=0.6, label="normal", color="tab:blue")
        ax.hist(err, bins=bins, alpha=0.6, label=name, color="tab:red")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")
        ax.set_yscale("log")
        ax.set_xlabel("reconstruction error")
        ax.set_title(name)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count (log)")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    return fig


def plot_pr_curves(
    err_normal: np.ndarray,
    err_attacks: dict[str, np.ndarray],
    save_path: Path | str | None = None,
    title: str = "Precision-recall curves",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, err in err_attacks.items():
        errors = np.concatenate([err_normal, err])
        labels = np.concatenate([np.zeros(len(err_normal)), np.ones(len(err))])
        precision, recall, _ = precision_recall_curve(labels, errors)
        ap = average_precision_score(labels, errors)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    return fig
