"""Evaluate a trained model on attack files.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --raw-dir data/otids/raw --out-dir results
    python scripts/evaluate.py --threshold-percentile 95
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load import find_file, load_otids_file
from src.data.preprocess import FeatureBuilder
from src.models.autoencoder import MLPAutoencoder
from src.training.evaluate import compute_recon_errors, evaluate_at_threshold, pick_threshold
from src.utils.plot import plot_pr_curves, plot_recon_error_hist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/otids/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/_latest"),
                        help="Where to read the trained model from and write eval "
                             "artifacts. Match the value passed to scripts/train.py.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit rows per attack file (fast iteration)")
    parser.add_argument("--threshold-percentile", type=float, default=99.0,
                        help="Percentile of training error to use as threshold (default: 99)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:    {device}")

    # Load model and FeatureBuilder from training artifacts.
    with open(args.out_dir / "model_config.json") as f:
        mc = json.load(f)
    model = MLPAutoencoder(
        n_features=mc["n_features"], hidden=mc["hidden"], bottleneck=mc["bottleneck"],
    ).to(device)
    model.load_state_dict(torch.load(args.out_dir / "model.pt", map_location=device))
    fb = FeatureBuilder.load(args.out_dir / "feature_builder.npz")
    print(f"Loaded model + FeatureBuilder from {args.out_dir}")

    # Held-out attack-free test errors.
    X_test_normal = np.load(args.out_dir / "X_test_normal.npy")
    err_normal = compute_recon_errors(model, X_test_normal, device=device)
    print(f"\nAttack-free test: n={len(err_normal):,}  mean_err={err_normal.mean():.6f}")

    # Threshold computed from training errors (saved by train.py).
    train_errors = np.load(args.out_dir / "train_errors.npy")
    threshold = pick_threshold(train_errors, percentile=args.threshold_percentile)
    print(f"Threshold ({args.threshold_percentile}th pct of train errors): {threshold:.6f}")
    print(f"Attack-free test flagged rate: {(err_normal > threshold).mean()*100:.2f}%")

    # Each attack file.
    attack_files = {
        "dos":           "DoS",
        "fuzzy":         "Fuzzy",
        "impersonation": "Impersonation",
    }
    err_attacks: dict[str, np.ndarray] = {}
    print()
    for label, keyword in attack_files.items():
        path = find_file(args.raw_dir, keyword)
        if path is None:
            print(f"  {label}: file not found, skipping")
            continue
        df = load_otids_file(path, limit=args.limit)
        X = fb.transform(df)
        err = compute_recon_errors(model, X, device=device)
        err_attacks[label] = err
        print(f"  {label:<14}: n={len(err):>9,}  mean_err={err.mean():.6f}")

    # Metrics.
    print(f"\n{'attack':<16s} {'precision':>10s} {'recall':>10s} {'F1':>10s} {'PR-AUC':>10s}")
    print("-" * 60)
    metrics = {}
    for label, err in err_attacks.items():
        m = evaluate_at_threshold(err_normal, err, threshold)
        metrics[label] = m
        print(f"{label:<16s} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {m['pr_auc']:>10.3f}")

    # Save metrics.json
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump({
            "threshold_percentile": args.threshold_percentile,
            "threshold": threshold,
            **metrics,
        }, f, indent=2)

    # Save plots.
    plot_recon_error_hist(
        err_normal, err_attacks, threshold,
        save_path=args.out_dir / "recon_error_hist.png",
    )
    plot_pr_curves(
        err_normal, err_attacks,
        save_path=args.out_dir / "pr_curves.png",
    )

    print(f"\nSaved metrics and plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
