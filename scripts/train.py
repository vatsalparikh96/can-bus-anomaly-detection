"""Train an MLP autoencoder on OTIDS attack-free data.

Usage:
    python scripts/train.py
    python scripts/train.py --limit 200000 --epochs 10
    python scripts/train.py --config configs/default.yaml --out-dir results/run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load import find_file, load_otids_file
from src.data.preprocess import FeatureBuilder
from src.data.dataset import CANFrameDataset
from src.models.autoencoder import MLPAutoencoder
from src.training.train import train
from src.training.evaluate import compute_recon_errors
from src.utils.plot import plot_loss_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Override data.raw_dir from config")
    parser.add_argument("--out-dir", type=Path, default=Path("results/_latest"),
                        help="Where to write run artifacts. Default is a gitignored "
                             "scratch dir; pass results/runs/<name> for a kept run.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit rows per file (fast iteration)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override data.top_k_can_ids from config")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Override outer hidden dim")
    parser.add_argument("--bottleneck", type=int, default=None,
                        help="Override bottleneck dim")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir    = args.raw_dir    or Path(cfg["data"]["raw_dir"])
    epochs     = args.epochs     or cfg["training"]["epochs"]
    lr         = args.lr         or cfg["training"]["lr"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    top_k      = args.top_k      or cfg["data"]["top_k_can_ids"]
    cfg_hidden, cfg_bottleneck, _ = cfg["model"]["hidden_dims"]
    hidden     = args.hidden     or cfg_hidden
    bottleneck = args.bottleneck or cfg_bottleneck
    out_dir    = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:    {device}")
    print(f"raw_dir:   {raw_dir}")
    print(f"out_dir:   {out_dir}")
    print(f"limit:     {args.limit}")

    # Load attack-free traffic.
    af_path = find_file(raw_dir, "attack_free")
    if af_path is None:
        sys.exit(f"No attack-free file in {raw_dir}")
    print(f"\nLoading {af_path.name}...")
    df = load_otids_file(af_path, limit=args.limit)
    print(f"  rows: {len(df):,}")

    # Chronological train/val/test split.
    n = len(df)
    n_train = int(cfg["data"]["train_frac"] * n)
    n_val   = int(cfg["data"]["val_frac"]   * n)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test  = df.iloc[n_train + n_val:].reset_index(drop=True)
    print(f"  split: train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    # Feature engineering: fit on train ONLY.
    fb = FeatureBuilder(top_k=top_k)
    fb.fit(df_train)
    X_train = fb.transform(df_train)
    X_val   = fb.transform(df_val)
    X_test  = fb.transform(df_test)
    print(f"  features: {X_train.shape[1]} per frame")

    # DataLoaders.
    train_loader = DataLoader(CANFrameDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(CANFrameDataset(X_val),   batch_size=batch_size, shuffle=False)

    # Model.
    model = MLPAutoencoder(
        n_features=fb.n_features, hidden=hidden, bottleneck=bottleneck
    ).to(device)
    print(f"  model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train.
    print("\nTraining...")
    history = train(
        model, train_loader, val_loader,
        n_epochs=epochs, lr=lr,
        patience=cfg["training"]["early_stopping_patience"],
        device=device, verbose=not args.quiet,
    )

    # Compute training reconstruction errors — needed by evaluate.py for thresholding.
    print("\nComputing training reconstruction errors (for threshold use)...")
    train_errors = compute_recon_errors(model, X_train, device=device)

    # Save artifacts: model weights, FeatureBuilder, history, X_test, train errors, model config.
    torch.save(model.state_dict(), out_dir / "model.pt")
    fb.save(out_dir / "feature_builder.npz")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.save(out_dir / "X_test_normal.npy", X_test)
    np.save(out_dir / "train_errors.npy", train_errors)
    with open(out_dir / "model_config.json", "w") as f:
        json.dump({
            "n_features": fb.n_features,
            "hidden": hidden,
            "bottleneck": bottleneck,
            "top_k_can_ids": top_k,
        }, f, indent=2)
    plot_loss_curves(history, save_path=out_dir / "loss_curve.png")

    print(f"\nSaved artifacts to {out_dir}/")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
