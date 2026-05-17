"""MLP autoencoder for per-frame CAN bus anomaly detection.

Architecture:
    Linear(n_features, hidden)     ReLU
    Linear(hidden, bottleneck)     ReLU      <- compression
    Linear(bottleneck, hidden)     ReLU
    Linear(hidden, n_features)     Sigmoid    <- output in [0, 1]

The sigmoid output range matches the input range (features normalized to
[0, 1] by FeatureBuilder), so MSE loss is well-defined.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPAutoencoder(nn.Module):
    def __init__(self, n_features: int = 30, hidden: int = 16, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
