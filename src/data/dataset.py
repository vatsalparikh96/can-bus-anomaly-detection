"""PyTorch Dataset wrapper for the feature matrix produced by FeatureBuilder."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class CANFrameDataset(Dataset):
    """Wrap a (n, n_features) NumPy array as a PyTorch Dataset.

    One sample = one CAN frame's feature vector. For an autoencoder the
    target is identical to the input, so __getitem__ returns a single
    tensor; the training loop uses it as both pred and target.
    """

    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.X[idx]
