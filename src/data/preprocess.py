"""Feature engineering for CAN frames.

The `FeatureBuilder` class follows the sklearn `fit` / `transform` pattern.
`fit` learns parameters from training data only; `transform` applies them to
any DataFrame. This separation enforces the no-data-leakage rule: parameters
learned from training data are reused unchanged on val, test, or production
inputs.

Output layout (top_k=20):
    cols  0..19  -> one-hot for top-20 CAN IDs (learned from train)
    col   20      -> "other" bucket for any CAN ID outside the top-20
    cols  21..28 -> data_0/255 ... data_7/255 (fixed; NaN -> 0)
    col   29      -> dlc / 8 (fixed)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class FeatureBuilder:
    """Convert CAN frames into a (n, n_features) float32 feature matrix."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.top_ids: list[int] | None = None
        # _lookup[i] = column index for CAN ID i; top_k means 'other'.
        self._lookup: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return self.top_k + 1 + 8 + 1

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        """Learn top_k most common CAN IDs from `df`."""
        counts = df["can_id"].value_counts()
        self.top_ids = counts.head(self.top_k).index.tolist()
        # 2048 covers all 11-bit standard CAN IDs.
        self._lookup = np.full(2048, self.top_k, dtype=np.int32)
        for col, cid in enumerate(self.top_ids):
            self._lookup[cid] = col
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessing. Returns shape (len(df), n_features)."""
        if self._lookup is None:
            raise RuntimeError("FeatureBuilder not fitted — call .fit() first.")
        n = len(df)
        X = np.zeros((n, self.n_features), dtype=np.float32)

        # One-hot CAN IDs (vectorized via lookup table).
        can_ids = df["can_id"].to_numpy()
        # Clip to 11-bit range so unknown extended IDs don't index out of bounds.
        can_ids = np.clip(can_ids, 0, 2047)
        cols = self._lookup[can_ids]
        X[np.arange(n), cols] = 1.0

        # Data bytes / 255 (fill NaN with 0).
        for j in range(8):
            X[:, self.top_k + 1 + j] = df[f"data_{j}"].fillna(0).to_numpy() / 255.0

        # DLC / 8.
        X[:, -1] = df["dlc"].to_numpy() / 8.0

        return X

    def save(self, path: Path | str) -> None:
        """Save fitted state to a .npz file."""
        if self.top_ids is None:
            raise RuntimeError("FeatureBuilder not fitted.")
        np.savez(
            str(path),
            top_ids=np.array(self.top_ids, dtype=np.int64),
            top_k=np.array(self.top_k, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Path | str) -> "FeatureBuilder":
        """Load fitted state from a .npz file."""
        data = np.load(str(path))
        fb = cls(top_k=int(data["top_k"]))
        fb.top_ids = data["top_ids"].tolist()
        fb._lookup = np.full(2048, fb.top_k, dtype=np.int32)
        for col, cid in enumerate(fb.top_ids):
            fb._lookup[cid] = col
        return fb
