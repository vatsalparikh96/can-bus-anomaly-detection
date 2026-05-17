"""Tests for FeatureBuilder."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import FeatureBuilder


def _make_df(n: int = 100, n_unique_ids: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": np.arange(n) * 0.001,
        "can_id":    rng.integers(0, n_unique_ids, n),
        "dlc":       rng.integers(0, 9, n),
        **{f"data_{i}": rng.integers(0, 256, n).astype(float) for i in range(8)},
    })


def test_fit_learns_top_k_ids():
    df = _make_df(n=1000, n_unique_ids=50)
    fb = FeatureBuilder(top_k=20).fit(df)
    assert fb.top_ids is not None
    assert len(fb.top_ids) == 20


def test_transform_shape_is_30():
    df = _make_df(n=100)
    fb = FeatureBuilder(top_k=20).fit(df)
    X = fb.transform(df)
    # 20 one-hot + 1 other + 8 data bytes + 1 dlc = 30
    assert X.shape == (100, 30)
    assert X.dtype == np.float32


def test_transform_values_in_zero_one():
    df = _make_df(n=100)
    fb = FeatureBuilder(top_k=20).fit(df)
    X = fb.transform(df)
    assert X.min() >= 0.0
    assert X.max() <= 1.0


def test_onehot_block_sums_to_one():
    df = _make_df(n=100)
    fb = FeatureBuilder(top_k=20).fit(df)
    X = fb.transform(df)
    onehot_block = X[:, :21]   # top_k + 'other'
    assert np.allclose(onehot_block.sum(axis=1), 1.0)


def test_transform_without_fit_raises():
    fb = FeatureBuilder(top_k=20)
    with pytest.raises(RuntimeError):
        fb.transform(_make_df())


def test_save_load_roundtrip(tmp_path):
    df = _make_df(n=100)
    fb = FeatureBuilder(top_k=20).fit(df)
    fb.save(tmp_path / "fb.npz")

    fb2 = FeatureBuilder.load(tmp_path / "fb.npz")
    assert fb2.top_k == fb.top_k
    assert fb2.top_ids == fb.top_ids

    # Same input must give same output after a save/load cycle.
    assert np.allclose(fb.transform(df), fb2.transform(df))


def test_unknown_can_id_falls_into_other_bucket():
    df_train = _make_df(n=100, n_unique_ids=10)   # IDs in [0, 10)
    fb = FeatureBuilder(top_k=5).fit(df_train)

    # Build a new df with IDs the FB has never seen.
    df_new = pd.DataFrame({
        "timestamp": [0.0],
        "can_id":    [1500],   # well outside top-5
        "dlc":       [4],
        **{f"data_{i}": [0.0] for i in range(8)},
    })
    X = fb.transform(df_new)
    # All onehot mass should be in the 'other' column (index 5)
    assert X[0, 5] == 1.0
    assert X[0, :5].sum() == 0.0


def test_nan_data_bytes_become_zero():
    df = pd.DataFrame({
        "timestamp": [0.0],
        "can_id":    [0],
        "dlc":       [3],   # only 3 bytes used; 4..7 are NaN
        "data_0": [0.0], "data_1": [0.0], "data_2": [0.0],
        "data_3": [np.nan], "data_4": [np.nan], "data_5": [np.nan],
        "data_6": [np.nan], "data_7": [np.nan],
    })
    fb = FeatureBuilder(top_k=5).fit(df)
    X = fb.transform(df)
    # The data byte columns (cols 6..13 with top_k=5) should all be 0 here.
    data_cols = X[0, 5 + 1: 5 + 1 + 8]
    assert np.allclose(data_cols, 0.0)
