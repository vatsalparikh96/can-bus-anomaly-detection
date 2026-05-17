"""Tests for MLPAutoencoder."""

import torch

from src.models.autoencoder import MLPAutoencoder


def test_forward_shape_matches_input():
    model = MLPAutoencoder(n_features=30)
    x = torch.randn(4, 30)
    y = model(x)
    assert y.shape == x.shape


def test_output_in_zero_one():
    """Sigmoid output: every value must lie in [0, 1]."""
    model = MLPAutoencoder(n_features=30)
    x = torch.randn(100, 30) * 10  # large inputs to push activations hard
    y = model(x)
    assert (y >= 0.0).all()
    assert (y <= 1.0).all()


def test_parameter_count_correct_for_default_dims():
    """Default config: 30 -> 16 -> 8 -> 16 -> 30 = 4 Linear layers = 8 param groups."""
    model = MLPAutoencoder(n_features=30, hidden=16, bottleneck=8)
    params = list(model.parameters())
    assert len(params) == 8

    # Spot-check one weight tensor's shape.
    # Encoder Linear(30, 16) -> weight shape is (16, 30) and bias is (16,).
    assert params[0].shape == (16, 30)
    assert params[1].shape == (16,)


def test_backward_produces_gradients_on_all_params():
    """Smoke test: a forward + backward should set .grad on every parameter."""
    model = MLPAutoencoder(n_features=30)
    x = torch.randn(4, 30)
    y = model(x)
    loss = ((y - x) ** 2).mean()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_supports_custom_dimensions():
    model = MLPAutoencoder(n_features=50, hidden=32, bottleneck=4)
    x = torch.randn(8, 50)
    y = model(x)
    assert y.shape == (8, 50)
