"""Tests for turbofan.models.gru."""
from __future__ import annotations

import pytest
import torch

from turbofan.models.gru import GRURULRegressor


def test_forward_returns_batch_vector_for_single_layer_gru() -> None:
    """Single-layer GRU regressor returns one RUL prediction per sequence."""
    model = GRURULRegressor(
        input_size=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    X = torch.randn(3, 5, 4)

    predictions = model(X)

    assert predictions.shape == (3,)


def test_forward_supports_stacked_layers_with_dropout() -> None:
    """Stacked GRU regressor supports dropout between recurrent layers."""
    model = GRURULRegressor(
        input_size=4,
        hidden_size=8,
        num_layers=2,
        dropout=0.2,
    )
    X = torch.randn(3, 5, 4)

    predictions = model(X)

    assert predictions.shape == (3,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_size": 0}, "input_size must be positive"),
        ({"hidden_size": 0}, "hidden_size must be positive"),
        ({"num_layers": 0}, "num_layers must be positive"),
        ({"dropout": -0.1}, "dropout must be at least 0"),
        ({"dropout": 1.0}, "dropout must be less than 1"),
    ],
)
def test_invalid_constructor_values_raise_value_error(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    """Invalid GRU hyperparameters fail fast with clear errors."""
    params: dict[str, int | float] = {
        "input_size": 4,
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        GRURULRegressor(**params)
