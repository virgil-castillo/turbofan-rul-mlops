"""Tests for the shared sequence regressor and architecture registry."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from turbofan.models.sequence_models import (
    SEQUENCE_ARCHITECTURES,
    SequenceRULRegressor,
    build_sequence_model,
)


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_forward_returns_batch_vector_for_single_layer(architecture: str) -> None:
    """Single-layer regressor returns one RUL prediction per sequence."""
    model = SequenceRULRegressor(
        architecture=architecture,
        input_size=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    X = torch.randn(3, 5, 4)

    predictions = model(X)

    assert predictions.shape == (3,)


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_forward_supports_stacked_layers_with_dropout(architecture: str) -> None:
    """Stacked regressor supports dropout between recurrent layers."""
    model = SequenceRULRegressor(
        architecture=architecture,
        input_size=4,
        hidden_size=8,
        num_layers=2,
        dropout=0.2,
    )
    X = torch.randn(3, 5, 4)

    predictions = model(X)

    assert predictions.shape == (3,)


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_forward_with_full_lengths_matches_unpacked(architecture: str) -> None:
    """Full-length packed forward matches the unpacked path for both RNNs."""
    torch.manual_seed(0)
    model = SequenceRULRegressor(
        architecture=architecture,
        input_size=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    X = torch.randn(3, 6, 4)
    lengths = torch.full((3,), 6, dtype=torch.int64)
    with torch.no_grad():
        unpacked = model(X)
        packed = model(X, lengths=lengths)
    torch.testing.assert_close(packed, unpacked, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_forward_with_padded_batch_uses_real_timesteps_only(
    architecture: str,
) -> None:
    """Right-zero-padded packed forward reflects only real timesteps."""
    torch.manual_seed(0)
    model = SequenceRULRegressor(
        architecture=architecture,
        input_size=2,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    real = torch.randn(1, 3, 2)
    padded = torch.zeros(1, 5, 2)
    padded[0, :3, :] = real[0]
    lengths = torch.tensor([3], dtype=torch.int64)
    with torch.no_grad():
        reference = model(padded[:, :3, :])
        packed = model(padded, lengths=lengths)
    torch.testing.assert_close(packed, reference, atol=1e-6, rtol=1e-6)


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
@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_invalid_constructor_values_raise_value_error(
    architecture: str,
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    """Invalid hyperparameters fail fast with clear errors for every RNN."""
    params: dict[str, object] = {
        "architecture": architecture,
        "input_size": 4,
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SequenceRULRegressor(**params)  # type: ignore[arg-type]


def test_lstm_uses_lstm_layer_and_gru_uses_gru_layer() -> None:
    """Each architecture selects the matching recurrent layer type."""
    lstm = SequenceRULRegressor(
        architecture="lstm",
        input_size=3,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    gru = SequenceRULRegressor(
        architecture="gru",
        input_size=3,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )

    assert isinstance(lstm.lstm, nn.LSTM)
    assert isinstance(gru.gru, nn.GRU)


def test_build_sequence_model_returns_requested_layer_type() -> None:
    """The registry builds the correct recurrent layer per architecture name."""
    lstm = build_sequence_model(
        "lstm", input_size=3, hidden_size=4, num_layers=1, dropout=0.0
    )
    gru = build_sequence_model(
        "gru", input_size=3, hidden_size=4, num_layers=1, dropout=0.0
    )

    assert isinstance(lstm, SequenceRULRegressor)
    assert isinstance(lstm.lstm, nn.LSTM)
    assert isinstance(gru.gru, nn.GRU)


def test_build_sequence_model_unknown_name_lists_supported() -> None:
    """An unknown architecture name raises ValueError naming supported names."""
    with pytest.raises(ValueError, match="transformer") as excinfo:
        build_sequence_model(
            "transformer", input_size=3, hidden_size=4, num_layers=1, dropout=0.0
        )
    message = str(excinfo.value)
    for supported in SEQUENCE_ARCHITECTURES:
        assert supported in message


def test_sequence_regressor_rejects_unknown_architecture() -> None:
    """Constructing with an unsupported architecture name raises ValueError."""
    with pytest.raises(ValueError, match="transformer"):
        SequenceRULRegressor(
            architecture="transformer",
            input_size=3,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )
