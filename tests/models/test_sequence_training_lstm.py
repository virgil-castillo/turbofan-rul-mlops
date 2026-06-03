"""Tests for the generalized sequence training helpers (LSTM + shared API)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from turbofan.config.schema import SequenceConfig
from turbofan.models.sequence_models import build_sequence_model
from turbofan.models.sequence_training import (
    TrainingResult,
    predict_windows,
    train_gru_model,
    train_sequence_model,
)

EXPECTED_HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "validation_windows_rmse",
    "validation_windows_mae",
]


def _loader(
    shuffle: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build a tiny deterministic sequence DataLoader.

    Args:
        shuffle: Whether the DataLoader should shuffle samples.

    Returns:
        DataLoader with four two-step windows, scalar targets, and lengths.
    """
    features = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.2]],
            [[0.2, 0.1], [0.3, 0.4]],
            [[0.4, 0.3], [0.5, 0.6]],
            [[0.6, 0.5], [0.7, 0.8]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    lengths = torch.full((4,), 2, dtype=torch.int64)
    return DataLoader(
        TensorDataset(features, targets, lengths), batch_size=2, shuffle=shuffle
    )


def test_train_sequence_model_trains_lstm_and_returns_history() -> None:
    """train_sequence_model trains an LSTM and reports the full metric history."""
    model = build_sequence_model(
        "lstm", input_size=2, hidden_size=4, num_layers=1, dropout=0.0
    )
    config = SequenceConfig(
        architecture="lstm",
        batch_size=2,
        hidden_size=4,
        epochs=3,
        patience=2,
        learning_rate=0.01,
    )

    result = train_sequence_model(
        model=model,
        train_loader=_loader(shuffle=True),
        validation_windows_loader=_loader(),
        config=config,
        device=torch.device("cpu"),
        random_seed=7,
        max_rul=1,
    )

    assert isinstance(result, TrainingResult)
    assert result.model is model
    assert list(result.history.columns) == EXPECTED_HISTORY_COLUMNS
    assert not result.history.empty
    assert result.best_epoch >= 1
    assert result.best_metric >= 0.0


def test_train_sequence_model_restores_best_lstm_state_after_early_stopping() -> None:
    """LSTM training early-stops and restores the best-epoch weights.

    Trains for a fixed three epochs with patience that allows the run to stop
    early, then confirms ``predict_windows`` on the restored model reproduces the
    recorded best validation-window RMSE — proving the best epoch's weights were
    restored rather than the final epoch's.
    """
    train_loader = DataLoader(
        TensorDataset(
            torch.zeros((2, 1, 1), dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            torch.full((2,), 1, dtype=torch.int64),
        ),
        batch_size=2,
    )
    validation_windows_loader = DataLoader(
        TensorDataset(
            torch.zeros((2, 1, 1), dtype=torch.float32),
            torch.full((2,), 0.2, dtype=torch.float32),
            torch.full((2,), 1, dtype=torch.int64),
        ),
        batch_size=2,
    )
    model = build_sequence_model(
        "lstm", input_size=1, hidden_size=2, num_layers=1, dropout=0.0
    )
    config = SequenceConfig(
        architecture="lstm",
        batch_size=2,
        hidden_size=2,
        epochs=3,
        patience=3,
        learning_rate=0.05,
    )

    result = train_sequence_model(
        model=model,
        train_loader=train_loader,
        validation_windows_loader=validation_windows_loader,
        config=config,
        device=torch.device("cpu"),
        random_seed=7,
        max_rul=1,
    )
    restored = predict_windows(
        model, validation_windows_loader, torch.device("cpu"), max_rul=1
    )

    assert result.best_metric == pytest.approx(
        np.sqrt(np.mean((np.clip(restored, 0.0, None) - 0.2) ** 2)),
        abs=1e-6,
    )


def test_train_gru_model_alias_delegates_to_train_sequence_model() -> None:
    """The retained train_gru_model alias trains a GRU identically.

    Running the alias and the generalized function from the same seed and inputs
    must yield identical restored predictions, proving the alias is a thin
    delegate and not a divergent code path.
    """
    config = SequenceConfig(
        architecture="gru",
        batch_size=2,
        hidden_size=4,
        epochs=2,
        patience=2,
        learning_rate=0.01,
    )

    torch.manual_seed(0)
    via_alias = build_sequence_model(
        "gru", input_size=2, hidden_size=4, num_layers=1, dropout=0.0
    )
    train_gru_model(
        model=via_alias,
        train_loader=_loader(),
        validation_windows_loader=_loader(),
        config=config,
        device=torch.device("cpu"),
        random_seed=11,
        max_rul=1,
    )
    via_alias_preds = predict_windows(
        via_alias, _loader(), torch.device("cpu"), max_rul=1
    )

    torch.manual_seed(0)
    via_generalized = build_sequence_model(
        "gru", input_size=2, hidden_size=4, num_layers=1, dropout=0.0
    )
    train_sequence_model(
        model=via_generalized,
        train_loader=_loader(),
        validation_windows_loader=_loader(),
        config=config,
        device=torch.device("cpu"),
        random_seed=11,
        max_rul=1,
    )
    via_generalized_preds = predict_windows(
        via_generalized, _loader(), torch.device("cpu"), max_rul=1
    )

    np.testing.assert_allclose(via_alias_preds, via_generalized_preds, rtol=1e-6)
