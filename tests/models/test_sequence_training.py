"""Tests for GRU sequence training helpers."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from turbofan.config.schema import SequenceConfig
from turbofan.models.gru import GRURULRegressor
from turbofan.models.sequence_training import (
    TrainingResult,
    predict_windows,
    resolve_device,
    train_gru_model,
)

EXPECTED_HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "validation_final_window_rmse",
    "validation_final_window_mae",
    "validation_final_window_phm08_score",
    "validation_windows_rmse",
    "validation_windows_mae",
    "validation_windows_phm08_score",
]


def _loader(shuffle: bool = False) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a tiny deterministic sequence DataLoader.

    Args:
        shuffle: Whether the DataLoader should shuffle samples.

    Returns:
        DataLoader with four two-step windows and scalar targets.
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
    return DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=shuffle)


def test_resolve_device_cpu_returns_cpu_device() -> None:
    """CPU device resolution returns a torch CPU device."""
    device = resolve_device("cpu")

    assert device.type == "cpu"


def test_resolve_device_cuda_unavailable_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable CUDA requests fail with a clear ValueError."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA.*not available"):
        resolve_device("cuda")


def test_predict_windows_returns_float64_prediction_per_window() -> None:
    """Window prediction returns one float64 numpy value per input window."""
    model = GRURULRegressor(input_size=2, hidden_size=4, num_layers=1, dropout=0.0)

    predictions = predict_windows(model, _loader(), torch.device("cpu"))

    assert predictions.dtype == np.float64
    assert predictions.shape == (4,)


def test_train_gru_model_returns_result_with_expected_history() -> None:
    """GRU training returns model, metrics history, and best validation metadata."""
    model = GRURULRegressor(input_size=2, hidden_size=4, num_layers=1, dropout=0.0)
    config = SequenceConfig(
        batch_size=2,
        hidden_size=4,
        epochs=3,
        patience=2,
        learning_rate=0.01,
    )

    result = train_gru_model(
        model=model,
        train_loader=_loader(shuffle=True),
        validation_final_loader=_loader(),
        validation_windows_loader=_loader(),
        config=config,
        device=torch.device("cpu"),
        random_seed=7,
    )

    assert isinstance(result, TrainingResult)
    assert result.model is model
    assert not result.history.empty
    assert list(result.history.columns) == EXPECTED_HISTORY_COLUMNS
    assert result.best_epoch >= 1
    assert result.best_metric >= 0.0
