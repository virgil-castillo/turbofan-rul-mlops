"""Tests for GRU sequence training helpers."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from turbofan.config.schema import SequenceConfig
from turbofan.models.gru import GRURULRegressor
from turbofan.models.sequence_training import (
    TrainingResult,
    _evaluate_loader,
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


class _AlternatingOrderSampler(torch.utils.data.Sampler[int]):
    """Sampler that changes order on each DataLoader iteration."""

    def __init__(self, length: int) -> None:
        self._length = length
        self._iteration = 0

    def __iter__(self) -> Iterator[int]:
        indices = list(range(self._length))
        if self._iteration % 2 == 1:
            indices.reverse()
        self._iteration += 1
        return iter(indices)

    def __len__(self) -> int:
        return self._length


class _LastFeatureRegressor(torch.nn.Module):
    """Identity-like model that predicts the final feature value."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict targets encoded in the final feature column.

        Args:
            features: Sequence feature batch.

        Returns:
            One prediction per sequence window.
        """
        return features[:, -1, 0]


class _BiasRegressor(torch.nn.Module):
    """Single-parameter regressor for focused training-loop assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict the learned bias for every sequence window.

        Args:
            features: Sequence feature batch.

        Returns:
            One bias prediction per sequence window.
        """
        return self.bias.expand(features.shape[0])


class _NegativeRegressor(torch.nn.Module):
    """Fixed-output regressor that emits negative RUL predictions."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return negative predictions for every sequence window.

        Args:
            features: Sequence feature batch.

        Returns:
            One negative prediction per sequence window.
        """
        return torch.full((features.shape[0],), -5.0, dtype=torch.float32)


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


def test_evaluate_loader_keeps_predictions_and_targets_paired() -> None:
    """Evaluation pairs predictions and targets from the same loader pass."""
    targets = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    features = targets.reshape(-1, 1, 1)
    dataset = TensorDataset(features, targets)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=_AlternatingOrderSampler(len(dataset)),
    )
    model = _LastFeatureRegressor()

    metrics = _evaluate_loader(model, loader, torch.device("cpu"))

    assert metrics["rmse"] == 0.0


def test_evaluate_loader_clips_negative_predictions_before_metrics() -> None:
    """Evaluation clips impossible negative RUL predictions before metrics."""
    targets = torch.tensor([2.0, 4.0], dtype=torch.float32)
    features = torch.zeros((2, 1, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(features, targets), batch_size=2)
    model = _NegativeRegressor()

    metrics = _evaluate_loader(model, loader, torch.device("cpu"))

    assert metrics["rmse"] == pytest.approx(np.sqrt(10.0))
    assert metrics["mae"] == pytest.approx(3.0)


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


def test_train_gru_model_restores_best_state_after_early_stopping() -> None:
    """Training restores best validation state when later epochs regress."""
    train_loader = DataLoader(
        TensorDataset(
            torch.zeros((2, 1, 1), dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
        ),
        batch_size=2,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.zeros((2, 1, 1), dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
        ),
        batch_size=2,
    )
    model = _BiasRegressor()
    config = SequenceConfig(
        batch_size=2,
        hidden_size=1,
        epochs=3,
        patience=1,
        learning_rate=0.1,
    )

    result = train_gru_model(
        model=model,
        train_loader=train_loader,
        validation_final_loader=validation_loader,
        validation_windows_loader=validation_loader,
        config=config,
        device=torch.device("cpu"),
        random_seed=7,
    )
    restored_predictions = predict_windows(
        model,
        validation_loader,
        torch.device("cpu"),
    )

    assert len(result.history) == 2
    assert result.best_epoch == 1
    assert result.history["validation_final_window_rmse"].iloc[-1] > result.best_metric
    assert np.sqrt(np.mean(restored_predictions**2)) == pytest.approx(
        result.best_metric
    )
