"""Tests for GRU sequence training helpers."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from turbofan.config.schema import SequenceConfig
from turbofan.models.gru import GRURULRegressor
from turbofan.models.sequence_training import (
    TrainingResult,
    _evaluate_loader,
    _train_one_epoch,
    predict_windows,
    resolve_device,
    train_gru_model,
)
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.windowing import WindowedSequences

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

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict targets encoded in the final feature column.

        Args:
            features: Sequence feature batch.
            lengths: Ignored; accepted for interface compatibility.

        Returns:
            One prediction per sequence window.
        """
        return features[:, -1, 0]


class _BiasRegressor(torch.nn.Module):
    """Single-parameter regressor for focused training-loop assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the learned bias for every sequence window.

        Args:
            features: Sequence feature batch.
            lengths: Ignored; accepted for interface compatibility.

        Returns:
            One bias prediction per sequence window.
        """
        return self.bias.expand(features.shape[0])


class _NegativeRegressor(torch.nn.Module):
    """Fixed-output regressor that emits negative RUL predictions."""

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return negative predictions for every sequence window.

        Args:
            features: Sequence feature batch.
            lengths: Ignored; accepted for interface compatibility.

        Returns:
            One negative prediction per sequence window.
        """
        return torch.full((features.shape[0],), -5.0, dtype=torch.float32)


class _ConstantRegressor(torch.nn.Module):
    """Fixed-output regressor for target normalization tests."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a constant prediction for every window.

        Args:
            features: Sequence feature batch.
            lengths: Ignored; accepted for interface compatibility.

        Returns:
            One constant prediction per window.
        """
        return torch.full(
            (features.shape[0],), self._value, dtype=torch.float32
        )


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


def test_resolve_device_auto_falls_back_to_cpu_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'auto' resolves to CPU (no error) when no GPU is visible."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto").type == "cpu"


def test_resolve_device_auto_selects_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'auto' resolves to CUDA when a GPU is visible."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto").type == "cuda"


def test_predict_windows_returns_float64_prediction_per_window() -> None:
    """Window prediction returns one float64 numpy value per input window."""
    model = GRURULRegressor(input_size=2, hidden_size=4, num_layers=1, dropout=0.0)

    predictions = predict_windows(model, _loader(), torch.device("cpu"), max_rul=1)

    assert predictions.dtype == np.float64
    assert predictions.shape == (4,)


def test_evaluate_loader_keeps_predictions_and_targets_paired() -> None:
    """Evaluation pairs predictions and targets from the same loader pass."""
    targets = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    features = targets.reshape(-1, 1, 1)
    lengths = torch.full((4,), 1, dtype=torch.int64)
    dataset = TensorDataset(features, targets, lengths)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=_AlternatingOrderSampler(len(dataset)),
    )
    model = _LastFeatureRegressor()

    metrics = _evaluate_loader(model, loader, torch.device("cpu"), max_rul=1)

    assert metrics["rmse"] == 0.0


def test_evaluate_loader_clips_negative_predictions_before_metrics() -> None:
    """Evaluation clips impossible negative RUL predictions before metrics."""
    targets = torch.tensor([2.0, 4.0], dtype=torch.float32)
    features = torch.zeros((2, 1, 1), dtype=torch.float32)
    lengths = torch.full((2,), 1, dtype=torch.int64)
    loader = DataLoader(TensorDataset(features, targets, lengths), batch_size=2)
    model = _NegativeRegressor()

    metrics = _evaluate_loader(model, loader, torch.device("cpu"), max_rul=1)

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
        validation_windows_loader=_loader(),
        config=config,
        device=torch.device("cpu"),
        random_seed=7,
        max_rul=1,
    )

    assert isinstance(result, TrainingResult)
    assert result.model is model
    assert not result.history.empty
    assert list(result.history.columns) == EXPECTED_HISTORY_COLUMNS
    assert result.best_epoch >= 1
    assert result.best_metric >= 0.0


def test_train_gru_model_restores_best_state_after_early_stopping() -> None:
    """Training restores best state selected by validation-window RMSE."""
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
        validation_windows_loader=validation_windows_loader,
        config=config,
        device=torch.device("cpu"),
        random_seed=7,
        max_rul=1,
    )
    restored_predictions = predict_windows(
        model,
        validation_windows_loader,
        torch.device("cpu"),
        max_rul=1,
    )

    assert len(result.history) == 3
    assert result.best_epoch == 2
    assert result.best_metric == pytest.approx(
        result.history["validation_windows_rmse"].iloc[1]
    )
    assert np.sqrt(np.mean((restored_predictions - 0.2) ** 2)) == pytest.approx(
        result.best_metric,
        abs=1e-7,
    )



def test_evaluate_loader_rescales_predictions_by_max_rul() -> None:
    """Evaluate loader multiplies raw predictions by max_rul before clipping."""
    targets = torch.tensor([10.0, 10.0], dtype=torch.float32)
    features = torch.zeros((2, 1, 1), dtype=torch.float32)
    lengths = torch.full((2,), 1, dtype=torch.int64)
    loader = DataLoader(TensorDataset(features, targets, lengths), batch_size=2)
    model = _ConstantRegressor(0.1)
    device = torch.device("cpu")

    # With max_rul=100: predictions = 0.1 * 100 = 10.0, matches targets → rmse=0
    metrics_scaled = _evaluate_loader(model, loader, device, max_rul=100)
    # With max_rul=1: predictions = 0.1 * 1 = 0.1, targets=10.0 → rmse=9.9
    metrics_identity = _evaluate_loader(model, loader, device, max_rul=1)

    assert metrics_scaled["rmse"] == pytest.approx(0.0, abs=1e-5)
    assert metrics_identity["rmse"] == pytest.approx(9.9, abs=1e-5)


def test_evaluate_loader_returns_normalized_validation_loss() -> None:
    """Evaluate loader reports MSE on the normalized scale as ``loss``.

    The loss must use the same normalized [0, 1] target scale as the training
    criterion (predictions vs ``targets / max_rul``), unclipped, so it is
    directly comparable to ``train_loss``.
    """
    targets = torch.tensor([10.0, 10.0], dtype=torch.float32)
    features = torch.zeros((2, 1, 1), dtype=torch.float32)
    lengths = torch.full((2,), 1, dtype=torch.int64)
    loader = DataLoader(TensorDataset(features, targets, lengths), batch_size=2)
    model = _ConstantRegressor(0.1)
    device = torch.device("cpu")

    # Normalized target = 10/100 = 0.1, prediction = 0.1 -> loss = 0.
    metrics_scaled = _evaluate_loader(model, loader, device, max_rul=100)
    # Normalized target = 10/1 = 10, prediction = 0.1 -> loss = (0.1 - 10)^2.
    metrics_identity = _evaluate_loader(model, loader, device, max_rul=1)

    assert metrics_scaled["loss"] == pytest.approx(0.0, abs=1e-6)
    assert metrics_identity["loss"] == pytest.approx((0.1 - 10.0) ** 2, abs=1e-4)


def test_predict_windows_rescales_by_max_rul() -> None:
    """predict_windows multiplies raw model output by max_rul before returning."""
    model = GRURULRegressor(input_size=2, hidden_size=4, num_layers=1, dropout=0.0)
    device = torch.device("cpu")
    loader = _loader()

    preds_identity = predict_windows(model, loader, device, max_rul=1)
    preds_scaled = predict_windows(model, loader, device, max_rul=10)

    np.testing.assert_allclose(preds_scaled, preds_identity * 10, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3-tuple length threading tests
# ---------------------------------------------------------------------------


def _tiny_windows(n: int = 4, window: int = 6, features: int = 3) -> WindowedSequences:
    """Build a tiny deterministic WindowedSequences with varied lengths.

    Args:
        n: Number of windows.
        window: Padded window size.
        features: Number of feature columns.

    Returns:
        WindowedSequences with left-zero-padded short windows.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, window, features)).astype(np.float32)
    y = rng.uniform(20.0, 120.0, size=(n,)).astype(np.float32)
    lengths = np.asarray([window, window - 1, window, window - 2], dtype=np.int64)
    # Zero out padded prefixes to mimic the real padded format
    for i, length in enumerate(lengths):
        pad = window - int(length)
        if pad:
            X[i, :pad, :] = 0.0
    metadata = pd.DataFrame(
        {
            "engine_id": list(range(n)),
            "cycle": [window] * n,
            "padded": [length < window for length in lengths.tolist()],
        }
    )
    return WindowedSequences(X=X, y=y, metadata=metadata, lengths=lengths)


def test_train_one_epoch_handles_three_tuple_batches() -> None:
    """Training loop unpacks 3-tuple batches and passes lengths to the model."""
    torch.manual_seed(0)
    model = GRURULRegressor(input_size=3, hidden_size=4, num_layers=1, dropout=0.0)
    loader = build_sequence_loader(_tiny_windows(), batch_size=2, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    loss = _train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device=torch.device("cpu"),
        max_rul=125,
    )
    assert np.isfinite(loss)


def test_evaluate_loader_returns_finite_metrics() -> None:
    """Evaluation loop unpacks 3-tuple batches and returns finite metrics."""
    torch.manual_seed(0)
    model = GRURULRegressor(input_size=3, hidden_size=4, num_layers=1, dropout=0.0)
    loader = build_sequence_loader(_tiny_windows(), batch_size=2, shuffle=False)
    metrics = _evaluate_loader(
        model, loader, device=torch.device("cpu"), max_rul=125
    )
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mae"])
    assert "phm08_score" not in metrics
