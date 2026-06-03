"""Training helpers for sequence RUL models (architecture-agnostic)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from turbofan.config.schema import SequenceConfig
from turbofan.models.metrics import regression_metrics

type SequenceBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
type SequenceLoader = DataLoader[SequenceBatch]


@dataclass(frozen=True)
class TrainingResult:
    """Result from sequence model training.

    Args:
        model: Trained model restored to the best validation epoch.
        history: Per-epoch training loss and validation metrics.
        best_epoch: One-indexed epoch with the best validation-window RMSE.
        best_metric: Best validation-window RMSE.
    """

    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_metric: float


def resolve_device(requested: Literal["cpu", "cuda"] = "cpu") -> torch.device:
    """Resolve a requested torch device.

    Args:
        requested: Requested device name.

    Returns:
        Torch device for CPU or available CUDA.

    Raises:
        ValueError: If CUDA is requested but unavailable.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise ValueError("CUDA requested but not available.")


def train_sequence_model(
    model: nn.Module,
    train_loader: SequenceLoader,
    validation_windows_loader: SequenceLoader,
    config: SequenceConfig,
    device: torch.device,
    random_seed: int,
    max_rul: int,
) -> TrainingResult:
    """Train a sequence RUL regressor with validation metrics and early stopping.

    The loop is architecture-agnostic: it depends only on the
    ``(features, targets, lengths)`` batch contract and the model's
    ``(batch_size,)`` forward output, so it trains any registered RNN (GRU or
    LSTM) without change.

    Args:
        model: Unfitted sequence model (e.g. a :class:`SequenceRULRegressor`).
        train_loader: Mini-batch loader for training windows.
        validation_windows_loader: Evaluation loader for all validation windows.
        config: Sequence model training configuration.
        device: Torch device used for training and evaluation.
        random_seed: Seed for Python, NumPy, and torch random generators.
        max_rul: Maximum RUL used to normalise targets during training and
            rescale predictions during evaluation.

    Returns:
        Training result containing the best restored model and metric history.
    """
    seed_everything(random_seed)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_metric = float("inf")
    best_state = _clone_state_dict(model)
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_rul
        )
        window_metrics = _evaluate_loader(
            model, validation_windows_loader, device, max_rul
        )
        current_metric = window_metrics["rmse"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": window_metrics["loss"],
                "validation_windows_rmse": window_metrics["rmse"],
                "validation_windows_mae": window_metrics["mae"],
            }
        )

        if current_metric < best_metric:
            best_epoch = epoch
            best_metric = current_metric
            best_state = _clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_metric=best_metric,
    )


def train_gru_model(
    model: nn.Module,
    train_loader: SequenceLoader,
    validation_windows_loader: SequenceLoader,
    config: SequenceConfig,
    device: torch.device,
    random_seed: int,
    max_rul: int,
) -> TrainingResult:
    """Backward-compatible alias for :func:`train_sequence_model`.

    Retained so existing GRU call sites and imports keep working; delegates
    unchanged to the generalized training entrypoint.

    Args:
        model: Unfitted sequence model.
        train_loader: Mini-batch loader for training windows.
        validation_windows_loader: Evaluation loader for all validation windows.
        config: Sequence model training configuration.
        device: Torch device used for training and evaluation.
        random_seed: Seed for Python, NumPy, and torch random generators.
        max_rul: Maximum RUL used to normalise targets and rescale predictions.

    Returns:
        Training result containing the best restored model and metric history.
    """
    return train_sequence_model(
        model=model,
        train_loader=train_loader,
        validation_windows_loader=validation_windows_loader,
        config=config,
        device=device,
        random_seed=random_seed,
        max_rul=max_rul,
    )


def predict_windows(
    model: nn.Module,
    loader: SequenceLoader,
    device: torch.device,
    max_rul: int,
) -> npt.NDArray[np.float64]:
    """Predict RUL values for sequence windows.

    Args:
        model: Trained sequence model.
        loader: Loader containing sequence feature batches.
        device: Torch device used for inference.
        max_rul: Maximum RUL used to rescale normalised model outputs back to
            the original RUL scale.

    Returns:
        One-dimensional float64 array with one prediction per input window,
        rescaled by ``max_rul``.
    """
    predictions, _ = _predict_windows_and_targets(model, loader, device)
    return predictions * max_rul


def _predict_windows_and_targets(
    model: nn.Module,
    loader: SequenceLoader,
    device: torch.device,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    was_training = model.training
    model.eval()
    predictions: list[npt.NDArray[np.float64]] = []
    targets: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for features, batch_targets, lengths in loader:
            batch_predictions = model(features.to(device), lengths=lengths)
            predictions.append(
                batch_predictions.detach().cpu().numpy().astype(np.float64)
            )
            targets.append(batch_targets.detach().cpu().numpy().astype(np.float64))
    if was_training:
        model.train()
    if not predictions:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    return (
        np.concatenate(predictions).astype(np.float64),
        np.concatenate(targets).astype(np.float64),
    )


def seed_everything(random_seed: int) -> None:
    """Seed Python, NumPy, and torch random generators.

    Args:
        random_seed: Seed applied to all supported random generators.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _train_one_epoch(
    model: nn.Module,
    loader: SequenceLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_rul: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, targets, lengths in loader:
        features = features.to(device)
        targets = targets.to(device) / max_rul
        optimizer.zero_grad()
        predictions = model(features, lengths=lengths)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size
    if total_count == 0:
        return 0.0
    return total_loss / total_count


def _evaluate_loader(
    model: nn.Module,
    loader: SequenceLoader,
    device: torch.device,
    max_rul: int,
) -> dict[str, float]:
    """Evaluate a loader, returning reporting metrics plus the validation loss.

    The returned ``loss`` is the MSE on the normalized [0, 1] target scale
    (predictions vs ``targets / max_rul``), unclipped — matching the training
    criterion so ``train_loss`` and ``val_loss`` are directly comparable.
    ``rmse``/``mae`` remain on the rescaled, clipped RUL (cycle) scale for
    human-readable reporting.

    Args:
        model: Model to evaluate.
        loader: Evaluation loader yielding feature/target/length batches.
        device: Torch device used for inference.
        max_rul: Maximum RUL used to normalize targets and rescale predictions.

    Returns:
        Mapping with ``rmse``, ``mae`` (cycle scale) and ``loss`` (normalized
        MSE).
    """
    predictions, targets = _predict_windows_and_targets(model, loader, device)
    if predictions.size:
        normalized_targets = targets / max_rul
        loss = float(np.mean((predictions - normalized_targets) ** 2))
    else:
        loss = 0.0
    rescaled = np.clip(predictions * max_rul, 0.0, None)
    metrics = regression_metrics(targets, rescaled)
    metrics["loss"] = loss
    return metrics
