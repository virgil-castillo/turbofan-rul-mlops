"""Training helpers for GRU sequence RUL models."""
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
from turbofan.models.gru import GRURULRegressor
from turbofan.models.metrics import regression_metrics

type SequenceBatch = tuple[torch.Tensor, torch.Tensor]
type SequenceLoader = DataLoader[SequenceBatch]


@dataclass(frozen=True)
class TrainingResult:
    """Result from GRU sequence model training.

    Args:
        model: Trained model restored to the best validation epoch.
        history: Per-epoch training loss and validation metrics.
        best_epoch: One-indexed epoch with the best validation-window RMSE.
        best_metric: Best validation-window RMSE.
    """

    model: GRURULRegressor
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


def train_gru_model(
    model: GRURULRegressor,
    train_loader: SequenceLoader,
    validation_final_loader: SequenceLoader,
    validation_windows_loader: SequenceLoader,
    config: SequenceConfig,
    device: torch.device,
    random_seed: int,
) -> TrainingResult:
    """Train a GRU RUL regressor with validation metrics and early stopping.

    Args:
        model: Unfitted GRU model.
        train_loader: Mini-batch loader for training windows.
        validation_final_loader: Evaluation loader for final validation windows.
        validation_windows_loader: Evaluation loader for all validation windows.
        config: Sequence model training configuration.
        device: Torch device used for training and evaluation.
        random_seed: Seed for Python, NumPy, and torch random generators.

    Returns:
        Training result containing the best restored model and metric history.
    """
    seed_everything(random_seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_metric = float("inf")
    best_state = _clone_state_dict(model)
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        final_metrics = _evaluate_loader(model, validation_final_loader, device)
        window_metrics = _evaluate_loader(model, validation_windows_loader, device)
        current_metric = window_metrics["rmse"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_final_window_rmse": final_metrics["rmse"],
                "validation_final_window_mae": final_metrics["mae"],
                "validation_final_window_phm08_score": final_metrics["phm08_score"],
                "validation_windows_rmse": window_metrics["rmse"],
                "validation_windows_mae": window_metrics["mae"],
                "validation_windows_phm08_score": window_metrics["phm08_score"],
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


def predict_windows(
    model: GRURULRegressor,
    loader: SequenceLoader,
    device: torch.device,
) -> npt.NDArray[np.float64]:
    """Predict RUL values for sequence windows.

    Args:
        model: Trained GRU model.
        loader: Loader containing sequence feature batches.
        device: Torch device used for inference.

    Returns:
        One-dimensional float64 array with one prediction per input window.
    """
    predictions, _ = _predict_windows_and_targets(model, loader, device)
    return predictions


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
        for features, batch_targets in loader:
            batch_predictions = model(features.to(device))
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


def _clone_state_dict(model: GRURULRegressor) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _train_one_epoch(
    model: GRURULRegressor,
    loader: SequenceLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predictions = model(features)
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
    model: GRURULRegressor,
    loader: SequenceLoader,
    device: torch.device,
) -> dict[str, float]:
    predictions, targets = _predict_windows_and_targets(model, loader, device)
    predictions = np.clip(predictions, 0.0, None)
    return regression_metrics(targets, predictions)
