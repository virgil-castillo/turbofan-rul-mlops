"""Tests that weight_decay flows from config into the training optimizer."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from turbofan.config.schema import SequenceConfig
from turbofan.models.sequence_models import build_sequence_model
from turbofan.training.sequence_training import train_sequence_model


def _loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build a tiny deterministic sequence DataLoader.

    Returns:
        DataLoader with two single-step windows, scalar targets, and lengths.
    """
    features = torch.zeros((2, 1, 1), dtype=torch.float32)
    targets = torch.ones(2, dtype=torch.float32)
    lengths = torch.full((2,), 1, dtype=torch.int64)
    return DataLoader(TensorDataset(features, targets, lengths), batch_size=2)


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_weight_decay_passed_to_adam_optimizer(
    architecture: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_sequence_model builds Adam with the configured weight_decay.

    Spies on ``torch.optim.Adam`` to capture the ``weight_decay`` it is
    constructed with, confirming the configured value reaches the optimizer for
    both recurrent architectures rather than being silently dropped.
    """
    captured: dict[str, float] = {}
    real_adam = torch.optim.Adam

    def spy_adam(params: object, **kwargs: object) -> torch.optim.Adam:
        captured["weight_decay"] = float(kwargs.get("weight_decay", 0.0))
        return real_adam(params, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.optim, "Adam", spy_adam)

    model = build_sequence_model(
        architecture, input_size=1, hidden_size=2, num_layers=1, dropout=0.0
    )
    config = SequenceConfig(
        architecture=architecture,
        batch_size=2,
        hidden_size=2,
        epochs=1,
        patience=1,
        learning_rate=0.01,
        weight_decay=5e-4,
    )

    train_sequence_model(
        model=model,
        train_loader=_loader(),
        validation_windows_loader=_loader(),
        config=config,
        device=torch.device("cpu"),
        random_seed=0,
        max_rul=1,
    )

    assert captured["weight_decay"] == pytest.approx(5e-4)


def test_weight_decay_defaults_to_zero_in_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no weight_decay set, the optimizer receives 0.0 (unchanged behavior)."""
    captured: dict[str, float] = {}
    real_adam = torch.optim.Adam

    def spy_adam(params: object, **kwargs: object) -> torch.optim.Adam:
        captured["weight_decay"] = float(kwargs.get("weight_decay", 0.0))
        return real_adam(params, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.optim, "Adam", spy_adam)

    model = build_sequence_model(
        "gru", input_size=1, hidden_size=2, num_layers=1, dropout=0.0
    )
    config = SequenceConfig(
        architecture="gru",
        batch_size=2,
        hidden_size=2,
        epochs=1,
        patience=1,
        learning_rate=0.01,
    )

    train_sequence_model(
        model=model,
        train_loader=_loader(),
        validation_windows_loader=_loader(),
        config=config,
        device=torch.device("cpu"),
        random_seed=0,
        max_rul=1,
    )

    assert captured["weight_decay"] == 0.0
