"""GRU-based regressors for sequence RUL prediction."""
from __future__ import annotations

from typing import cast

import torch
from torch import nn


class GRURULRegressor(nn.Module):
    """Predict remaining useful life from fixed-length feature sequences.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Number of hidden units in each GRU layer.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability between GRU layers.

    Raises:
        ValueError: If any constructor value is outside the supported range.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the GRU encoder and scalar regression head.

        Args:
            input_size: Number of input features per timestep.
            hidden_size: Number of hidden units in each GRU layer.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability between GRU layers.

        Raises:
            ValueError: If any constructor value is outside the supported range.
        """
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be at least 0")
        if dropout >= 1.0:
            raise ValueError("dropout must be less than 1")

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(
        self,
        X: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a sequence batch through the GRU regressor.

        Args:
            X: Input tensor with shape ``(batch_size, sequence_length,
                input_size)``. Left-zero-padded entries are tolerated when
                ``lengths`` is provided.
            lengths: Optional actual timestep counts per row, shape
                ``(batch_size,)``. When supplied, the GRU runs over a packed
                sequence so the final hidden state reflects only real
                timesteps. When ``None``, behaviour matches the unpacked
                path.

        Returns:
            Tensor with shape ``(batch_size,)`` containing RUL predictions.
        """
        if lengths is None:
            _, hidden = self.gru(X)
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                X,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
        final_hidden = hidden[-1]
        predictions = self.regressor(final_hidden)
        return cast(torch.Tensor, predictions.squeeze(dim=-1))
