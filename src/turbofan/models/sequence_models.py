"""Shared sequence RUL regressor and an RNN architecture registry.

The :class:`SequenceRULRegressor` owns the structure shared by every recurrent
RUL model: a recurrent encoder, a scalar regression head, the
``pack_padded_sequence`` path, and final-hidden-state extraction. The recurrent
layer (``nn.GRU`` or ``nn.LSTM``) is selected by the ``architecture`` argument.
The single behavioural difference between the two — GRU's ``forward`` returns a
hidden tensor while LSTM's returns a ``(hidden, cell)`` tuple — is normalized
inside :meth:`SequenceRULRegressor.forward`.

The :data:`SEQUENCE_ARCHITECTURES` mapping and :func:`build_sequence_model` form
the single seam through which new RNN architectures register. The contract is
deliberately RNN-scoped; it will be widened when a non-RNN architecture lands
with real requirements in hand.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import nn

#: Recurrent-layer builders keyed by architecture name. Each builder returns an
#: ``nn.Module`` whose ``__call__`` accepts a (possibly packed) batch and yields
#: ``output, hidden_state`` where ``hidden_state`` is either the hidden tensor
#: (GRU) or a ``(hidden, cell)`` tuple (LSTM).
_RNN_BUILDERS: dict[str, Callable[[int, int, int, float], nn.Module]] = {
    "gru": lambda input_size, hidden_size, num_layers, dropout: nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_first=True,
    ),
    "lstm": lambda input_size, hidden_size, num_layers, dropout: nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_first=True,
    ),
}

#: Supported RNN architecture names, in registration order.
SEQUENCE_ARCHITECTURES: tuple[str, ...] = tuple(_RNN_BUILDERS)


class SequenceRULRegressor(nn.Module):
    """Predict remaining useful life from fixed-length feature sequences.

    The recurrent layer is chosen by ``architecture`` and stored under an
    attribute named after that architecture (``gru`` or ``lstm``), so the
    state-dict key names match the per-architecture standalone modules and
    previously registered checkpoints remain loadable.

    Args:
        architecture: Recurrent architecture name; one of
            :data:`SEQUENCE_ARCHITECTURES`.
        input_size: Number of input features per timestep.
        hidden_size: Number of hidden units in each recurrent layer.
        num_layers: Number of stacked recurrent layers.
        dropout: Dropout probability between recurrent layers.

    Raises:
        ValueError: If ``architecture`` is unsupported or any constructor value
            is outside the supported range.
    """

    def __init__(
        self,
        architecture: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the recurrent encoder and scalar regression head.

        Args:
            architecture: Recurrent architecture name; one of
                :data:`SEQUENCE_ARCHITECTURES`.
            input_size: Number of input features per timestep.
            hidden_size: Number of hidden units in each recurrent layer.
            num_layers: Number of stacked recurrent layers.
            dropout: Dropout probability between recurrent layers.

        Raises:
            ValueError: If ``architecture`` is unsupported or any constructor
                value is outside the supported range.
        """
        super().__init__()
        if architecture not in _RNN_BUILDERS:
            supported = ", ".join(SEQUENCE_ARCHITECTURES)
            raise ValueError(
                f"Unknown sequence architecture {architecture!r}; "
                f"supported architectures are: {supported}."
            )
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

        self.architecture = architecture
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn = _RNN_BUILDERS[architecture](
            input_size, hidden_size, num_layers, rnn_dropout
        )
        # Store the recurrent layer under an architecture-named attribute so the
        # state-dict keys (e.g. ``gru.weight_ih_l0``) match the standalone
        # modules and existing checkpoints load unchanged.
        setattr(self, architecture, rnn)
        self.regressor = nn.Linear(hidden_size, 1)

    @property
    def _rnn(self) -> nn.Module:
        """Return the recurrent layer regardless of its architecture name.

        Returns:
            The recurrent submodule (``nn.GRU`` or ``nn.LSTM``).
        """
        return cast(nn.Module, getattr(self, self.architecture))

    def forward(
        self,
        X: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a sequence batch through the recurrent regressor.

        Args:
            X: Input tensor with shape ``(batch_size, sequence_length,
                input_size)``. Right-zero-padded entries are tolerated when
                ``lengths`` is provided.
            lengths: Optional actual timestep counts per row, shape
                ``(batch_size,)``. When supplied, the recurrent layer runs over
                a packed sequence so the final hidden state reflects only real
                timesteps. When ``None``, behaviour matches the unpacked path.

        Returns:
            Tensor with shape ``(batch_size,)`` containing RUL predictions.
        """
        if lengths is None:
            _, hidden_state = self._rnn(X)
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                X,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden_state = self._rnn(packed)
        # GRU returns the hidden tensor; LSTM returns a (hidden, cell) tuple.
        # Normalize to the hidden tensor, then take the last layer's state.
        hidden = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
        final_hidden = hidden[-1]
        predictions = self.regressor(final_hidden)
        return cast(torch.Tensor, predictions.squeeze(dim=-1))


def build_sequence_model(
    architecture: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> SequenceRULRegressor:
    """Build a sequence regressor for a registered RNN architecture.

    This is the single seam through which architecture names resolve to a
    constructed module. New RNN architectures register by adding a builder to
    :data:`SEQUENCE_ARCHITECTURES`.

    Args:
        architecture: Recurrent architecture name; one of
            :data:`SEQUENCE_ARCHITECTURES`.
        input_size: Number of input features per timestep.
        hidden_size: Number of hidden units in each recurrent layer.
        num_layers: Number of stacked recurrent layers.
        dropout: Dropout probability between recurrent layers.

    Returns:
        A constructed :class:`SequenceRULRegressor` for the requested
        architecture.

    Raises:
        ValueError: If ``architecture`` is unsupported or any hyperparameter is
            outside the supported range.
    """
    if architecture not in _RNN_BUILDERS:
        supported = ", ".join(SEQUENCE_ARCHITECTURES)
        raise ValueError(
            f"Unknown sequence architecture {architecture!r}; "
            f"supported architectures are: {supported}."
        )
    return SequenceRULRegressor(
        architecture=architecture,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
