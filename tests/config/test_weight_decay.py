"""Tests for the shared weight_decay sequence hyperparameter."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from turbofan.config.schema import SequenceConfig


def test_weight_decay_defaults_to_zero() -> None:
    """weight_decay defaults to 0.0 so existing runs are unregularized."""
    config = SequenceConfig()

    assert config.weight_decay == 0.0


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_weight_decay_is_shared_across_architectures(architecture: str) -> None:
    """weight_decay applies to both GRU and LSTM via the shared config."""
    config = SequenceConfig(architecture=architecture, weight_decay=1e-4)

    assert config.weight_decay == pytest.approx(1e-4)


def test_weight_decay_rejects_negative_values() -> None:
    """A negative weight_decay is rejected by validation."""
    with pytest.raises(ValidationError):
        SequenceConfig(weight_decay=-0.1)
