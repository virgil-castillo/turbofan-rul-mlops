"""Tests that the committed LSTM subset configs load and resolve correctly."""
from __future__ import annotations

from pathlib import Path

import pytest

from turbofan.config.schema import load_config

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


@pytest.mark.parametrize("subset", _SUBSETS)
def test_lstm_subset_config_selects_lstm_and_inherits_hyperparameters(
    subset: str,
) -> None:
    """Each fd00X_lstm.yaml selects LSTM while inheriting the GRU subset config.

    The LSTM variant must override only the architecture: the data subset and
    the shared sequence hyperparameters (window size, hidden size) come straight
    from the matching GRU subset config.
    """
    lstm_path = (
        _PROJECT_ROOT / "configs" / "subsets" / f"{subset.lower()}_lstm.yaml"
    )
    gru_path = _PROJECT_ROOT / "configs" / "subsets" / f"{subset.lower()}.yaml"

    lstm_cfg = load_config(lstm_path)
    gru_cfg = load_config(gru_path)

    assert lstm_cfg.sequence.architecture == "lstm"
    assert lstm_cfg.data.fd_subset == subset
    assert lstm_cfg.sequence.window_size == gru_cfg.sequence.window_size
    assert lstm_cfg.sequence.hidden_size == gru_cfg.sequence.hidden_size
    assert lstm_cfg.sequence.learning_rate == gru_cfg.sequence.learning_rate


@pytest.mark.parametrize("subset", _SUBSETS)
def test_lstm_subset_config_resolves_seeded_lstm_features(subset: str) -> None:
    """The LSTM feature block is seeded from the subset's GRU feature settings."""
    lstm_path = (
        _PROJECT_ROOT / "configs" / "subsets" / f"{subset.lower()}_lstm.yaml"
    )
    cfg = load_config(lstm_path)

    lstm_features = cfg.features.for_model("lstm")
    gru_features = cfg.features.for_model("gru")

    assert lstm_features.feature_set == gru_features.feature_set
    assert lstm_features.windows == gru_features.windows
