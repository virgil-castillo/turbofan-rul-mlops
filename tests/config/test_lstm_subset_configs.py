"""Tests that the committed LSTM subset configs load and resolve correctly."""
from __future__ import annotations

from pathlib import Path

import pytest

from turbofan.config.schema import load_config

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
_LSTM_SEQUENCE_WINDOWS = {
    "FD001": 60,
    "FD002": 30,
    "FD003": 60,
    "FD004": 30,
}
_LSTM_FEATURES = {
    "FD001": (["raw", "rolling_slope"], [20]),
    "FD002": (["raw", "rolling_slope"], [20]),
    "FD003": (["raw", "rolling_delta"], [20]),
    "FD004": (["raw", "rolling_mean"], [20]),
}


@pytest.mark.parametrize("subset", _SUBSETS)
def test_lstm_subset_config_selects_lstm_and_current_hyperparameters(
    subset: str,
) -> None:
    """Each fd00X_lstm.yaml selects LSTM and current sequence settings."""
    lstm_path = (
        _PROJECT_ROOT / "configs" / "subsets" / f"{subset.lower()}_lstm.yaml"
    )

    lstm_cfg = load_config(lstm_path)

    assert lstm_cfg.sequence.architecture == "lstm"
    assert lstm_cfg.data.fd_subset == subset
    assert lstm_cfg.sequence.window_size == _LSTM_SEQUENCE_WINDOWS[subset]
    assert lstm_cfg.sequence.hidden_size == 64
    assert lstm_cfg.sequence.learning_rate == 0.001


@pytest.mark.parametrize("subset", _SUBSETS)
def test_lstm_subset_config_resolves_current_lstm_features(subset: str) -> None:
    """The LSTM feature block matches the latest feature-family screen."""
    lstm_path = (
        _PROJECT_ROOT / "configs" / "subsets" / f"{subset.lower()}_lstm.yaml"
    )
    cfg = load_config(lstm_path)

    lstm_features = cfg.features.for_model("lstm")
    expected_families, expected_windows = _LSTM_FEATURES[subset]

    assert lstm_features.feature_families == expected_families
    assert lstm_features.windows == expected_windows
