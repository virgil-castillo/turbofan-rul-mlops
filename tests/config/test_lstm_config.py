"""Tests for LSTM support in config schema and feature resolution."""
from __future__ import annotations

from pathlib import Path

import yaml

from turbofan.config.schema import (
    FeatureConfig,
    ModelFeatureConfig,
    SequenceConfig,
    load_config,
)


def _write_config(tmp_path: Path, data: dict[str, object]) -> Path:
    """Write a config dict to a YAML file and return the path.

    Args:
        tmp_path: Temporary directory for the config file.
        data: Config mapping to serialize.

    Returns:
        Path to the written YAML file.
    """
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_sequence_config_accepts_lstm_architecture() -> None:
    """SequenceConfig accepts ``architecture: lstm`` directly."""
    config = SequenceConfig(architecture="lstm")

    assert config.architecture == "lstm"


def test_load_config_accepts_lstm_sequence_architecture(tmp_path: Path) -> None:
    """A YAML config selecting LSTM loads and validates without error."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "sequence": {"architecture": "lstm"},
        },
    )

    cfg = load_config(cfg_file)

    assert cfg.sequence.architecture == "lstm"


def test_for_model_lstm_returns_shared_defaults_without_override() -> None:
    """With no LSTM override block, LSTM receives the shared settings."""
    fc = FeatureConfig(feature_families=["rolling_mean"], windows=[10], lag_steps=[1])

    resolved = fc.for_model("lstm")

    assert resolved.feature_families == ["rolling_mean"]
    assert resolved.windows == [10]
    assert resolved.lag_steps == [1]


def test_for_model_lstm_applies_its_own_override() -> None:
    """The ``features.lstm`` block overrides shared settings for LSTM only."""
    fc = FeatureConfig(
        feature_families=["rolling_mean"],
        windows=[10],
        lag_steps=[1],
        gru=ModelFeatureConfig(windows=[20]),
        lstm=ModelFeatureConfig(feature_families=["raw", "rolling_mean"], windows=[5]),
    )

    lstm = fc.for_model("lstm")
    assert lstm.feature_families == ["raw", "rolling_mean"]
    assert lstm.windows == [5]

    # The LSTM override must not leak into the GRU resolution.
    gru = fc.for_model("gru")
    assert gru.feature_families == ["rolling_mean"]
    assert gru.windows == [20]


def test_for_model_lstm_partial_override_inherits_unset_fields() -> None:
    """Fields left unset on the LSTM override inherit the shared value."""
    fc = FeatureConfig(
        feature_families=["rolling_mean"],
        windows=[10],
        lag_steps=[3],
        lstm=ModelFeatureConfig(windows=[7]),
    )

    lstm = fc.for_model("lstm")
    assert lstm.windows == [7]
    assert lstm.feature_families == ["rolling_mean"]
    assert lstm.lag_steps == [3]


def test_base_composition_overrides_only_architecture(tmp_path: Path) -> None:
    """A `_base_` LSTM config overrides only ``sequence.architecture``.

    Mirrors the dedicated ``fd00X_lstm.yaml`` approach: layering an
    architecture-only override on a GRU subset config flips the architecture to
    LSTM while leaving every other sequence hyperparameter inherited from base.
    """
    base = tmp_path / "fd001.yaml"
    base.write_text(
        yaml.dump(
            {
                "project_name": "base",
                "data": {
                    "raw_dir": "data/raw",
                    "processed_dir": "data/processed",
                    "interim_dir": "data/interim",
                    "fd_subset": "FD001",
                },
                "sequence": {
                    "architecture": "gru",
                    "window_size": 60,
                    "hidden_size": 64,
                },
            }
        )
    )
    override = tmp_path / "fd001_lstm.yaml"
    override.write_text(
        yaml.dump({"_base_": "fd001.yaml", "sequence": {"architecture": "lstm"}})
    )

    cfg = load_config(override)

    assert cfg.sequence.architecture == "lstm"
    assert cfg.sequence.window_size == 60
    assert cfg.sequence.hidden_size == 64
