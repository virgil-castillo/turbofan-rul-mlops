"""Tests for per-model feature resolution in ``FeatureConfig``."""
from __future__ import annotations

from turbofan.config.schema import FeatureConfig, ModelFeatureConfig


def test_for_model_returns_shared_defaults_without_overrides() -> None:
    """With no override blocks, both models receive the shared settings."""
    fc = FeatureConfig(feature_set="rolling_mean", windows=[10], lag_steps=[1])

    for model in ("ridge", "gru"):
        resolved = fc.for_model(model)  # type: ignore[arg-type]
        assert resolved.feature_set == "rolling_mean"
        assert resolved.windows == [10]
        assert resolved.lag_steps == [1]


def test_for_model_applies_full_override() -> None:
    """A fully specified override replaces every shared field for that model."""
    fc = FeatureConfig(
        feature_set="rolling_mean",
        windows=[10],
        lag_steps=[1],
        ridge=ModelFeatureConfig(feature_set="raw_plus_rolling_mean", windows=[2]),
        gru=ModelFeatureConfig(feature_set="rolling_mean", windows=[15]),
    )

    ridge = fc.for_model("ridge")
    assert ridge.feature_set == "raw_plus_rolling_mean"
    assert ridge.windows == [2]

    gru = fc.for_model("gru")
    assert gru.feature_set == "rolling_mean"
    assert gru.windows == [15]


def test_for_model_partial_override_inherits_unset_fields() -> None:
    """Fields left unset on an override inherit the shared value."""
    fc = FeatureConfig(
        feature_set="rolling_mean",
        windows=[10],
        lag_steps=[3],
        gru=ModelFeatureConfig(windows=[15]),
    )

    gru = fc.for_model("gru")
    assert gru.windows == [15]
    assert gru.feature_set == "rolling_mean"
    assert gru.lag_steps == [3]


def test_for_model_override_does_not_leak_across_models() -> None:
    """A Ridge override leaves the GRU resolution on shared defaults."""
    fc = FeatureConfig(
        feature_set="rolling_mean",
        windows=[10],
        ridge=ModelFeatureConfig(feature_set="raw_plus_rolling_mean", windows=[8]),
    )

    assert fc.for_model("gru").feature_set == "rolling_mean"
    assert fc.for_model("gru").windows == [10]
