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


# ── new rolling families accepted by schema ───────────────────────────────────


def test_model_feature_config_accepts_rolling_std() -> None:
    """rolling_std is a valid FeatureSetName for ModelFeatureConfig."""
    cfg = ModelFeatureConfig(feature_set="rolling_std")  # type: ignore[arg-type]
    assert cfg.feature_set == "rolling_std"


def test_model_feature_config_accepts_rolling_slope() -> None:
    """rolling_slope is a valid FeatureSetName for ModelFeatureConfig."""
    cfg = ModelFeatureConfig(feature_set="rolling_slope")  # type: ignore[arg-type]
    assert cfg.feature_set == "rolling_slope"


def test_model_feature_config_accepts_rolling_delta() -> None:
    """rolling_delta is a valid FeatureSetName for ModelFeatureConfig."""
    cfg = ModelFeatureConfig(feature_set="rolling_delta")  # type: ignore[arg-type]
    assert cfg.feature_set == "rolling_delta"


def test_feature_config_for_model_rolling_std_gru() -> None:
    """rolling_std resolves correctly through FeatureConfig.for_model for gru."""
    fc = FeatureConfig(
        gru=ModelFeatureConfig(feature_set="rolling_std"),  # type: ignore[arg-type]
    )
    resolved = fc.for_model("gru")
    assert resolved.feature_set == "rolling_std"


def test_feature_config_for_model_rolling_slope_ridge() -> None:
    """rolling_slope resolves correctly through FeatureConfig.for_model for ridge."""
    fc = FeatureConfig(
        ridge=ModelFeatureConfig(feature_set="rolling_slope"),  # type: ignore[arg-type]
    )
    resolved = fc.for_model("ridge")
    assert resolved.feature_set == "rolling_slope"


def test_feature_config_for_model_rolling_delta_lstm() -> None:
    """rolling_delta resolves correctly through FeatureConfig.for_model for lstm."""
    fc = FeatureConfig(
        lstm=ModelFeatureConfig(feature_set="rolling_delta"),  # type: ignore[arg-type]
    )
    resolved = fc.for_model("lstm")
    assert resolved.feature_set == "rolling_delta"
