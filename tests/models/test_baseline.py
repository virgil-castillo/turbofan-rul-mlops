"""Tests for turbofan.models.baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.models.baseline import build_baseline_pipeline


def _fit_columns(feature_set: str) -> set[str]:
    """Return Ridge feature names for a fitted baseline feature set.

    Args:
        feature_set: Baseline feature family name.

    Returns:
        Set of fitted Ridge feature names.
    """
    X, y = _make_df()
    pipe = build_baseline_pipeline(
        windows=[3],
        feature_set=feature_set,  # type: ignore[arg-type]
    )
    pipe.fit(X, y)
    model = pipe.named_steps["model"]
    return set(model.feature_names_in_)


def _make_df() -> tuple[pd.DataFrame, pd.Series]:
    """Build model-ready synthetic turbofan rows and labels."""
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    for engine_id in [1, 2, 3]:
        for cycle in range(1, 8):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    "s_1": float(cycle) + rng.normal(0.0, 0.1),
                    "s_2": 200.0,
                    "s_3": float(engine_id) + rng.normal(0.0, 0.1),
                }
            )
            labels.append(float(8 - cycle))
    return pd.DataFrame(rows), pd.Series(labels, name="rul")


def _make_near_zero_std_df() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build train/validation rows that expose tiny rolling-feature stds."""
    train_rows = []
    labels = []
    for engine_id, offset in [(1, 0.0), (2, 1e-9)]:
        for cycle in range(1, 7):
            train_rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    "s_1": 100.0 + offset + cycle * 1e-9,
                    "s_2": float(cycle),
                }
            )
            labels.append(float(7 - cycle))

    val_rows = []
    for cycle in range(1, 7):
        val_rows.append(
            {
                "engine_id": 3,
                "cycle": cycle,
                "op_1": 0.0,
                "op_2": 0.0,
                "op_3": 0.0,
                "s_1": 100.0 + cycle * 25.0,
                "s_2": float(cycle),
            }
        )

    return (
        pd.DataFrame(train_rows),
        pd.Series(labels, name="rul"),
        pd.DataFrame(val_rows),
    )


def test_build_baseline_pipeline_named_steps() -> None:
    """Baseline pipeline exposes expected named steps."""
    pipe = build_baseline_pipeline()
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == [
        "features",
        "drop_identifiers",
        "select_model_features",
        "low_variance_filter",
        "imputer",
        "scaler",
        "model",
    ]


def test_default_model_is_ridge() -> None:
    """The default baseline estimator is Ridge."""
    pipe = build_baseline_pipeline()
    assert isinstance(pipe.named_steps["model"], Ridge)


def test_configures_ridge_alpha() -> None:
    """Ridge alpha is passed into the estimator."""
    pipe = build_baseline_pipeline(alpha=2.5)
    model = pipe.named_steps["model"]
    assert isinstance(model, Ridge)
    assert model.alpha == 2.5


def test_default_ridge_alpha_is_conservative() -> None:
    """The default Ridge alpha is conservative for engineered features."""
    model = build_baseline_pipeline().named_steps["model"]
    assert isinstance(model, Ridge)
    assert model.alpha == 100.0


def test_default_baseline_uses_final_phm08_feature_selection() -> None:
    """Default baseline uses the PHM08-selected rolling window."""
    pipe = build_baseline_pipeline()
    selector = pipe.named_steps["select_model_features"]
    rolling = pipe.named_steps["features"].named_steps["rolling_features"]

    assert selector.feature_set == "rolling"
    assert rolling.windows == [10]


def test_configures_sensor_std_threshold() -> None:
    """Sensor std threshold is passed into the feature pipeline."""
    pipe = build_baseline_pipeline(sensor_std_threshold=0.01)
    features = pipe.named_steps["features"]
    dropper = features.named_steps["sensor_dropper"]
    assert dropper.std_threshold == 0.01


def test_pipeline_can_fit_and_predict() -> None:
    """Synthetic data can fit and predict without NaNs."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(windows=[3])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_transformed_train_and_validation_features_are_finite() -> None:
    """Pre-Ridge train and validation matrices contain only finite values."""
    X, y, X_val = _make_near_zero_std_df()
    pipe = build_baseline_pipeline(windows=[3])
    pipe.fit(X, y)

    Xt_train = pipe[:-1].transform(X)
    Xt_val = pipe[:-1].transform(X_val)

    assert isinstance(Xt_train, pd.DataFrame)
    assert isinstance(Xt_val, pd.DataFrame)
    assert np.isfinite(Xt_train.to_numpy(dtype=np.float64)).all()
    assert np.isfinite(Xt_val.to_numpy(dtype=np.float64)).all()


def test_transformed_validation_features_are_not_catastrophically_large() -> None:
    """Near-zero rolling-feature stds do not explode validation features."""
    X, y, X_val = _make_near_zero_std_df()
    pipe = build_baseline_pipeline(windows=[3])
    pipe.fit(X, y)

    Xt_val = pipe[:-1].transform(X_val)

    assert np.abs(Xt_val.to_numpy(dtype=np.float64)).max() < 1e6


def test_predictions_are_finite_and_bounded_on_toy_instability_case() -> None:
    """Raw toy predictions remain finite and below catastrophic magnitudes."""
    X, y, X_val = _make_near_zero_std_df()
    pipe = build_baseline_pipeline(windows=[3])
    pipe.fit(X, y)

    preds = np.asarray(pipe.predict(X_val), dtype=np.float64)

    assert np.isfinite(preds).all()
    assert np.abs(preds).max() < 1e6


def test_model_receives_dataframe_feature_names() -> None:
    """The final estimator keeps sklearn feature_names_in_ metadata."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(windows=[3])
    pipe.fit(X, y)

    model = pipe.named_steps["model"]

    assert hasattr(model, "feature_names_in_")
    assert "engine_id" not in set(model.feature_names_in_)
    assert "cycle" not in set(model.feature_names_in_)


def test_pipeline_drops_identifier_columns_before_model() -> None:
    """Identifier columns are removed before the regressor."""
    X, y = _make_df()

    pipe = build_baseline_pipeline(windows=[3])
    pipe.fit(X, y)

    Xt = pipe[:-1].transform(X)

    forbidden_cols = {
        "engine_id",
        "unit_number",
        "rul",
        "cycle",
        "op_1",
        "op_2",
        "op_3",
    }

    assert forbidden_cols.isdisjoint(Xt.columns)
    assert "s_1" not in Xt.columns
    assert "s_1_rmean_3" in Xt.columns


def test_raw_feature_set_exposes_only_raw_sensor_features() -> None:
    """Raw feature set excludes identifiers, cycle, op columns, and rolling."""
    columns = _fit_columns("raw")

    assert {"engine_id", "cycle", "op_1", "op_2", "op_3"}.isdisjoint(columns)
    assert {"s_1", "s_3"}.issubset(columns)
    assert all("_rmean_" not in column for column in columns)


def test_raw_plus_rolling_feature_set_exposes_raw_and_rolling_features() -> None:
    """Raw plus rolling feature set keeps raw and rolling sensor features."""
    columns = _fit_columns("raw_plus_rolling")

    assert {"engine_id", "cycle", "op_1", "op_2", "op_3"}.isdisjoint(columns)
    assert "s_1" in columns
    assert "s_1_rmean_3" in columns


def test_rolling_feature_set_exposes_only_rolling_sensor_features() -> None:
    """Rolling feature set drops raw sensors and keeps rolling features."""
    columns = _fit_columns("rolling")

    assert {"engine_id", "cycle", "op_1", "op_2", "op_3"}.isdisjoint(columns)
    assert "s_1" not in columns
    assert "s_1_rmean_3" in columns


def test_unknown_feature_set_raises() -> None:
    """Unsupported feature-set names fail fast."""
    with pytest.raises(ValueError, match="Unsupported feature_set"):
        build_baseline_pipeline(feature_set="bad")  # type: ignore[arg-type]


def test_unknown_model_name_raises() -> None:
    """Unsupported model names fail fast."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_baseline_pipeline(model_name="random_forest")  # type: ignore[arg-type]
