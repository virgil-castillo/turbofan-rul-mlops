"""Tests for turbofan.models.baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.models.baseline import build_baseline_pipeline


def _make_df() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic turbofan rows and RUL labels."""
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    for engine_id in [1, 2, 3]:
        for cycle in range(1, 8):
            rows.append({
                "engine_id": engine_id,
                "cycle": cycle,
                "op_1": 0.0,
                "op_2": 0.0,
                "op_3": 0.0,
                "s_1": float(cycle) + rng.normal(0.0, 0.1),
                "s_2": 200.0,
                "s_3": float(engine_id) + rng.normal(0.0, 0.1),
            })
            labels.append(float(8 - cycle))
    return pd.DataFrame(rows), pd.Series(labels, name="rul")


def test_build_baseline_pipeline_returns_two_step_pipeline() -> None:
    """Baseline pipeline has exactly two steps: features and model."""
    pipe = build_baseline_pipeline()
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["features", "model"]


def test_default_model_is_ridge() -> None:
    """The default baseline estimator is Ridge."""
    pipe = build_baseline_pipeline()
    assert isinstance(pipe.named_steps["model"], Ridge)


def test_configures_ridge_alpha() -> None:
    """Ridge alpha is passed into the estimator."""
    pipe = build_baseline_pipeline(alpha=2.5)
    assert pipe.named_steps["model"].alpha == 2.5


def test_default_ridge_alpha() -> None:
    """Default Ridge alpha is 100.0."""
    assert build_baseline_pipeline().named_steps["model"].alpha == 100.0


def test_features_step_is_pipeline() -> None:
    """The features step is itself a Pipeline with expected steps."""
    pipe = build_baseline_pipeline()
    features = pipe.named_steps["features"]
    assert isinstance(features, Pipeline)
    assert list(features.named_steps) == [
        "sensor_dropper",
        "normalizer",
        "sensor_selector",
        "feature_engineer",
    ]


def test_configures_sensor_drop() -> None:
    """sensor_drop is forwarded into the feature pipeline's sensor_dropper."""
    pipe = build_baseline_pipeline(sensor_drop=["s_1", "s_5"])
    dropper = pipe.named_steps["features"].named_steps["sensor_dropper"]
    assert dropper.drop == ["s_1", "s_5"]


def test_pipeline_can_fit_and_predict() -> None:
    """Synthetic data can be fit and predicted without NaNs."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(feature_set="raw")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_rolling_mean_feature_set() -> None:
    """feature_set=rolling_mean produces rolling mean columns for Ridge."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(feature_set="rolling_mean", windows=[3])
    pipe.fit(X, y)
    assert any("_rmean_" in c for c in pipe.named_steps["model"].feature_names_in_)


def test_model_receives_dataframe_feature_names() -> None:
    """Ridge keeps sklearn feature_names_in_ metadata."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(feature_set="raw")
    pipe.fit(X, y)
    assert hasattr(pipe.named_steps["model"], "feature_names_in_")
    assert "engine_id" not in set(pipe.named_steps["model"].feature_names_in_)
    assert "cycle" not in set(pipe.named_steps["model"].feature_names_in_)
    assert "op_1" not in set(pipe.named_steps["model"].feature_names_in_)


def test_unknown_model_name_raises() -> None:
    """Unsupported model names fail fast."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_baseline_pipeline(model_name="random_forest")  # type: ignore[arg-type]
