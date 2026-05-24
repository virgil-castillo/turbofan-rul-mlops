"""Tests for turbofan.models.baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.models.baseline import build_baseline_pipeline


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


def test_build_baseline_pipeline_named_steps() -> None:
    """Baseline pipeline exposes expected named steps."""
    pipe = build_baseline_pipeline()
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["features", "drop_identifiers", "model"]


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


def test_pipeline_can_fit_and_predict() -> None:
    """Synthetic data can fit and predict without NaNs."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(windows=[3])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_pipeline_drops_engine_id_before_model() -> None:
    """Arbitrary engine identifiers are not passed into Ridge."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(windows=[3])
    pipe.fit(X, y)
    model = pipe.named_steps["model"]
    assert isinstance(model, Ridge)
    assert "engine_id" not in model.feature_names_in_
    assert "cycle" in model.feature_names_in_


def test_unknown_model_name_raises() -> None:
    """Unsupported model names fail fast."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_baseline_pipeline(model_name="random_forest")  # type: ignore[arg-type]
