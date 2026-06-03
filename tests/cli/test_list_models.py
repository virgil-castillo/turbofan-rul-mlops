"""Tests for the turbofan-models command."""
from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan import registry
from turbofan.cli import list_models
from turbofan.inference.schemas import FEATURE_COLUMNS


def _fitted_ridge_pipeline() -> Pipeline:
    """Fit a tiny self-contained Ridge pipeline on canonical raw records.

    Returns:
        A fitted sklearn ``Pipeline`` mapping canonical raw records to a target.
    """
    rows = [
        {
            "engine_id": engine_id,
            "cycle": cycle,
            **{column: float(engine_id) for column in FEATURE_COLUMNS},
        }
        for engine_id in (1, 2)
        for cycle in range(1, 5)
    ]
    frame = pd.DataFrame(rows)
    target = frame["cycle"].to_numpy(dtype=float)
    pipeline = Pipeline(
        [
            (
                "features",
                ColumnTransformer([("keep", "passthrough", FEATURE_COLUMNS)]),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(frame, target)
    return pipeline


def test_list_models_no_registered_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With an empty registry, the CLI reports that none are registered."""
    code = list_models.main([])

    assert code == 0
    out = capsys.readouterr().out
    assert "No registered models" in out


def test_list_models_reports_versions_and_production(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A registered model lists its name, versions, production alias, and metric."""
    pipeline = _fitted_ridge_pipeline()
    with mlflow.start_run():
        registry.log_and_register(pipeline, model_type="ridge", subset="FD001")
    with mlflow.start_run():
        mlflow.log_metric("val_rmse", 12.5)
        registry.log_and_register(pipeline, model_type="ridge", subset="FD001")
    name = registry.model_name("ridge", "FD001")
    registry.promote(name, 2)

    code = list_models.main([])

    assert code == 0
    out = capsys.readouterr().out
    assert name in out
    assert "1" in out
    assert "2" in out
    # The production version (2) and its val_rmse metric appear in the row.
    assert "12.5" in out
    # A header row is present (one of the column labels).
    assert "production" in out.lower()


def test_list_models_shows_placeholder_without_production_alias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A model with no production alias renders a '-' placeholder for it."""
    pipeline = _fitted_ridge_pipeline()
    with mlflow.start_run():
        registry.log_and_register(pipeline, model_type="ridge", subset="FD002")
    name = registry.model_name("ridge", "FD002")

    code = list_models.main([])

    assert code == 0
    out = capsys.readouterr().out
    assert name in out
    assert "-" in out

