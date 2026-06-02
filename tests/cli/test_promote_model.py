"""Tests for the turbofan-promote command."""
from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan import registry
from turbofan.cli import promote_model
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


def _register_two_versions(name_subset: str = "FD001") -> str:
    """Register v1 and v2 of a tiny Ridge model and return its name.

    Args:
        name_subset: C-MAPSS subset identifier for the registered-model name.

    Returns:
        The registered-model name.
    """
    pipeline = _fitted_ridge_pipeline()
    with mlflow.start_run():
        registry.log_and_register(pipeline, model_type="ridge", subset=name_subset)
    with mlflow.start_run():
        registry.log_and_register(pipeline, model_type="ridge", subset=name_subset)
    return registry.model_name("ridge", name_subset)


def test_promote_repoints_alias_to_requested_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Promoting version 2 points @production at version 2 and confirms on stdout."""
    name = _register_two_versions()

    code = promote_model.main([name, "2", "--to", "production"])

    assert code == 0
    out = capsys.readouterr().out
    assert "2" in out
    assert name in out
    aliased = MlflowClient().get_model_version_by_alias(name, "production")
    assert int(aliased.version) == 2


def test_promote_rollback_to_earlier_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A later promote of an older version rolls the alias back to that version."""
    name = _register_two_versions()

    assert promote_model.main([name, "2"]) == 0
    capsys.readouterr()
    assert promote_model.main([name, "1"]) == 0

    aliased = MlflowClient().get_model_version_by_alias(name, "production")
    assert int(aliased.version) == 1


def test_promote_unknown_model_returns_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Promoting a nonexistent model returns a nonzero code and logs an error.

    ``setup_logging`` reconfigures the root logger with ``force=True``, which
    detaches pytest's ``caplog`` handler, so the diagnostic is asserted on the
    stderr stream where leveled logs are emitted instead.
    """
    code = promote_model.main(["turbofan-ridge-does-not-exist", "1"])

    assert code == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert captured.out == ""

