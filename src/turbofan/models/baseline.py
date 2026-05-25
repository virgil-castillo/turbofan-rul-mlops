"""Baseline sklearn pipeline factory."""
from __future__ import annotations

from typing import Literal

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from turbofan.features.pipeline import build_feature_pipeline


def _drop_identifier_columns(X: object) -> object:
    """Remove identifier columns that should not be model features.

    Args:
        X: Feature matrix after feature engineering.

    Returns:
        Feature matrix without arbitrary row identifier columns.
    """
    if hasattr(X, "drop"):
        return X.drop(columns=["engine_id"], errors="ignore")
    return X


def build_baseline_pipeline(
    model_name: Literal["ridge"] = "ridge",
    alpha: float = 1.0,
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
    sensor_std_threshold: float = 0.0,
    sensor_keep: list[str] | None = None,
) -> Pipeline:
    """Build an unfitted feature-plus-regressor sklearn Pipeline.

    Args:
        model_name: Baseline model identifier. Only ``"ridge"`` is supported.
        alpha: Ridge regularization strength.
        windows: Rolling window sizes for the feature pipeline.
        op_cols: Operational setting columns for normalization.
        sensor_std_threshold: Maximum training standard deviation at
            which sensor columns are dropped.
        sensor_keep: Sensor columns to force-keep even when low-variance.

    Returns:
        Unfitted sklearn Pipeline with feature engineering, identifier
        dropping, and model steps.

    Raises:
        ValueError: If ``model_name`` is unsupported.
    """
    if model_name != "ridge":
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(
        [
            (
                "features",
                build_feature_pipeline(
                    windows=windows,
                    op_cols=op_cols,
                    sensor_std_threshold=sensor_std_threshold,
                    sensor_keep=sensor_keep,
                ),
            ),
            (
                "drop_identifiers",
                FunctionTransformer(_drop_identifier_columns, validate=False),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )
