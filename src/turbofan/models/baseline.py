"""Baseline sklearn pipeline factory."""
from __future__ import annotations

from typing import Literal, Self, cast

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from turbofan.features.pipeline import build_feature_pipeline


class _LowVarianceFeatureDropper(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Drop near-constant model features after feature engineering."""

    def __init__(self, std_threshold: float = 1e-6) -> None:
        self.std_threshold = std_threshold

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Identify features with meaningful training variation.

        Args:
            X: Engineered training feature matrix.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted transformer.
        """
        stds = X.std(numeric_only=True).fillna(0.0)
        self.columns_to_drop_: list[str] = [
            cast(str, column)
            for column, std in stds.items()
            if std <= self.std_threshold
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop near-constant columns found during fitting.

        Args:
            X: Engineered feature matrix.

        Returns:
            Feature matrix without near-constant columns.
        """
        return X.drop(columns=self.columns_to_drop_, errors="ignore")


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
    alpha: float = 100.0,
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
        dropping, imputation, scaling, and model steps.

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
            ("low_variance_filter", _LowVarianceFeatureDropper()),
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ).set_output(transform="pandas"),
            ),
            (
                "scaler",
                StandardScaler().set_output(transform="pandas"),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )
