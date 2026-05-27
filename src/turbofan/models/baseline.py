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

BaselineFeatureSet = Literal["raw", "raw_plus_rolling", "rolling"]
ROLLING_MARKERS = ("_rmean_", "_rstd_", "_rmin_", "_rmax_")
DEFAULT_BASELINE_WINDOWS = (10,)


def _is_rolling_feature(column: str) -> bool:
    """Return whether a column is a rolling sensor feature.

    Args:
        column: Feature column name.

    Returns:
        Whether the column is a rolling sensor-derived feature.
    """
    return column.startswith("s_") and any(
        marker in column for marker in ROLLING_MARKERS
    )


def _is_raw_sensor_feature(column: str) -> bool:
    """Return whether a column is a raw sensor feature.

    Args:
        column: Feature column name.

    Returns:
        Whether the column is a raw sensor column.
    """
    return column.startswith("s_") and not _is_rolling_feature(column)


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


class _ModelFeatureSelector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Select estimator-facing columns for a baseline feature family.

    Args:
        feature_set: Feature family to expose to the final estimator.
    """

    def __init__(
        self,
        feature_set: BaselineFeatureSet = "raw_plus_rolling",
    ) -> None:
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Validate and store selected model feature columns.

        Args:
            X: Engineered feature matrix.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted selector.

        Raises:
            ValueError: If the feature set is unsupported or empty.
        """
        if self.feature_set not in {"raw", "raw_plus_rolling", "rolling"}:
            raise ValueError(f"Unsupported feature_set: {self.feature_set}")
        self.columns_: list[str] = [
            column for column in X.columns if self._should_keep(column)
        ]
        if not self.columns_:
            raise ValueError(
                f"Feature set {self.feature_set!r} produced no model features."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return estimator-facing feature columns.

        Args:
            X: Engineered feature matrix.

        Returns:
            DataFrame containing only selected feature columns.
        """
        return X.loc[:, self.columns_]

    def _should_keep(self, column: str) -> bool:
        """Return whether a column belongs in the configured feature family.

        Args:
            column: Engineered feature column name.

        Returns:
            Whether to keep the feature.
        """
        if self.feature_set == "raw":
            return _is_raw_sensor_feature(column)
        if self.feature_set == "rolling":
            return _is_rolling_feature(column)
        return _is_raw_sensor_feature(column) or _is_rolling_feature(column)


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
    feature_set: BaselineFeatureSet = "rolling",
    n_modes: int = 1,
    random_state: int = 42,
) -> Pipeline:
    """Build an unfitted feature-plus-regressor sklearn Pipeline.

    Args:
        model_name: Baseline model identifier. Only ``"ridge"`` is supported.
        alpha: Ridge regularization strength.
        windows: Rolling window sizes for the feature pipeline.
            Default ``[10]`` for the PHM08-selected baseline.
        op_cols: Operational setting columns for normalization.
        sensor_std_threshold: Maximum training standard deviation at
            which sensor columns are dropped.
        sensor_keep: Sensor columns to force-keep even when low-variance.
        feature_set: Sensor-derived feature family to expose to the estimator.
        n_modes: Operating-mode count for ``OperatingModeNormalizer``.
            Derived from ``fd_subset`` by the caller via
            ``mode_count_for_subset``.
        random_state: KMeans random seed for the normalizer.

    Returns:
        Unfitted sklearn Pipeline with feature engineering, identifier
        dropping, imputation, scaling, and model steps.

    Raises:
        ValueError: If ``model_name`` or ``feature_set`` is unsupported.
    """
    if model_name != "ridge":
        raise ValueError(f"Unsupported model: {model_name}")
    if feature_set not in {"raw", "raw_plus_rolling", "rolling"}:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    effective_windows = (
        list(DEFAULT_BASELINE_WINDOWS) if windows is None else windows
    )
    return Pipeline(
        [
            (
                "features",
                build_feature_pipeline(
                    windows=effective_windows,
                    op_cols=op_cols,
                    sensor_std_threshold=sensor_std_threshold,
                    sensor_keep=sensor_keep,
                    n_modes=n_modes,
                    random_state=random_state,
                ),
            ),
            (
                "drop_identifiers",
                FunctionTransformer(_drop_identifier_columns, validate=False),
            ),
            (
                "select_model_features",
                _ModelFeatureSelector(feature_set=feature_set),
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
