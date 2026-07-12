"""Feature engineering pipeline factory."""
from __future__ import annotations

from typing import Any, Self

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from turbofan.features.engineering import FeatureEngineer, FeatureFamily
from turbofan.features.sensor_dropper import SensorDropper
from turbofan.preprocessing.normalization import OperatingModeNormalizer
from turbofan.sklearn_types import BaseEstimator, TransformerMixin


class SensorColumnSelector(BaseEstimator, TransformerMixin):
    """Select normalized sensor columns and keep engine_id for downstream grouping.

    ``fit`` records which columns start with ``s_``. ``transform`` returns
    those columns plus ``engine_id`` (when present) so that ``FeatureEngineer``
    can compute per-engine rolling and lag features.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:  # noqa: ARG002 - sklearn transformer API requires this signature
        """Record sensor column names from training data.

        Args:
            X: DataFrame after normalization.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted selector.
        """
        self.feature_cols_: list[str] = [
            c for c in X.columns if c.startswith("s_")
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return sensor columns plus engine_id for downstream grouping.

        Args:
            X: Normalized DataFrame.

        Returns:
            DataFrame with ``s_*`` columns and ``engine_id`` (if present).
        """
        keep = self.feature_cols_ + (
            ["engine_id"] if "engine_id" in X.columns else []
        )
        return X[keep]


class _AutoSensorNormalizer(OperatingModeNormalizer):
    """OperatingModeNormalizer variant that infers sensor feature_cols from data.

    Overrides ``fit`` to auto-detect ``s_*`` columns from the training
    DataFrame, so the pipeline does not need to know the exact sensor list
    at construction time.  Op-condition columns are used for KMeans but are
    never z-scored.

    Args:
        op_cols: Operating-condition columns for KMeans clustering.
        n_modes: Number of operating-mode KMeans clusters.
        random_state: KMeans random seed.
    """

    def __init__(
        self,
        op_cols: list[str] | None = None,
        n_modes: int = 1,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            feature_cols=None,
            op_cols=op_cols,
            n_modes=n_modes,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:  # noqa: ARG002 - sklearn transformer API requires this signature
        """Fit using only s_* columns present in X as feature_cols.

        Args:
            X: DataFrame after SensorDropper, containing s_* and op columns.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted normalizer.
        """
        sensor_cols: list[str] = [c for c in X.columns if c.startswith("s_")]
        self.feature_cols = sensor_cols
        return super().fit(X, None)

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002 - sklearn transformer API requires this signature
        """Return estimator parameters.

        Args:
            deep: Ignored; present for sklearn compatibility.

        Returns:
            Dictionary of constructor parameter names to values.
        """
        return {
            "op_cols": self.op_cols,
            "n_modes": self.n_modes,
            "random_state": self.random_state,
        }


def build_feature_pipeline(
    op_cols: list[str] | None = None,
    sensor_drop: list[str] | None = None,
    n_modes: int = 1,
    random_state: int = 42,
    feature_families: list[FeatureFamily] | None = None,
    windows: list[int] | None = None,
    lag_steps: list[int] | None = None,
) -> Pipeline:
    """Build the shared 5-step feature engineering pipeline.

    Steps: ``sensor_dropper`` → ``normalizer`` → ``sensor_selector``
    → ``feature_engineer`` → ``scaler``.

    The normalizer auto-detects sensor ``feature_cols`` from the data at fit
    time so op cols are available for KMeans but are not z-scored.
    ``SensorColumnSelector`` retains ``engine_id`` so that
    ``FeatureEngineer`` can compute per-engine rolling and lag features.
    The final output contains only the scaled engineered feature columns.

    Args:
        op_cols: Operating-condition columns for KMeans clustering.
            Defaults to ``["op_1", "op_2", "op_3"]``.
        sensor_drop: Sensor column names to exclude before normalization.
            Determined from EDA; passed to ``SensorDropper``.
        n_modes: Number of operating-mode KMeans clusters.
        random_state: KMeans random seed.
        feature_families: Ordered feature families to produce.
        windows: Rolling window sizes. Forwarded to ``FeatureEngineer``.
        lag_steps: Lag offsets. Forwarded to ``FeatureEngineer``.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            ("sensor_dropper", SensorDropper(drop=sensor_drop)),
            (
                "normalizer",
                _AutoSensorNormalizer(
                    op_cols=op_cols,
                    n_modes=n_modes,
                    random_state=random_state,
                ),
            ),
            ("sensor_selector", SensorColumnSelector()),
            (
                "feature_engineer",
                FeatureEngineer(
                    feature_families=feature_families,
                    windows=windows,
                    lag_steps=lag_steps,
                ),
            ),
            ("scaler", StandardScaler().set_output(transform="pandas")),
        ]
    )
