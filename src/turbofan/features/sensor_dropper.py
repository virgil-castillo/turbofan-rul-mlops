"""Low-variance sensor removal transformer."""
from __future__ import annotations

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SensorDropper(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Drop sensor columns below a standard-deviation threshold.

    Discovers sensor columns by the ``s_*`` naming convention. Columns
    with standard deviation at or below ``std_threshold`` across the
    training set carry little information and are removed. Non-sensor
    columns (engine_id, cycle, op_*) are always preserved.

    Args:
        std_threshold: Maximum training standard deviation at which a sensor
            column should be dropped.
        keep: Optional list of sensor column names to force-keep
            even if they appear constant in training data.
    """

    def __init__(
        self,
        std_threshold: float = 0.0,
        keep: list[str] | None = None,
    ) -> None:
        self.std_threshold = std_threshold
        self.keep = keep

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> Self:
        """Identify low-variance sensor columns to drop.

        Args:
            X: Training DataFrame with sensor columns.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted transformer.
        """
        keep_set = set(self.keep or [])
        sensor_cols = [
            c for c in X.columns if c.startswith("s_")
        ]
        self.sensor_stds_: pd.Series[float] = X[sensor_cols].std()
        self.columns_to_drop_: list[str] = [
            col
            for col in sensor_cols
            if self.sensor_stds_[col] <= self.std_threshold
            and col not in keep_set
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove low-variance sensor columns.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with constant sensors removed.
        """
        return X.drop(columns=self.columns_to_drop_)
