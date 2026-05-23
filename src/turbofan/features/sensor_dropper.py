"""Constant sensor removal transformer."""
from __future__ import annotations

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SensorDropper(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Drop sensor columns with zero variance in the training set.

    Discovers sensor columns by the ``s_*`` naming convention. Columns
    with zero standard deviation across the entire training set carry
    no information and are removed. Non-sensor columns (engine_id,
    cycle, op_*) are always preserved.

    Args:
        keep: Optional list of sensor column names to force-keep
            even if they appear constant in training data.
    """

    def __init__(
        self, keep: list[str] | None = None
    ) -> None:
        self.keep = keep

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> Self:
        """Identify constant sensor columns to drop.

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
        self.columns_to_drop_: list[str] = [
            col
            for col in sensor_cols
            if X[col].std() == 0.0 and col not in keep_set
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove constant sensor columns.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with constant sensors removed.
        """
        return X.drop(columns=self.columns_to_drop_)
