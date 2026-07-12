"""Config-driven sensor column removal transformer."""
from __future__ import annotations

from typing import Self

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from turbofan.sklearn_types import BaseEstimator, TransformerMixin


class SensorDropper(BaseEstimator, TransformerMixin):
    """Drop an explicit list of sensor columns determined during EDA.

    The drop list is injected via config — no statistics are computed
    from training data. Columns absent from the DataFrame are silently
    skipped so the transformer is safe to apply on subsets.

    Args:
        drop: Sensor column names to remove.
    """

    def __init__(self, drop: list[str] | None = None) -> None:
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:  # noqa: ARG002 - sklearn transformer API requires this signature
        """Record the drop list for sklearn pipeline compatibility.

        Args:
            X: Training DataFrame. Not used.
            y: Ignored.

        Returns:
            Fitted transformer.
        """
        self.columns_to_drop_: list[str] = list(self.drop or [])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove configured sensor columns.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with configured sensors removed.

        Raises:
            NotFittedError: If the transformer has not been fitted.
        """
        check_is_fitted(self)
        present = [c for c in self.columns_to_drop_ if c in X.columns]
        return X.drop(columns=present)
