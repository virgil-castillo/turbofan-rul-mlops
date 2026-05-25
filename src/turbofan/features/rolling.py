"""Multi-window rolling feature extraction transformer."""
from __future__ import annotations

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RollingFeatureExtractor(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Compute rolling statistics per engine for each sensor column.

    For each sensor and each window size, computes four rolling
    aggregations grouped by engine_id: mean, std, min, max.
    Uses ``min_periods=1`` so early cycles get values instead of
    NaN.

    Args:
        windows: List of rolling window sizes in cycles.
            Default ``[5, 10, 20]``.
    """

    def __init__(
        self, windows: list[int] | None = None
    ) -> None:
        self.windows = windows if windows is not None else [5, 10, 20]

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> Self:
        """Discover sensor columns from the training data.

        Args:
            X: Training DataFrame with sensor columns.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted transformer.
        """
        self.sensor_cols_: list[str] = [
            c for c in X.columns if c.startswith("s_")
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add rolling feature columns to the DataFrame.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with original columns plus rolling features.
        """
        feature_columns: dict[str, pd.Series] = {}
        for window in self.windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                feature_columns[f"{col}_rmean_{window}"] = grp.transform(
                    lambda s: s.rolling(
                        window, min_periods=1
                    ).mean()
                )
                feature_columns[f"{col}_rstd_{window}"] = (
                    grp.transform(
                        lambda s: s.rolling(
                            window, min_periods=1
                        ).std()
                    ).fillna(0.0)
                )
                feature_columns[f"{col}_rmin_{window}"] = grp.transform(
                    lambda s: s.rolling(
                        window, min_periods=1
                    ).min()
                )
                feature_columns[f"{col}_rmax_{window}"] = grp.transform(
                    lambda s: s.rolling(
                        window, min_periods=1
                    ).max()
                )
        if not feature_columns:
            return X.copy()

        features = pd.DataFrame(feature_columns, index=X.index)
        return pd.concat([X.copy(), features], axis=1)
