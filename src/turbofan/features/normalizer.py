"""Operational-condition-aware z-score normalization."""
from __future__ import annotations

from typing import Any, Self, cast

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OperationalNormalizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Z-score normalize sensor readings by operational condition.

    Groups data by unique combinations of operational setting
    columns, then applies per-group z-score normalization. Falls
    back to global statistics for operating conditions seen at
    test time but not in training.

    Args:
        op_cols: Operational setting column names.
            Default ``["op_1", "op_2", "op_3"]``.
        std_floor: Minimum meaningful standard deviation. Per-condition
            and global standard deviations at or below this value are treated
            as ``1.0`` to avoid numerical blow-ups on nearly constant columns.
    """

    def __init__(
        self,
        op_cols: list[str] | None = None,
        std_floor: float = 1e-3,
    ) -> None:
        self.op_cols = (
            op_cols
            if op_cols is not None
            else ["op_1", "op_2", "op_3"]
        )
        self.std_floor = std_floor

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> Self:
        """Compute per-condition and global normalization stats.

        Args:
            X: Training DataFrame.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted transformer.
        """
        exclude = {"engine_id", "cycle", *self.op_cols}
        self.numeric_cols_: list[str] = [
            c
            for c in X.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(X[c])
        ]
        grouped = X.groupby(self.op_cols)[self.numeric_cols_]
        self.group_means_: pd.DataFrame = grouped.mean()
        group_stds = grouped.std().fillna(1.0)
        self.group_stds_: pd.DataFrame = group_stds.mask(
            group_stds.abs() <= self.std_floor,
            1.0,
        )
        self.global_mean_: pd.Series[float] = (
            X[self.numeric_cols_].mean()
        )
        global_std = X[self.numeric_cols_].std().fillna(1.0)
        self.global_std_: pd.Series[float] = global_std.mask(
            global_std.abs() <= self.std_floor,
            1.0,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply per-condition z-score normalization.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with normalized numeric columns.
        """
        result = X.copy()
        existing_numeric_cols = [
            col for col in self.numeric_cols_ if col in result.columns
        ]
        result[existing_numeric_cols] = result[existing_numeric_cols].astype(
            "float64"
        )
        for condition, group in result.groupby(self.op_cols):
            if not isinstance(condition, tuple):
                condition = (condition,)
            idx = group.index
            if condition in self.group_means_.index:
                means = cast(
                    "pd.Series[float]",
                    self.group_means_.loc[cast(Any, condition)],
                )
                stds = cast(
                    "pd.Series[float]",
                    self.group_stds_.loc[cast(Any, condition)],
                )
            else:
                means = self.global_mean_
                stds = self.global_std_
            for col in self.numeric_cols_:
                if col in result.columns:
                    result.loc[cast(Any, idx), col] = (
                        (result.loc[idx, col] - means[col])
                        / stds[col]
                    )
        return result
