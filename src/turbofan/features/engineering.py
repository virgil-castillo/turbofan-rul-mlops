"""Config-driven feature engineering transformer."""
from __future__ import annotations

from typing import Literal, Self, get_args

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

FeatureSet = Literal[
    "raw",
    "rolling_mean",
    "rolling_stats",
    "raw_plus_rolling_mean",
    "raw_plus_rolling_stats",
    "lag",
]

_VALID_FEATURE_SETS: frozenset[str] = frozenset(get_args(FeatureSet))


class FeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Apply a config-driven feature transformation to normalized sensor columns.

    Expects a DataFrame whose columns are ``s_*`` sensor columns (already
    normalized) and optionally ``engine_id`` for per-engine grouping.
    Returns only the engineered feature columns; ``engine_id`` is dropped.

    Args:
        feature_set: Which feature family to compute.
        windows: Rolling window sizes in cycles. Required for rolling feature sets.
            Default ``[10]``.
        lag_steps: Lag offsets in cycles. Required for ``lag`` feature set.
            Default ``[1]``.
    """

    def __init__(
        self,
        feature_set: FeatureSet = "raw",
        windows: list[int] | None = None,
        lag_steps: list[int] | None = None,
    ) -> None:
        self.feature_set = feature_set
        self.windows = windows
        self.lag_steps = lag_steps

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Record input sensor columns and compute output column names.

        Args:
            X: Normalized sensor DataFrame (s_* columns, optionally engine_id).
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted transformer.

        Raises:
            ValueError: If ``feature_set`` is unsupported.
        """
        if self.feature_set not in _VALID_FEATURE_SETS:
            raise ValueError(
                f"Unsupported feature_set: {self.feature_set!r}. "
                f"Valid values: {sorted(_VALID_FEATURE_SETS)}"
            )
        self.sensor_cols_: list[str] = [
            c for c in X.columns if c.startswith("s_")
        ]
        self._windows: list[int] = self.windows if self.windows is not None else [10]
        self._lag_steps: list[int] = (
            self.lag_steps if self.lag_steps is not None else [1]
        )
        self.feature_cols_: list[str] = self._compute_output_cols()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and return only the feature columns.

        Args:
            X: Sensor DataFrame. Must contain ``sensor_cols_`` from fit; may
                contain ``engine_id`` for per-engine grouping.

        Returns:
            DataFrame with exactly ``feature_cols_`` columns.
        """
        check_is_fitted(self)
        if self.feature_set == "raw":
            return X[self.sensor_cols_].copy()
        if self.feature_set == "rolling_mean":
            return self._rolling_mean(X)
        if self.feature_set == "rolling_stats":
            return self._rolling_stats(X)
        if self.feature_set == "raw_plus_rolling_mean":
            raw = X[self.sensor_cols_].copy()
            return pd.concat([raw, self._rolling_mean(X)], axis=1)
        if self.feature_set == "raw_plus_rolling_stats":
            raw = X[self.sensor_cols_].copy()
            return pd.concat([raw, self._rolling_stats(X)], axis=1)
        # lag
        return self._lag_features(X)

    def _compute_output_cols(self) -> list[str]:
        """Return the list of output column names for the configured feature set.

        Returns:
            Ordered list of output column names.
        """
        if self.feature_set == "raw":
            return list(self.sensor_cols_)
        if self.feature_set == "rolling_mean":
            return [
                f"{col}_rmean_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        if self.feature_set == "rolling_stats":
            return [
                f"{col}_{stat}_{w}"
                for w in self._windows
                for col in self.sensor_cols_
                for stat in ["rmean", "rstd", "rmin", "rmax"]
            ]
        if self.feature_set == "raw_plus_rolling_mean":
            return list(self.sensor_cols_) + [
                f"{col}_rmean_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        if self.feature_set == "raw_plus_rolling_stats":
            return list(self.sensor_cols_) + [
                f"{col}_{stat}_{w}"
                for w in self._windows
                for col in self.sensor_cols_
                for stat in ["rmean", "rstd", "rmin", "rmax"]
            ]
        # lag
        return [
            f"{col}_lag_{step}"
            for step in self._lag_steps
            for col in self.sensor_cols_
        ]

    def _rolling_mean(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-engine rolling mean for each sensor and window.

        Args:
            X: Input DataFrame with ``engine_id`` and sensor columns.

        Returns:
            DataFrame of rolling-mean feature columns.
        """
        cols: dict[str, pd.Series] = {}
        for w in self._windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_rmean_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).mean()
                )
        return pd.DataFrame(cols, index=X.index)

    def _rolling_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-engine rolling mean/std/min/max for each sensor and window.

        Args:
            X: Input DataFrame with ``engine_id`` and sensor columns.

        Returns:
            DataFrame of rolling-stat feature columns (mean, std, min, max).
        """
        cols: dict[str, pd.Series] = {}
        for w in self._windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_rmean_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).mean()
                )
                cols[f"{col}_rstd_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).std()
                ).fillna(0.0)
                cols[f"{col}_rmin_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).min()
                )
                cols[f"{col}_rmax_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).max()
                )
        return pd.DataFrame(cols, index=X.index)

    def _lag_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-engine lagged sensor values, backfilling early cycles.

        Args:
            X: Input DataFrame with ``engine_id`` and sensor columns.

        Returns:
            DataFrame of lag feature columns.
        """
        cols: dict[str, pd.Series] = {}
        for step in self._lag_steps:
            for col in self.sensor_cols_:
                def _lag_and_backfill(s: pd.Series, _step: int = step) -> pd.Series:
                    return s.shift(_step).bfill()

                cols[f"{col}_lag_{step}"] = X.groupby("engine_id")[col].transform(
                    _lag_and_backfill
                )
        return pd.DataFrame(cols, index=X.index)
