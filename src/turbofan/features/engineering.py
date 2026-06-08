"""Config-driven feature engineering transformer."""
from __future__ import annotations

from typing import Any, Literal, Self, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _slope(x: npt.NDArray[np.floating[Any]]) -> float:
    """Compute the least-squares slope of ``x`` against an integer index.

    The index runs from ``0`` to ``len(x) - 1``.  When the window contains
    only a single point (denominator is zero) the slope is defined as ``0.0``
    to avoid division by zero.

    Args:
        x: 1-D array of sensor values in the rolling window (raw=True).

    Returns:
        Least-squares slope, or ``0.0`` when the window has length 1.
    """
    t = np.arange(len(x), dtype=float)
    denom = float(((t - t.mean()) ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(((t - t.mean()) * (x - x.mean())).sum() / denom)


FeatureFamily = Literal[
    "raw",
    "rolling_mean",
    "lag",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_slope",
    "rolling_delta",
]

_VALID_FEATURE_FAMILIES: frozenset[str] = frozenset(get_args(FeatureFamily))


class FeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Apply a config-driven feature transformation to normalized sensor columns.

    Expects a DataFrame whose columns are ``s_*`` sensor columns (already
    normalized) and optionally ``engine_id`` for per-engine grouping.
    Returns only the engineered feature columns; ``engine_id`` is dropped.

    Args:
        feature_families: Ordered feature families to compute. Defaults to
            ``["raw"]``.
        windows: Rolling window sizes in cycles. Required for rolling feature sets.
            Default ``[10]``.
        lag_steps: Lag offsets in cycles. Required for ``lag`` feature set.
            Default ``[1]``.
    """

    def __init__(
        self,
        feature_families: list[FeatureFamily] | None = None,
        windows: list[int] | None = None,
        lag_steps: list[int] | None = None,
    ) -> None:
        self.feature_families = feature_families
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
            ValueError: If ``feature_families`` contains unsupported values.
        """
        self.sensor_cols_: list[str] = [
            c for c in X.columns if c.startswith("s_")
        ]
        self._windows: list[int] = self.windows if self.windows is not None else [10]
        self._lag_steps: list[int] = (
            self.lag_steps if self.lag_steps is not None else [1]
        )
        self.feature_families_: list[FeatureFamily] = (
            self._resolve_feature_families()
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
        frames = [
            self._transform_family(X, family)
            for family in self.feature_families_
        ]
        return pd.concat(frames, axis=1)

    def _resolve_feature_families(self) -> list[FeatureFamily]:
        """Resolve constructor settings to an ordered feature family list.

        Returns:
            Ordered feature family names.

        Raises:
            ValueError: If ``feature_families`` is empty or unsupported.
        """
        if self.feature_families is None:
            return ["raw"]
        if not self.feature_families:
            raise ValueError("feature_families must contain at least one family")
        invalid = sorted(set(self.feature_families) - _VALID_FEATURE_FAMILIES)
        if invalid:
            raise ValueError(
                f"Unsupported feature_families: {invalid}. "
                f"Valid values: {sorted(_VALID_FEATURE_FAMILIES)}"
            )
        return list(self.feature_families)

    def _transform_family(
        self,
        X: pd.DataFrame,
        family: FeatureFamily,
    ) -> pd.DataFrame:
        """Apply one feature family.

        Args:
            X: Sensor DataFrame.
            family: Feature family to compute.

        Returns:
            DataFrame for the requested feature family.
        """
        if family == "raw":
            return X[self.sensor_cols_].copy()
        if family == "rolling_mean":
            return self._rolling_mean(X)
        if family == "lag":
            return self._lag_features(X)
        if family == "rolling_std":
            return self._rolling_std(X)
        if family == "rolling_min":
            return self._rolling_min(X)
        if family == "rolling_max":
            return self._rolling_max(X)
        if family == "rolling_slope":
            return self._rolling_slope(X)
        return self._rolling_delta(X)

    def _compute_output_cols(self) -> list[str]:
        """Return the list of output column names for the configured feature set.

        Returns:
            Ordered list of output column names.
        """
        cols: list[str] = []
        for family in self.feature_families_:
            cols.extend(self._output_cols_for_family(family))
        return cols

    def _output_cols_for_family(self, family: FeatureFamily) -> list[str]:
        """Return output column names for one feature family.

        Args:
            family: Feature family.

        Returns:
            Ordered output column names for ``family``.
        """
        if family == "raw":
            return list(self.sensor_cols_)
        if family == "rolling_mean":
            return [
                f"{col}_rmean_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        if family == "lag":
            return [
                f"{col}_lag_{step}"
                for step in self._lag_steps
                for col in self.sensor_cols_
            ]
        if family == "rolling_std":
            return [
                f"{col}_rstd_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        if family == "rolling_min":
            return [
                f"{col}_rmin_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        if family == "rolling_max":
            return [
                f"{col}_rmax_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        if family == "rolling_slope":
            return [
                f"{col}_rslope_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        if family == "rolling_delta":
            return [
                f"{col}_rdelta_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        raise ValueError(f"Unsupported feature family: {family!r}")

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

    def _lag_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-engine lagged sensor-value features.

        Each feature is ``x[t-N]``, where ``N`` is the lag step. Early cycles
        where ``x[t-N]`` has no history are backfilled within the same engine
        so the sklearn pipeline receives finite values without crossing engine
        boundaries.

        Args:
            X: Input DataFrame with ``engine_id`` and sensor columns.

        Returns:
            DataFrame of lagged sensor-value feature columns.
        """
        cols: dict[str, pd.Series] = {}
        for step in self._lag_steps:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_lag_{step}"] = grp.transform(
                    lambda s, _step=step: s.shift(_step).bfill().fillna(s)
                )
        return pd.DataFrame(cols, index=X.index)

    def _rolling_std(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-engine rolling standard deviation for each sensor and window.

        The first cycle of each engine yields a single-point window whose std
        is ``NaN``; that value is filled with ``0.0``.

        Args:
            X: Input DataFrame with ``engine_id`` and sensor columns.

        Returns:
            DataFrame of rolling-std feature columns (``{s}_rstd_{w}``).
        """
        cols: dict[str, pd.Series] = {}
        for w in self._windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_rstd_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).std()
                ).fillna(0.0)
        return pd.DataFrame(cols, index=X.index)

    def _rolling_min(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-engine rolling minimum for each sensor and window.

        Args:
            X: Input DataFrame with ``engine_id`` and sensor columns.

        Returns:
            DataFrame of rolling-min feature columns (``{s}_rmin_{w}``).
        """
        cols: dict[str, pd.Series] = {}
        for w in self._windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_rmin_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).min()
                )
        return pd.DataFrame(cols, index=X.index)

    def _rolling_max(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-engine rolling maximum for each sensor and window.

        Args:
            X: Input DataFrame with ``engine_id`` and sensor columns.

        Returns:
            DataFrame of rolling-max feature columns (``{s}_rmax_{w}``).
        """
        cols: dict[str, pd.Series] = {}
        for w in self._windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_rmax_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).max()
                )
        return pd.DataFrame(cols, index=X.index)

    def _rolling_slope(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-engine rolling least-squares slope for each sensor and window.

        The slope is estimated by ordinary least squares of the sensor values
        against the integer index ``0 .. L-1`` within each rolling window of
        length ``L``.  Windows containing only a single point have a zero
        denominator; the slope is defined as ``0.0`` in that case.

        Args:
            X: Input DataFrame with ``engine_id`` and sensor columns.

        Returns:
            DataFrame of rolling-slope feature columns (``{s}_rslope_{w}``).
        """
        cols: dict[str, pd.Series] = {}
        for w in self._windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_rslope_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).apply(
                        _slope, raw=True
                    )
                )
        return pd.DataFrame(cols, index=X.index)

    def _rolling_delta(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute per-engine rolling windowed difference for each sensor and window.

        Each feature is ``x[t] - x[t - (w - 1)]``, i.e. the raw change over
        the span of the rolling window.  Early cycles where ``x[t - (w-1)]``
        has no history are backfilled to the earliest available value, yielding
        a difference of ``0.0`` for the first cycle of each engine.

        Note: ``rolling_delta`` is distinct from the ``lag`` family. ``lag``
        emits shifted prior values keyed to an arbitrary lag step ``N``;
        ``rolling_delta`` emits windowed differences keyed directly to the
        rolling span ``w - 1``.

        Args:
            X: Input DataFrame with ``engine_id`` and sensor columns.

        Returns:
            DataFrame of rolling-delta feature columns (``{s}_rdelta_{w}``).
        """
        cols: dict[str, pd.Series] = {}
        for w in self._windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_rdelta_{w}"] = grp.transform(
                    lambda s, _w=w: s - s.shift(_w - 1).fillna(s)
                )
        return pd.DataFrame(cols, index=X.index)
