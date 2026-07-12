"""Degradation analysis for C-MAPSS turbofan data."""
from __future__ import annotations

import pandas as pd

from turbofan.data import labels


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return column names matching the s_* sensor naming convention."""
    return [c for c in df.columns if c.startswith("s_")]


def compute_rul_curves(
    df: pd.DataFrame, max_rul: int = 125
) -> pd.DataFrame:
    """Add piecewise-linear RUL labels to the DataFrame.

    Delegates to ``turbofan.data.labels.compute_rul_labels``. Returns
    a copy of ``df`` with an additional ``rul`` column.

    Args:
        df: Training DataFrame with engine_id and cycle columns.
        max_rul: Maximum RUL cap.

    Returns:
        DataFrame with all original columns plus ``rul``.
    """
    result = df.copy()
    result["rul"] = labels.compute_rul_labels(df, max_rul=max_rul)
    return result


def compute_sensor_trends(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 10,
) -> pd.DataFrame:
    """Compute rolling-mean smoothed sensor values per engine.

    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
        sensor_cols: Sensor columns to smooth.
        window: Rolling window size in cycles.

    Returns:
        DataFrame with engine_id, cycle, and smoothed sensor columns.
    """
    result = df[["engine_id", "cycle"]].copy()
    grouped = df.groupby("engine_id")[sensor_cols]
    smoothed = grouped.transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    for col in sensor_cols:
        result[col] = smoothed[col]
    return result


def select_informative_sensors(
    df: pd.DataFrame,
    rul: pd.Series[int],
    threshold: float = 0.1,
) -> list[str]:
    """Select sensors whose absolute Pearson correlation with RUL
    exceeds a threshold.

    Args:
        df: DataFrame with sensor columns.
        rul: Series of RUL values aligned to df.index.
        threshold: Minimum absolute correlation to keep a sensor.

    Returns:
        List of sensor column names that pass the threshold.
    """
    sensors = _sensor_columns(df)
    result = []
    for col in sensors:
        corr = df[col].corr(rul)
        if abs(corr) >= threshold:
            result.append(col)
    return result
