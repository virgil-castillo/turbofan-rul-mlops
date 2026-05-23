"""Sensor characterization for C-MAPSS turbofan data."""
from __future__ import annotations

import pandas as pd
from scipy import stats as scipy_stats


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return column names matching the s_* sensor naming convention."""
    return [c for c in df.columns if c.startswith("s_")]


def compute_sensor_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for all sensor columns.

    Args:
        df: DataFrame containing sensor columns (s_1 through s_21).

    Returns:
        DataFrame indexed by sensor name with columns: mean, std, min,
        max, skewness, kurtosis.
    """
    sensors = _sensor_columns(df)
    records = []
    for col in sensors:
        series = df[col].dropna()
        records.append(
            {
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "skewness": float(scipy_stats.skew(series)),
                "kurtosis": float(scipy_stats.kurtosis(series)),
            }
        )
    return pd.DataFrame(records, index=pd.Index(sensors))


def compute_correlation_matrix(
    df: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """Compute Pearson correlation matrix for the given columns.

    Args:
        df: Input DataFrame.
        cols: Column names to include in the correlation matrix.

    Returns:
        Square DataFrame of pairwise Pearson correlations.
    """
    return df[cols].corr()


def estimate_noise(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 5,
) -> pd.DataFrame:
    """Estimate per-sensor noise level via rolling standard deviation.

    Computes rolling std within each engine's time series, then
    averages across all engines to produce one noise estimate per sensor.

    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
        sensor_cols: Sensor columns to analyze.
        window: Rolling window size in cycles.

    Returns:
        DataFrame with columns: sensor, mean_rolling_std.
    """
    grouped = df.groupby("engine_id")[sensor_cols]
    rolling_std = grouped.transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )
    mean_noise = rolling_std.mean()
    return pd.DataFrame(
        {"sensor": sensor_cols, "mean_rolling_std": mean_noise.values}
    )
