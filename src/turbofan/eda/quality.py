"""Data quality assessment for C-MAPSS turbofan data."""
from __future__ import annotations

import pandas as pd


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return column names matching the s_* sensor naming convention."""
    return [c for c in df.columns if c.startswith("s_")]


def find_missing_values(df: pd.DataFrame) -> pd.Series[int]:
    """Count NaN values per column.

    Args:
        df: Input DataFrame.

    Returns:
        Series indexed by column name with NaN counts.
    """
    return df.isna().sum()


def find_constant_sensors(df: pd.DataFrame) -> list[str]:
    """Identify sensor columns with zero variance globally (across all
    engines and cycles).

    A sensor that is constant across the entire dataset carries no
    information and should be dropped before modeling.

    Args:
        df: Input DataFrame with sensor columns (s_1 through s_21).

    Returns:
        List of column names whose standard deviation is zero.
    """
    sensors = _sensor_columns(df)
    return [col for col in sensors if df[col].std() == 0.0]


def find_low_variance_sensors(df: pd.DataFrame, tol: float = 1e-3) -> list[str]:
    """Identify sensor columns whose sample standard deviation is at or below
    a tolerance threshold.

    Complements :func:`find_constant_sensors` by catching near-constant sensors
    that carry negligible information for modeling.

    Args:
        df: Input DataFrame with sensor columns (s_1 through s_21).
        tol: Maximum sample std (ddof=1) to consider low-variance. Defaults to 1e-3.

    Returns:
        List of sensor column names whose sample std is <= tol.
    """
    sensor_cols = _sensor_columns(df)
    return [col for col in sensor_cols if df[col].std() <= tol]


def summarize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize data types and unique value counts per column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: column_name, dtype, n_unique.
    """
    records = [
        {
            "column_name": col,
            "dtype": str(df[col].dtype),
            "n_unique": df[col].nunique(),
        }
        for col in df.columns
    ]
    return pd.DataFrame(records)
