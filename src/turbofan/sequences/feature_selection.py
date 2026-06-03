"""Correlation-based feature selection for turbofan sensor columns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_correlated_sensors(
    df: pd.DataFrame,
    target_col: str = "rul",
    threshold: float = 0.5,
) -> list[str]:
    """Select sensor columns with high absolute correlation to the target.

    Computes the absolute Pearson correlation between each ``s_*`` column
    and ``target_col``. Returns sensor column names where
    ``|r| >= threshold``, sorted by descending ``|r|``.

    Args:
        df: DataFrame containing sensor and target columns.
        target_col: Name of the target column.
        threshold: Minimum absolute correlation to include a sensor.

    Returns:
        Sensor column names sorted by descending absolute correlation.

    Raises:
        ValueError: If no sensors meet the correlation threshold.
    """
    sensor_cols = [col for col in df.columns if col.startswith("s_")]
    # A constant sensor column has zero variance, so its Pearson correlation is
    # 0/0 -> NaN. That NaN is intentional: it fails the ``>= threshold``
    # comparison below and is excluded. Silence the expected divide-by-zero so
    # the well-defined NaN does not leak an uncontrolled RuntimeWarning.
    with np.errstate(invalid="ignore", divide="ignore"):
        abs_corr = df[sensor_cols].corrwith(df[target_col]).abs()
    passing = abs_corr[abs_corr >= threshold].sort_values(ascending=False)
    if passing.empty:
        raise ValueError(
            f"No sensors meet the correlation threshold {threshold}."
        )
    return passing.index.tolist()
