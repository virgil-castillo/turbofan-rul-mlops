"""Tests for turbofan.eda.quality."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.eda.quality import (
    find_constant_sensors,
    find_missing_values,
    summarize_dtypes,
)


def _make_sensor_df() -> pd.DataFrame:
    """Build a small DataFrame with 3 sensors, one constant."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "s_1": [100.0, 101.0, 102.0, 100.5, 101.5],
            "s_2": [200.0, 200.0, 200.0, 200.0, 200.0],
            "s_3": [50.0, 51.0, np.nan, 50.5, 51.5],
        }
    )


def test_find_missing_values_complete_data() -> None:
    """Returns zero for columns with no NaNs."""
    df = _make_sensor_df()
    result = find_missing_values(df)
    assert result["s_1"] == 0
    assert result["s_2"] == 0


def test_find_missing_values_with_nans() -> None:
    """Returns correct NaN count for columns with missing data."""
    df = _make_sensor_df()
    result = find_missing_values(df)
    assert result["s_3"] == 1


def test_find_constant_sensors_detects_constant() -> None:
    """Identifies s_2 as constant (zero variance)."""
    df = _make_sensor_df()
    result = find_constant_sensors(df)
    assert "s_2" in result


def test_find_constant_sensors_excludes_varying() -> None:
    """Does not flag s_1 which varies across rows."""
    df = _make_sensor_df()
    result = find_constant_sensors(df)
    assert "s_1" not in result


def test_find_constant_sensors_empty_when_all_vary() -> None:
    """Returns empty list when no sensor is constant."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 2],
            "cycle": [1, 1],
            "s_1": [1.0, 2.0],
            "s_2": [3.0, 4.0],
        }
    )
    result = find_constant_sensors(df)
    assert result == []


def test_summarize_dtypes_columns() -> None:
    """Returns DataFrame with column_name, dtype, n_unique columns."""
    df = _make_sensor_df()
    result = summarize_dtypes(df)
    assert list(result.columns) == ["column_name", "dtype", "n_unique"]


def test_summarize_dtypes_row_count() -> None:
    """Returns one row per column in the input DataFrame."""
    df = _make_sensor_df()
    result = summarize_dtypes(df)
    assert len(result) == len(df.columns)
