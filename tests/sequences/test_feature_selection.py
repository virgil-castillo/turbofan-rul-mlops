"""Tests for turbofan.sequences.feature_selection."""
from __future__ import annotations

import pandas as pd
import pytest

from turbofan.sequences.feature_selection import select_correlated_sensors


def test_select_correlated_sensors_returns_above_threshold() -> None:
    """Only sensors with |r| >= threshold are returned."""
    df = pd.DataFrame(
        {
            "rul": [4, 3, 2, 1, 0],
            "s_1": [10, 8, 6, 4, 2],   # perfect correlation
            "s_2": [5, 5, 5, 5, 5],    # constant -> NaN correlation
            "s_3": [3, 1, 4, 1, 5],    # weak correlation ~0.35
        }
    )

    result = select_correlated_sensors(df, threshold=0.5)

    assert "s_1" in result
    assert "s_2" not in result
    assert "s_3" not in result


def test_select_correlated_sensors_sorts_by_descending_abs_correlation() -> None:
    """Result is ordered from highest to lowest absolute correlation."""
    df = pd.DataFrame(
        {
            "rul": [4, 3, 2, 1, 0],
            "s_1": [10, 8, 6, 4, 2],   # |r| = 1.0
            "s_2": [8, 7, 5, 3, 2],    # |r| ≈ 0.99
        }
    )

    result = select_correlated_sensors(df, threshold=0.5)

    assert result[0] == "s_1"
    assert result[1] == "s_2"
    assert len(result) == 2


def test_select_correlated_sensors_excludes_non_sensor_columns() -> None:
    """Non-s_* columns are never included regardless of their correlation."""
    df = pd.DataFrame(
        {
            "rul": [4, 3, 2, 1, 0],
            "engine_id": [1, 2, 3, 4, 5],
            "cycle": [5, 4, 3, 2, 1],
            "op_1": [10, 8, 6, 4, 2],
            "s_1": [10, 8, 6, 4, 2],
        }
    )

    result = select_correlated_sensors(df, threshold=0.5)

    assert "op_1" not in result
    assert "engine_id" not in result
    assert "cycle" not in result
    assert "rul" not in result
    assert "s_1" in result


def test_select_correlated_sensors_raises_when_none_pass() -> None:
    """ValueError is raised when no sensors meet the correlation threshold."""
    df = pd.DataFrame(
        {
            "rul": [2, 1, 0],
            "s_1": [5, 5, 5],   # constant -> NaN
            "s_2": [3, 1, 4],   # |r| < 1.0
        }
    )

    with pytest.raises(ValueError, match="No sensors"):
        select_correlated_sensors(df, threshold=1.0)
