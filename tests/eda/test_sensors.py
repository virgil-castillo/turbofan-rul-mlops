"""Tests for turbofan.eda.sensors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.eda.sensors import (
    compute_correlation_matrix,
    compute_sensor_stats,
    estimate_noise,
)


def _make_sensor_df() -> pd.DataFrame:
    """Build a DataFrame with 2 engines, 10 cycles each, 3 sensors."""
    rng = np.random.default_rng(42)
    rows = []
    for engine_id in [1, 2]:
        for cycle in range(1, 11):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "s_1": float(cycle) + rng.normal(0, 0.1),
                    "s_2": 200.0 + rng.normal(0, 0.5),
                    "s_3": float(cycle) * 2 + rng.normal(0, 0.2),
                }
            )
    return pd.DataFrame(rows)


def test_compute_sensor_stats_row_count() -> None:
    """Returns one row per sensor column."""
    df = _make_sensor_df()
    result = compute_sensor_stats(df)
    assert len(result) == 3  # s_1, s_2, s_3


def test_compute_sensor_stats_columns() -> None:
    """Output has mean, std, min, max, skewness, kurtosis."""
    df = _make_sensor_df()
    result = compute_sensor_stats(df)
    expected_cols = {"mean", "std", "min", "max", "skewness", "kurtosis"}
    assert set(result.columns) == expected_cols


def test_compute_sensor_stats_index_names() -> None:
    """Index contains sensor column names."""
    df = _make_sensor_df()
    result = compute_sensor_stats(df)
    assert list(result.index) == ["s_1", "s_2", "s_3"]


def test_correlation_matrix_is_square() -> None:
    """Correlation matrix has same rows and columns."""
    df = _make_sensor_df()
    cols = ["s_1", "s_2", "s_3"]
    result = compute_correlation_matrix(df, cols)
    assert result.shape == (3, 3)


def test_correlation_matrix_diagonal_is_one() -> None:
    """Diagonal of correlation matrix is 1.0."""
    df = _make_sensor_df()
    cols = ["s_1", "s_2", "s_3"]
    result = compute_correlation_matrix(df, cols)
    for col in cols:
        assert abs(result.loc[col, col] - 1.0) < 1e-10


def test_correlation_matrix_matches_input_columns() -> None:
    """Output shape matches the number of input columns."""
    df = _make_sensor_df()
    cols = ["s_1", "s_3"]
    result = compute_correlation_matrix(df, cols)
    assert result.shape == (2, 2)


def test_estimate_noise_positive_values() -> None:
    """Noise estimates are positive for all sensors."""
    df = _make_sensor_df()
    result = estimate_noise(df, ["s_1", "s_2", "s_3"], window=3)
    assert (result["mean_rolling_std"] > 0).all()


def test_estimate_noise_sensor_column() -> None:
    """Output has a 'sensor' column listing analyzed sensors."""
    df = _make_sensor_df()
    result = estimate_noise(df, ["s_1", "s_2"], window=3)
    assert list(result["sensor"]) == ["s_1", "s_2"]
