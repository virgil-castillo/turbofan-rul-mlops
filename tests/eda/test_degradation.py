"""Tests for turbofan.eda.degradation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.data.labels import compute_rul_labels
from turbofan.eda.degradation import (
    compute_rul_curves,
    compute_sensor_trends,
    select_informative_sensors,
)


def _make_degradation_df() -> pd.DataFrame:
    """Build a DataFrame with 2 engines showing degradation in s_1."""
    rng = np.random.default_rng(42)
    rows = []
    for engine_id in [1, 2]:
        n_cycles = 50 if engine_id == 1 else 30
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "s_1": float(cycle) * 0.5
                    + rng.normal(0, 0.1),
                    "s_2": rng.normal(100, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_compute_rul_curves_adds_rul_column() -> None:
    """Output DataFrame has a 'rul' column."""
    df = _make_degradation_df()
    result = compute_rul_curves(df, max_rul=125)
    assert "rul" in result.columns


def test_compute_rul_curves_matches_labels() -> None:
    """RUL column matches compute_rul_labels output."""
    df = _make_degradation_df()
    result = compute_rul_curves(df, max_rul=125)
    expected = compute_rul_labels(df, max_rul=125)
    pd.testing.assert_series_equal(result["rul"], expected)


def test_compute_rul_curves_preserves_original_columns() -> None:
    """Original columns are preserved in the output."""
    df = _make_degradation_df()
    result = compute_rul_curves(df, max_rul=125)
    for col in df.columns:
        assert col in result.columns


def test_compute_sensor_trends_smoother_than_raw() -> None:
    """Smoothed sensor values have lower variance than raw."""
    df = _make_degradation_df()
    smoothed = compute_sensor_trends(df, ["s_1"], window=5)
    raw_var = df.groupby("engine_id")["s_1"].var().mean()
    smooth_var = (
        smoothed.groupby("engine_id")["s_1"].var().mean()
    )
    assert smooth_var < raw_var


def test_compute_sensor_trends_output_columns() -> None:
    """Output has engine_id, cycle, and the requested sensor columns."""
    df = _make_degradation_df()
    result = compute_sensor_trends(df, ["s_1", "s_2"], window=5)
    assert "engine_id" in result.columns
    assert "cycle" in result.columns
    assert "s_1" in result.columns
    assert "s_2" in result.columns


def test_select_informative_sensors_includes_correlated() -> None:
    """Selects s_1 which is correlated with RUL (degrades over time)."""
    df = _make_degradation_df()
    rul = compute_rul_labels(df, max_rul=125)
    result = select_informative_sensors(df, rul, threshold=0.1)
    assert "s_1" in result


def test_select_informative_sensors_excludes_uncorrelated() -> None:
    """Excludes s_2 which is random noise uncorrelated with RUL."""
    df = _make_degradation_df()
    rul = compute_rul_labels(df, max_rul=125)
    result = select_informative_sensors(df, rul, threshold=0.3)
    assert "s_2" not in result
