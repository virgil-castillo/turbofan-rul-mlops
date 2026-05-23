"""Tests for turbofan.features.rolling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.features.rolling import RollingFeatureExtractor


def _make_two_engine_df() -> pd.DataFrame:
    """Two engines, 10 cycles each, 2 sensors."""
    rng = np.random.default_rng(42)
    rows = []
    for eid in [1, 2]:
        for cyc in range(1, 11):
            rows.append(
                {
                    "engine_id": eid,
                    "cycle": cyc,
                    "op_1": 0.0,
                    "s_1": float(cyc) + rng.normal(0, 0.1),
                    "s_2": 50.0 + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_correct_number_of_new_columns() -> None:
    """2 sensors x 2 windows x 4 stats = 16 new columns."""
    df = _make_two_engine_df()
    ext = RollingFeatureExtractor(windows=[5, 10])
    result = ext.fit_transform(df)
    original_cols = len(df.columns)
    expected_new = 2 * 2 * 4  # sensors * windows * stats
    assert len(result.columns) == original_cols + expected_new


def test_column_naming_convention() -> None:
    """Rolling columns follow {sensor}_{stat}_{window} pattern."""
    df = _make_two_engine_df()
    ext = RollingFeatureExtractor(windows=[5])
    result = ext.fit_transform(df)
    assert "s_1_rmean_5" in result.columns
    assert "s_1_rstd_5" in result.columns
    assert "s_1_rmin_5" in result.columns
    assert "s_1_rmax_5" in result.columns


def test_rolling_mean_on_constant_series() -> None:
    """Rolling mean of a constant series equals that constant."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
            "s_1": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    ext = RollingFeatureExtractor(windows=[3])
    result = ext.fit_transform(df)
    assert np.allclose(result["s_1_rmean_3"], 10.0)


def test_no_cross_engine_bleed() -> None:
    """Rolling stats reset at engine boundaries."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "s_1": [100.0, 200.0, 0.0, 0.0],
        }
    )
    ext = RollingFeatureExtractor(windows=[3])
    result = ext.fit_transform(df)
    engine2_first = result.loc[
        result["engine_id"] == 2, "s_1_rmean_3"
    ].iloc[0]
    assert engine2_first == 0.0


def test_min_periods_fills_early_cycles() -> None:
    """First cycle of each engine has values, not NaN."""
    df = _make_two_engine_df()
    ext = RollingFeatureExtractor(windows=[5])
    result = ext.fit_transform(df)
    first_rows = result.groupby("engine_id").first()
    assert not first_rows["s_1_rmean_5"].isna().any()
    assert not first_rows["s_1_rstd_5"].isna().any()
    assert not first_rows["s_1_rmin_5"].isna().any()
    assert not first_rows["s_1_rmax_5"].isna().any()


def test_original_columns_preserved() -> None:
    """Original columns remain unchanged."""
    df = _make_two_engine_df()
    ext = RollingFeatureExtractor(windows=[5])
    result = ext.fit_transform(df)
    pd.testing.assert_series_equal(
        result["s_1"], df["s_1"], check_names=True
    )
