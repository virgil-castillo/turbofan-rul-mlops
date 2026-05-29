"""Tests for turbofan.features.engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.features.engineering import FeatureEngineer


def _sensor_df(n_engines: int = 2, n_cycles: int = 10) -> pd.DataFrame:
    """Sensor-only DataFrame with engine_id for groupby testing."""
    rng = np.random.default_rng(0)
    rows = []
    for eid in range(1, n_engines + 1):
        for cyc in range(1, n_cycles + 1):
            rows.append({
                "engine_id": eid,
                "s_1": float(cyc) + rng.normal(0, 0.1),
                "s_2": 50.0 + rng.normal(0, 0.5),
                "s_3": float(eid) * 10 + rng.normal(0, 0.2),
            })
    return pd.DataFrame(rows)


def test_raw_returns_sensor_columns_only() -> None:
    """feature_set=raw returns only s_* columns with no engine_id."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw")
    result = eng.fit_transform(df)
    assert list(result.columns) == ["s_1", "s_2", "s_3"]
    assert "engine_id" not in result.columns


def test_raw_feature_cols_attribute() -> None:
    """feature_cols_ matches the output column list for raw."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw")
    eng.fit(df)
    assert eng.feature_cols_ == ["s_1", "s_2", "s_3"]


def test_rolling_mean_columns() -> None:
    """rolling_mean produces {sensor}_rmean_{window} columns."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_mean", windows=[5, 10])
    result = eng.fit_transform(df)
    expected = [
        "s_1_rmean_5", "s_2_rmean_5", "s_3_rmean_5",
        "s_1_rmean_10", "s_2_rmean_10", "s_3_rmean_10",
    ]
    assert list(result.columns) == expected


def test_rolling_stats_columns() -> None:
    """rolling_stats produces mean/std/min/max columns per sensor per window."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_stats", windows=[5])
    result = eng.fit_transform(df)
    for stat in ["rmean", "rstd", "rmin", "rmax"]:
        for sensor in ["s_1", "s_2", "s_3"]:
            assert f"{sensor}_{stat}_5" in result.columns
    assert "engine_id" not in result.columns


def test_raw_plus_rolling_mean_columns() -> None:
    """raw_plus_rolling_mean produces raw + rolling_mean columns."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw_plus_rolling_mean", windows=[5])
    result = eng.fit_transform(df)
    assert "s_1" in result.columns
    assert "s_1_rmean_5" in result.columns
    assert "engine_id" not in result.columns


def test_raw_plus_rolling_stats_columns() -> None:
    """raw_plus_rolling_stats produces raw + all rolling stat columns."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw_plus_rolling_stats", windows=[5])
    result = eng.fit_transform(df)
    assert "s_1" in result.columns
    assert "s_1_rmean_5" in result.columns
    assert "s_1_rstd_5" in result.columns
    assert "engine_id" not in result.columns


def test_lag_columns() -> None:
    """lag produces {sensor}_lag_{step} columns."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1, 2])
    result = eng.fit_transform(df)
    for step in [1, 2]:
        for sensor in ["s_1", "s_2", "s_3"]:
            assert f"{sensor}_lag_{step}" in result.columns
    assert "engine_id" not in result.columns


def test_rolling_no_nan_min_periods() -> None:
    """Rolling features have no NaN due to min_periods=1."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_stats", windows=[10])
    result = eng.fit_transform(df)
    assert not result.isna().any().any()


def test_rolling_respects_engine_boundaries() -> None:
    """Rolling mean for engine 2 cycle 1 is unaffected by engine 1 values."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "s_1": [100.0, 200.0, 5.0, 10.0],
    })
    eng = FeatureEngineer(feature_set="rolling_mean", windows=[5])
    result = eng.fit_transform(df)
    engine2_first_rmean = result.loc[df["engine_id"] == 2, "s_1_rmean_5"].iloc[0]
    assert engine2_first_rmean == pytest.approx(5.0)


def test_lag_first_cycle_yields_zero() -> None:
    """Cycle 1 has no prior history so backfill makes diff=0, feature=0."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2, 2],
        "s_1": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1])
    result = eng.fit_transform(df)
    # Both engines: cycle 1 diff is zero because backfill = current value
    assert result.iloc[0]["s_1_lag_1"] == pytest.approx(0.0)
    assert result.iloc[3]["s_1_lag_1"] == pytest.approx(0.0)


def test_unsupported_feature_set_raises() -> None:
    """Unknown feature_set raises ValueError on fit."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="fft")
    with pytest.raises(ValueError, match="Unsupported feature_set"):
        eng.fit(df)


def test_feature_cols_attribute_rolling_stats() -> None:
    """feature_cols_ is set correctly on fit for rolling_stats."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_stats", windows=[5])
    eng.fit(df)
    assert "s_1_rmean_5" in eng.feature_cols_
    assert "s_1_rstd_5" in eng.feature_cols_
    assert len(eng.feature_cols_) == 3 * 4 * 1  # 3 sensors * 4 stats * 1 window


def test_no_engine_id_in_output() -> None:
    """engine_id is never present in transform output."""
    df = _sensor_df()
    for fs in ["raw", "rolling_mean", "rolling_stats", "lag", "raw_plus_lag"]:
        kwargs = {"windows": [3]} if "rolling" in fs else {"lag_steps": [1]}
        eng = FeatureEngineer(feature_set=fs, **kwargs)  # type: ignore[arg-type]
        result = eng.fit_transform(df)
        assert "engine_id" not in result.columns, f"engine_id in output for {fs}"


def test_raw_plus_lag_columns() -> None:
    """raw_plus_lag produces raw sensor columns followed by lag columns."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw_plus_lag", lag_steps=[1])
    result = eng.fit_transform(df)
    expected = ["s_1", "s_2", "s_3", "s_1_lag_1", "s_2_lag_1", "s_3_lag_1"]
    assert list(result.columns) == expected
    assert "engine_id" not in result.columns


def test_raw_plus_lag_feature_cols_attribute() -> None:
    """feature_cols_ matches raw + lag column list for raw_plus_lag."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw_plus_lag", lag_steps=[1, 2])
    eng.fit(df)
    expected = (
        ["s_1", "s_2", "s_3"]
        + ["s_1_lag_1", "s_2_lag_1", "s_3_lag_1"]
        + ["s_1_lag_2", "s_2_lag_2", "s_3_lag_2"]
    )
    assert eng.feature_cols_ == expected


def test_lag_computes_normalized_difference_lag1() -> None:
    """lag features are (x[t] - x[t-N]) / rolling_mean(x, N), not raw lag values."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "s_1": [100.0, 110.0, 120.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1])
    result = eng.fit_transform(df)
    # cycle 1: diff=0 (backfill), mean=100, feature=0
    # cycle 2: diff=10, mean(w=1)=110, feature=10/110
    # cycle 3: diff=10, mean(w=1)=120, feature=10/120
    values = result["s_1_lag_1"].tolist()
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(10.0 / 110.0)
    assert values[2] == pytest.approx(10.0 / 120.0)


def test_lag_computes_normalized_difference_lag2() -> None:
    """lag-2 feature normalizes by rolling_mean over 2 cycles."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "s_1": [100.0, 110.0, 120.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[2])
    result = eng.fit_transform(df)
    # shift(2).bfill() = [100, 100, 100]
    # rolling_mean(w=2): [100, 105, 115]
    # diffs: [0, 10, 20]; features: [0/100, 10/105, 20/115]
    values = result["s_1_lag_2"].tolist()
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(10.0 / 105.0)
    assert values[2] == pytest.approx(20.0 / 115.0)


def test_lag_constant_sensor_yields_zero() -> None:
    """Constant sensor produces zero lag-diff features (no change to measure)."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 1],
        "s_1": [50.0, 50.0, 50.0, 50.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1, 3])
    result = eng.fit_transform(df)
    assert result.abs().max().max() == pytest.approx(0.0, abs=1e-9)


def test_lag_no_nan_in_output() -> None:
    """Lag-diff output contains no NaN values."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1, 3])
    result = eng.fit_transform(df)
    assert not result.isna().any().any()


def test_lag_diff_respects_engine_boundaries() -> None:
    """Lag-diff for engine 2 cycle 1 is not contaminated by engine 1 values."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2, 2],
        "s_1": [10.0, 20.0, 30.0, 100.0, 110.0, 120.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1])
    result = eng.fit_transform(df)
    # Engine 2 cycle 1: backfilled → diff=0, feature=0
    e2 = result.loc[df["engine_id"] == 2, "s_1_lag_1"].tolist()
    assert e2[0] == pytest.approx(0.0)
