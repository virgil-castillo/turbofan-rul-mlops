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


def test_lag_first_cycle_backfills_with_engine_first_value() -> None:
    """Cycle 1 lag values are backfilled within each engine."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2, 2],
        "s_1": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1])
    result = eng.fit_transform(df)
    assert result.iloc[0]["s_1_lag_1"] == pytest.approx(10.0)
    assert result.iloc[3]["s_1_lag_1"] == pytest.approx(100.0)


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


def test_lag_returns_shifted_previous_value_lag1() -> None:
    """lag features are shifted prior values within each engine."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "s_1": [100.0, 110.0, 120.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1])
    result = eng.fit_transform(df)
    # cycle 1: no prior history, so backfill uses the first value.
    # cycle 2: previous value is 100.
    # cycle 3: previous value is 110.
    values = result["s_1_lag_1"].tolist()
    assert values == pytest.approx([100.0, 100.0, 110.0])


def test_lag_returns_shifted_previous_value_lag2() -> None:
    """lag-2 features are values from two cycles earlier."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "s_1": [100.0, 110.0, 120.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[2])
    result = eng.fit_transform(df)
    # shift(2).bfill() = [100, 100, 100]
    values = result["s_1_lag_2"].tolist()
    assert values == pytest.approx([100.0, 100.0, 100.0])


def test_lag_constant_sensor_returns_constant_history() -> None:
    """Constant sensors produce constant lagged values."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 1],
        "s_1": [50.0, 50.0, 50.0, 50.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1, 3])
    result = eng.fit_transform(df)
    assert result["s_1_lag_1"].tolist() == pytest.approx(
        [50.0, 50.0, 50.0, 50.0]
    )
    assert result["s_1_lag_3"].tolist() == pytest.approx(
        [50.0, 50.0, 50.0, 50.0]
    )


def test_lag_no_nan_in_output() -> None:
    """Lag output contains no NaN values."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1, 3])
    result = eng.fit_transform(df)
    assert not result.isna().any().any()


def test_lag_respects_engine_boundaries() -> None:
    """Lag values for engine 2 are not contaminated by engine 1 values."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2, 2],
        "s_1": [10.0, 20.0, 30.0, 100.0, 110.0, 120.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1])
    result = eng.fit_transform(df)
    e2 = result.loc[df["engine_id"] == 2, "s_1_lag_1"].tolist()
    assert e2 == pytest.approx([100.0, 100.0, 110.0])


# ── rolling_std ───────────────────────────────────────────────────────────────


def test_rolling_std_column_names_and_order() -> None:
    """rolling_std produces {sensor}_rstd_{window} columns in window-major order."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_std", windows=[5, 10])
    result = eng.fit_transform(df)
    expected = [
        "s_1_rstd_5", "s_2_rstd_5", "s_3_rstd_5",
        "s_1_rstd_10", "s_2_rstd_10", "s_3_rstd_10",
    ]
    assert list(result.columns) == expected


def test_rolling_std_no_engine_id_in_output() -> None:
    """engine_id is never present in rolling_std transform output."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_std", windows=[5])
    result = eng.fit_transform(df)
    assert "engine_id" not in result.columns


def test_rolling_std_shape() -> None:
    """rolling_std output has correct shape: same rows, n_windows * n_sensors cols."""
    df = _sensor_df(n_engines=2, n_cycles=10)
    eng = FeatureEngineer(feature_set="rolling_std", windows=[5, 10])
    result = eng.fit_transform(df)
    assert result.shape == (20, 6)  # 2 engines * 10 cycles, 2 windows * 3 sensors


def test_rolling_std_feature_cols_attribute() -> None:
    """feature_cols_ matches the transform output columns for rolling_std."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_std", windows=[5])
    eng.fit(df)
    assert eng.feature_cols_ == ["s_1_rstd_5", "s_2_rstd_5", "s_3_rstd_5"]
    result = eng.transform(df)
    assert list(result.columns) == eng.feature_cols_


def test_rolling_std_first_cycle_nan_filled_with_zero() -> None:
    """rolling_std fills first-cycle NaN (std of single value) with 0.0."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "s_1": [10.0, 20.0, 30.0],
    })
    eng = FeatureEngineer(feature_set="rolling_std", windows=[5])
    result = eng.fit_transform(df)
    # Column presence is asserted first; value check requires the column to exist.
    cols = list(result.columns)
    assert "s_1_rstd_5" in result.columns, f"expected s_1_rstd_5 in {cols}"
    # std of a single value is NaN → must be filled to 0.0
    assert result["s_1_rstd_5"].iloc[0] == pytest.approx(0.0)


def test_rolling_std_no_nan_in_output() -> None:
    """rolling_std output contains no NaN values (min_periods=1 + fillna)."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_std", windows=[5, 10])
    result = eng.fit_transform(df)
    assert not result.isna().any().any()


def test_rolling_std_respects_engine_boundaries() -> None:
    """Engine 2 cycle-1 std is unaffected by engine 1 values."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "s_1": [100.0, 200.0, 5.0, 10.0],
    })
    eng = FeatureEngineer(feature_set="rolling_std", windows=[5])
    result = eng.fit_transform(df)
    cols = list(result.columns)
    assert "s_1_rstd_5" in result.columns, f"expected s_1_rstd_5 in {cols}"
    # Engine 2 cycle 1: single point → std NaN → 0.0
    e2_first = result.loc[df["engine_id"] == 2, "s_1_rstd_5"].iloc[0]
    assert e2_first == pytest.approx(0.0)
    # Engine 2 cycle 2: std of [5.0, 10.0], not contaminated by engine 1
    e2_second = result.loc[df["engine_id"] == 2, "s_1_rstd_5"].iloc[1]
    expected_std = pd.Series([5.0, 10.0]).std()
    assert e2_second == pytest.approx(expected_std)


# ── rolling_slope ─────────────────────────────────────────────────────────────


def test_rolling_slope_column_names_and_order() -> None:
    """rolling_slope produces {sensor}_rslope_{window} columns in window-major order."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_slope", windows=[5, 10])
    result = eng.fit_transform(df)
    expected = [
        "s_1_rslope_5", "s_2_rslope_5", "s_3_rslope_5",
        "s_1_rslope_10", "s_2_rslope_10", "s_3_rslope_10",
    ]
    assert list(result.columns) == expected


def test_rolling_slope_no_engine_id_in_output() -> None:
    """engine_id is never present in rolling_slope transform output."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_slope", windows=[5])
    result = eng.fit_transform(df)
    assert "engine_id" not in result.columns


def test_rolling_slope_shape() -> None:
    """rolling_slope output has correct shape."""
    df = _sensor_df(n_engines=2, n_cycles=10)
    eng = FeatureEngineer(feature_set="rolling_slope", windows=[5, 10])
    result = eng.fit_transform(df)
    assert result.shape == (20, 6)


def test_rolling_slope_feature_cols_attribute() -> None:
    """feature_cols_ matches transform output columns for rolling_slope."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_slope", windows=[5])
    eng.fit(df)
    assert eng.feature_cols_ == ["s_1_rslope_5", "s_2_rslope_5", "s_3_rslope_5"]
    result = eng.transform(df)
    assert list(result.columns) == eng.feature_cols_


def test_rolling_slope_known_slope_value() -> None:
    """Sensor increasing by 1 per cycle yields slope=1.0 once window is full."""
    # s_1 = 1,2,3,4,5 → slope = 1.0 everywhere once window=5 is satisfied
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 1, 1],
        "s_1": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    eng = FeatureEngineer(feature_set="rolling_slope", windows=[5])
    result = eng.fit_transform(df)
    cols = list(result.columns)
    assert "s_1_rslope_5" in result.columns, f"expected s_1_rslope_5 in {cols}"
    # At row 4 (cycle 5), window = [1,2,3,4,5], t = [0,1,2,3,4]
    # tbar=2, xbar=3; sum((t-tbar)*(x-xbar))=10; sum((t-tbar)^2)=10 → slope=1.0
    assert result["s_1_rslope_5"].iloc[4] == pytest.approx(1.0)


def test_rolling_slope_window_1_yields_zero() -> None:
    """Window size 1 → length-1 window → zero denominator → slope=0.0."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "s_1": [10.0, 20.0, 30.0],
    })
    eng = FeatureEngineer(feature_set="rolling_slope", windows=[1])
    result = eng.fit_transform(df)
    cols = list(result.columns)
    assert "s_1_rslope_1" in result.columns, f"expected s_1_rslope_1 in {cols}"
    # Every window has exactly 1 point → denominator=0 → slope=0.0 for all rows
    assert result["s_1_rslope_1"].iloc[0] == pytest.approx(0.0)
    assert result["s_1_rslope_1"].iloc[1] == pytest.approx(0.0)
    assert result["s_1_rslope_1"].iloc[2] == pytest.approx(0.0)


def test_rolling_slope_no_nan_in_output() -> None:
    """rolling_slope output contains no NaN values."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_slope", windows=[5, 10])
    result = eng.fit_transform(df)
    assert not result.isna().any().any()


def test_rolling_slope_respects_engine_boundaries() -> None:
    """Engine 2 cycle-1 slope is unaffected by engine 1 values."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2, 2],
        "s_1": [100.0, 200.0, 300.0, 1.0, 2.0, 3.0],
    })
    eng = FeatureEngineer(feature_set="rolling_slope", windows=[5])
    result = eng.fit_transform(df)
    cols = list(result.columns)
    assert "s_1_rslope_5" in result.columns, f"expected s_1_rslope_5 in {cols}"
    # Engine 2 cycle 1: single point → slope=0.0 (length-1 window)
    e2_first = result.loc[df["engine_id"] == 2, "s_1_rslope_5"].iloc[0]
    assert e2_first == pytest.approx(0.0)
    # Engine 2 cycle 3 (index 2 of engine 2): t=[0,1,2], x=[1,2,3]
    # tbar=1, xbar=2; sum((t-tbar)*(x-xbar))=(-1)(-1)+0*0+1*1=2; sum((t-tbar)^2)=2
    # slope = 2/2 = 1.0
    e2_third = result.loc[df["engine_id"] == 2, "s_1_rslope_5"].iloc[2]
    assert e2_third == pytest.approx(1.0)


# ── rolling_delta ─────────────────────────────────────────────────────────────


def test_rolling_delta_column_names_and_order() -> None:
    """rolling_delta produces {sensor}_rdelta_{window} columns in window-major order."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_delta", windows=[5, 10])
    result = eng.fit_transform(df)
    expected = [
        "s_1_rdelta_5", "s_2_rdelta_5", "s_3_rdelta_5",
        "s_1_rdelta_10", "s_2_rdelta_10", "s_3_rdelta_10",
    ]
    assert list(result.columns) == expected


def test_rolling_delta_no_engine_id_in_output() -> None:
    """engine_id is never present in rolling_delta transform output."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_delta", windows=[5])
    result = eng.fit_transform(df)
    assert "engine_id" not in result.columns


def test_rolling_delta_shape() -> None:
    """rolling_delta output has correct shape."""
    df = _sensor_df(n_engines=2, n_cycles=10)
    eng = FeatureEngineer(feature_set="rolling_delta", windows=[5, 10])
    result = eng.fit_transform(df)
    assert result.shape == (20, 6)


def test_rolling_delta_feature_cols_attribute() -> None:
    """feature_cols_ matches transform output columns for rolling_delta."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_delta", windows=[5])
    eng.fit(df)
    assert eng.feature_cols_ == ["s_1_rdelta_5", "s_2_rdelta_5", "s_3_rdelta_5"]
    result = eng.transform(df)
    assert list(result.columns) == eng.feature_cols_


def test_rolling_delta_first_cycle_is_zero() -> None:
    """First cycle of each engine has insufficient history → backfill → 0.0."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "s_1": [10.0, 20.0, 30.0],
    })
    eng = FeatureEngineer(feature_set="rolling_delta", windows=[3])
    result = eng.fit_transform(df)
    cols = list(result.columns)
    assert "s_1_rdelta_3" in result.columns, f"expected s_1_rdelta_3 in {cols}"
    # rdelta_3 = x[t] - x[t-2]; cycle 1: no history → backfill → 0.0
    assert result["s_1_rdelta_3"].iloc[0] == pytest.approx(0.0)


def test_rolling_delta_known_value() -> None:
    """rdelta_3 at t=4 equals x[4] - x[2] (shift by w-1=2)."""
    # s_1 = [10, 20, 30, 40, 50], window=3 → shift(2)
    # rdelta_3[4] = 50 - 30 = 20
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 1, 1],
        "s_1": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    eng = FeatureEngineer(feature_set="rolling_delta", windows=[3])
    result = eng.fit_transform(df)
    cols = list(result.columns)
    assert "s_1_rdelta_3" in result.columns, f"expected s_1_rdelta_3 in {cols}"
    # Also check t=2: x[2] - x[0] = 30 - 10 = 20
    assert result["s_1_rdelta_3"].iloc[2] == pytest.approx(20.0)
    assert result["s_1_rdelta_3"].iloc[4] == pytest.approx(20.0)


def test_rolling_delta_no_nan_in_output() -> None:
    """rolling_delta output contains no NaN values (backfill covers leading history)."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_delta", windows=[5, 10])
    result = eng.fit_transform(df)
    assert not result.isna().any().any()


def test_rolling_delta_respects_engine_boundaries() -> None:
    """Engine 2 cycle-1 delta is unaffected by engine 1 values."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "s_1": [100.0, 200.0, 5.0, 10.0],
    })
    eng = FeatureEngineer(feature_set="rolling_delta", windows=[3])
    result = eng.fit_transform(df)
    cols = list(result.columns)
    assert "s_1_rdelta_3" in result.columns, f"expected s_1_rdelta_3 in {cols}"
    # Engine 2 cycle 1: shift(2) within engine → backfill → x[0]=5.0, delta=0.0
    e2_first = result.loc[df["engine_id"] == 2, "s_1_rdelta_3"].iloc[0]
    assert e2_first == pytest.approx(0.0)
