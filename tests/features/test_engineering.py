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


def test_lag_backfills_within_engine() -> None:
    """Lag is backfilled for early cycles within each engine (no cross-engine leak)."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2, 2],
        "s_1": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1])
    result = eng.fit_transform(df)
    # Engine 1 cycle 1 has no prior cycle: backfill from cycle 2 value
    e1_lag = result.loc[df["engine_id"] == 1, "s_1_lag_1"].tolist()
    assert e1_lag[0] == pytest.approx(e1_lag[1])  # backfilled to cycle 2's lag value
    # Engine 2 cycle 1 should not be affected by engine 1
    e2_lag = result.loc[df["engine_id"] == 2, "s_1_lag_1"].tolist()
    assert e2_lag[0] == pytest.approx(100.0)  # backfilled from engine 2's own data


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
    for fs in ["raw", "rolling_mean", "rolling_stats", "lag"]:
        kwargs = {"windows": [3]} if "rolling" in fs else {"lag_steps": [1]}
        eng = FeatureEngineer(feature_set=fs, **kwargs)  # type: ignore[arg-type]
        result = eng.fit_transform(df)
        assert "engine_id" not in result.columns, f"engine_id in output for {fs}"
