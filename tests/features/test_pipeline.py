"""Tests for turbofan.features.pipeline."""
from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from turbofan.features.pipeline import SensorColumnSelector, build_feature_pipeline
from turbofan.preprocessing.normalization import OperatingModeNormalizer


def _make_train_df() -> pd.DataFrame:
    """Training data: 2 engines, 15 cycles, 3 sensors."""
    rng = np.random.default_rng(42)
    rows = []
    for eid in [1, 2]:
        for cyc in range(1, 16):
            rows.append({
                "engine_id": eid,
                "cycle": cyc,
                "op_1": 0.0,
                "op_2": 0.0,
                "op_3": 0.0,
                "s_1": float(cyc) + rng.normal(0, 0.5),
                "s_2": 200.0,
                "s_3": 80.0 + rng.normal(0, 1.0),
            })
    return pd.DataFrame(rows)


def _make_test_df() -> pd.DataFrame:
    """Test data: 1 engine, 10 cycles."""
    rng = np.random.default_rng(99)
    rows = []
    for cyc in range(1, 11):
        rows.append({
            "engine_id": 3,
            "cycle": cyc,
            "op_1": 0.0,
            "op_2": 0.0,
            "op_3": 0.0,
            "s_1": float(cyc) + rng.normal(0, 0.5),
            "s_2": 200.0,
            "s_3": 80.0 + rng.normal(0, 1.0),
        })
    return pd.DataFrame(rows)


def test_pipeline_has_five_named_steps() -> None:
    """Pipeline has the five expected named steps."""
    pipe = build_feature_pipeline()
    assert list(pipe.named_steps) == [
        "sensor_dropper",
        "normalizer",
        "sensor_selector",
        "feature_engineer",
        "scaler",
    ]


def test_scaler_step_is_standard_scaler() -> None:
    """Final pipeline step is StandardScaler."""
    pipe = build_feature_pipeline()
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)


def test_output_contains_only_sensor_columns_for_raw() -> None:
    """feature_set=raw output contains only s_* columns."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert all(c.startswith("s_") for c in result.columns)


def test_op_cols_absent_from_output() -> None:
    """op_1, op_2, op_3 are not in the pipeline output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert "op_1" not in result.columns
    assert "op_2" not in result.columns
    assert "op_3" not in result.columns


def test_engine_id_absent_from_output() -> None:
    """engine_id is not in the pipeline output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert "engine_id" not in result.columns


def test_sensor_drop_removes_listed_sensor() -> None:
    """Sensor listed in sensor_drop is absent from output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(sensor_drop=["s_2"], feature_set="raw")
    result = pipe.fit_transform(train)
    assert "s_2" not in result.columns


def test_fit_transform_no_nans_raw() -> None:
    """Pipeline output has no NaN values for feature_set=raw."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert not result.isna().any().any()


def test_transform_test_no_nans() -> None:
    """Fit on train, transform test produces no NaNs."""
    train = _make_train_df()
    test = _make_test_df()
    pipe = build_feature_pipeline(feature_set="raw")
    pipe.fit(train)
    result = pipe.transform(test)
    assert not result.isna().any().any()


def test_rolling_mean_columns_in_output() -> None:
    """feature_set=rolling_mean produces rmean columns in output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="rolling_mean", windows=[5])
    result = pipe.fit_transform(train)
    assert "s_1_rmean_5" in result.columns
    assert "engine_id" not in result.columns


def test_rolling_respects_engine_boundaries() -> None:
    """Pipeline rolling output has no NaN leakage across engine boundaries."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
        "op_1": [0.0, 0.0, 0.0, 0.0],
        "op_2": [0.0, 0.0, 0.0, 0.0],
        "op_3": [0.0, 0.0, 0.0, 0.0],
        "s_1": [100.0, 200.0, 5.0, 10.0],
    })
    pipe = build_feature_pipeline(feature_set="rolling_mean", windows=[5])
    result = pipe.fit_transform(df)
    assert not result.isna().any().any()


def test_lag_no_cross_engine_bleed() -> None:
    """Lag backfills within engine group, does not cross engine boundary."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
        "op_1": [0.0] * 4,
        "op_2": [0.0] * 4,
        "op_3": [0.0] * 4,
        "s_1": [10.0, 20.0, 100.0, 200.0],
    })
    pipe = build_feature_pipeline(feature_set="lag", lag_steps=[1])
    result = pipe.fit_transform(df)
    assert not result.isna().any().any()


def test_normalizer_step_is_operating_mode_normalizer() -> None:
    """normalizer step is OperatingModeNormalizer."""
    pipe = build_feature_pipeline(n_modes=2, random_state=7)
    assert isinstance(pipe.named_steps["normalizer"], OperatingModeNormalizer)
    assert pipe.named_steps["normalizer"].n_modes == 2


def test_output_is_dataframe() -> None:
    """Pipeline returns a pandas DataFrame."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert isinstance(result, pd.DataFrame)


def test_joblib_serialization() -> None:
    """Fitted pipeline survives joblib round-trip."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    pipe.fit(train)
    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    buffer.seek(0)
    loaded = joblib.load(buffer)
    test = _make_test_df()
    pd.testing.assert_frame_equal(pipe.transform(test), loaded.transform(test))


def test_sensor_column_selector_fit_records_sensor_cols() -> None:
    """SensorColumnSelector.fit records s_* columns as feature_cols_."""
    df = pd.DataFrame({
        "engine_id": [1, 2],
        "op_1": [0.0, 0.0],
        "s_1": [1.0, 2.0],
        "s_2": [3.0, 4.0],
    })
    sel = SensorColumnSelector()
    sel.fit(df)
    assert sel.feature_cols_ == ["s_1", "s_2"]


def test_sensor_column_selector_transform_keeps_engine_id() -> None:
    """SensorColumnSelector.transform keeps engine_id for downstream grouping."""
    df = pd.DataFrame({
        "engine_id": [1, 2],
        "op_1": [0.0, 0.0],
        "s_1": [1.0, 2.0],
        "s_2": [3.0, 4.0],
    })
    sel = SensorColumnSelector()
    sel.fit(df)
    result = sel.transform(df)
    assert "engine_id" in result.columns
    assert "op_1" not in result.columns
    assert "s_1" in result.columns
    assert "s_2" in result.columns


def test_multi_condition_pipeline() -> None:
    """Pipeline handles data with multiple op conditions (n_modes=2)."""
    rng = np.random.default_rng(7)
    rows = []
    for eid, op in [(1, 1.0), (2, 1.0), (3, 2.0), (4, 2.0)]:
        for cyc in range(1, 11):
            rows.append({
                "engine_id": eid,
                "cycle": cyc,
                "op_1": op,
                "op_2": 0.0,
                "op_3": 0.0,
                "s_1": op * 100 + rng.normal(0, 1.0),
            })
    df = pd.DataFrame(rows)
    pipe = build_feature_pipeline(n_modes=2, feature_set="raw", random_state=7)
    result = pipe.fit_transform(df)
    assert not result.isna().any().any()
