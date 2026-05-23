"""Tests for turbofan.features.pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.features.pipeline import build_feature_pipeline


def _make_train_df() -> pd.DataFrame:
    """Training data: 2 engines, 15 cycles, 3 sensors (1 constant)."""
    rng = np.random.default_rng(42)
    rows = []
    for eid in [1, 2]:
        for cyc in range(1, 16):
            rows.append(
                {
                    "engine_id": eid,
                    "cycle": cyc,
                    "op_1": 0.0,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    "s_1": float(cyc) + rng.normal(0, 0.5),
                    "s_2": 200.0,
                    "s_3": 80.0 + rng.normal(0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def _make_test_df() -> pd.DataFrame:
    """Test data: 1 engine, 10 cycles, same columns."""
    rng = np.random.default_rng(99)
    rows = []
    for cyc in range(1, 11):
        rows.append(
            {
                "engine_id": 3,
                "cycle": cyc,
                "op_1": 0.0,
                "op_2": 0.0,
                "op_3": 0.0,
                "s_1": float(cyc) + rng.normal(0, 0.5),
                "s_2": 200.0,
                "s_3": 80.0 + rng.normal(0, 1.0),
            }
        )
    return pd.DataFrame(rows)


def test_fit_transform_no_nans() -> None:
    """Pipeline output has no NaN values."""
    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[3, 5])
    result = pipe.fit_transform(train)
    assert not result.isna().any().any()


def test_transform_test_no_nans() -> None:
    """Fit on train, transform test produces no NaNs."""
    train = _make_train_df()
    test = _make_test_df()
    pipe = build_feature_pipeline(windows=[3, 5])
    pipe.fit(train)
    result = pipe.transform(test)
    assert not result.isna().any().any()


def test_constant_sensor_dropped() -> None:
    """Constant sensor s_2 is removed by the pipeline."""
    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[3])
    result = pipe.fit_transform(train)
    assert "s_2" not in result.columns
    s2_rolling = [
        c for c in result.columns if c.startswith("s_2_")
    ]
    assert s2_rolling == []


def test_rolling_columns_present() -> None:
    """Rolling feature columns exist in output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[5])
    result = pipe.fit_transform(train)
    assert "s_1_rmean_5" in result.columns
    assert "s_3_rstd_5" in result.columns


def test_output_is_dataframe() -> None:
    """Pipeline returns a pandas DataFrame, not numpy array."""
    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[3])
    result = pipe.fit_transform(train)
    assert isinstance(result, pd.DataFrame)


def test_pipeline_named_steps() -> None:
    """Pipeline has the three expected named steps."""
    pipe = build_feature_pipeline()
    assert "sensor_dropper" in pipe.named_steps
    assert "rolling_features" in pipe.named_steps
    assert "normalizer" in pipe.named_steps


def test_multi_condition_pipeline() -> None:
    """Pipeline handles data with multiple op conditions."""
    rng = np.random.default_rng(7)
    rows = []
    for eid, op in [(1, 1.0), (2, 1.0), (3, 2.0), (4, 2.0)]:
        for cyc in range(1, 11):
            rows.append(
                {
                    "engine_id": eid,
                    "cycle": cyc,
                    "op_1": op,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    "s_1": op * 100 + rng.normal(0, 1.0),
                }
            )
    df = pd.DataFrame(rows)
    pipe = build_feature_pipeline(windows=[3])
    result = pipe.fit_transform(df)
    assert not result.isna().any().any()


def test_joblib_serialization() -> None:
    """Fitted pipeline survives joblib round-trip."""
    import io

    import joblib

    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[3])
    pipe.fit(train)

    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    buffer.seek(0)
    loaded = joblib.load(buffer)

    test = _make_test_df()
    result_original = pipe.transform(test)
    result_loaded = loaded.transform(test)
    pd.testing.assert_frame_equal(result_original, result_loaded)
