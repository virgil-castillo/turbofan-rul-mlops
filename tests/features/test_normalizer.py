"""Tests for turbofan.features.normalizer."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from turbofan.features.normalizer import OperationalNormalizer


def _make_single_condition_df() -> pd.DataFrame:
    """Single operating condition, 2 sensors."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "op_1": [0.0] * 8,
            "op_2": [0.0] * 8,
            "op_3": [0.0] * 8,
            "s_1": [
                100.0, 102.0, 104.0, 106.0,
                101.0, 103.0, 105.0, 107.0,
            ],
            "s_2": [
                50.0, 52.0, 54.0, 56.0,
                51.0, 53.0, 55.0, 57.0,
            ],
        }
    )


def _make_multi_condition_df() -> pd.DataFrame:
    """Two operating conditions with different sensor ranges."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "op_1": [
                1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0,
            ],
            "op_2": [0.0] * 8,
            "op_3": [0.0] * 8,
            "s_1": [
                98.0, 100.0, 102.0, 104.0,
                198.0, 200.0, 202.0, 204.0,
            ],
        }
    )


def test_zscore_mean_near_zero() -> None:
    """After fit_transform, sensor columns have ~0 mean."""
    df = _make_single_condition_df()
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    assert abs(result["s_1"].mean()) < 1e-10
    assert abs(result["s_2"].mean()) < 1e-10


def test_zscore_std_near_one() -> None:
    """After fit_transform, sensor columns have ~1 std."""
    df = _make_single_condition_df()
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    assert abs(result["s_1"].std() - 1.0) < 1e-10
    assert abs(result["s_2"].std() - 1.0) < 1e-10


def test_multi_condition_normalization() -> None:
    """Each condition is normalized independently."""
    df = _make_multi_condition_df()
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    cond_a = result[result["op_1"] == 1.0]["s_1"]
    cond_b = result[result["op_1"] == 2.0]["s_1"]
    assert abs(cond_a.mean()) < 1e-10
    assert abs(cond_b.mean()) < 1e-10


def test_unseen_condition_uses_global_stats() -> None:
    """Transform with an unseen condition falls back to globals."""
    train = _make_multi_condition_df()
    test = pd.DataFrame(
        {
            "engine_id": [3, 3],
            "cycle": [1, 2],
            "op_1": [9.0, 9.0],
            "op_2": [0.0, 0.0],
            "op_3": [0.0, 0.0],
            "s_1": [150.0, 152.0],
        }
    )
    norm = OperationalNormalizer()
    norm.fit(train)
    result = norm.transform(test)
    assert not result["s_1"].isna().any()


def test_zero_std_no_nan() -> None:
    """Constant sensor within a condition produces 0, not NaN."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "op_1": [0.0] * 4,
            "op_2": [0.0] * 4,
            "op_3": [0.0] * 4,
            "s_1": [100.0, 100.0, 100.0, 100.0],
        }
    )
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    assert not result["s_1"].isna().any()
    assert not np.isinf(result["s_1"]).any()


def test_integer_sensor_columns_normalize_without_dtype_warning() -> None:
    """Integer sensor columns are cast before float z-scores are assigned."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            "op_1": [0.0, 0.0],
            "op_2": [0.0, 0.0],
            "op_3": [0.0, 0.0],
            "s_1": [100, 102],
        }
    )
    norm = OperationalNormalizer()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = norm.fit_transform(df)

    dtype_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, FutureWarning)
        and "incompatible dtype" in str(warning.message)
    ]
    assert dtype_warnings == []
    assert pd.api.types.is_float_dtype(result["s_1"])


def test_non_sensor_columns_unchanged() -> None:
    """engine_id, cycle, op_cols pass through untouched."""
    df = _make_single_condition_df()
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    pd.testing.assert_series_equal(
        result["engine_id"], df["engine_id"]
    )
    pd.testing.assert_series_equal(
        result["cycle"], df["cycle"]
    )
    pd.testing.assert_series_equal(
        result["op_1"], df["op_1"]
    )
