"""Tests for turbofan.features.sensor_dropper."""
from __future__ import annotations

import pandas as pd

from turbofan.features.sensor_dropper import SensorDropper


def _make_df_with_constant() -> pd.DataFrame:
    """DataFrame with s_1 varying, s_2 constant, s_3 varying."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "op_1": [0.0, 0.0, 0.0, 0.0, 0.0],
            "s_1": [100.0, 101.0, 102.0, 100.5, 101.5],
            "s_2": [200.0, 200.0, 200.0, 200.0, 200.0],
            "s_3": [50.0, 51.0, 52.0, 50.5, 51.5],
        }
    )


def test_drops_constant_sensor() -> None:
    """Constant sensor s_2 is removed after fit + transform."""
    df = _make_df_with_constant()
    dropper = SensorDropper()
    dropper.fit(df)
    result = dropper.transform(df)
    assert "s_2" not in result.columns


def test_keeps_varying_sensors() -> None:
    """Non-constant sensors s_1 and s_3 survive."""
    df = _make_df_with_constant()
    dropper = SensorDropper()
    result = dropper.fit_transform(df)
    assert "s_1" in result.columns
    assert "s_3" in result.columns


def test_keeps_non_sensor_columns() -> None:
    """engine_id, cycle, op_1 are never dropped."""
    df = _make_df_with_constant()
    dropper = SensorDropper()
    result = dropper.fit_transform(df)
    assert "engine_id" in result.columns
    assert "cycle" in result.columns
    assert "op_1" in result.columns


def test_fit_train_transform_test() -> None:
    """Fit on train, transform on test drops same columns."""
    train = _make_df_with_constant()
    test = pd.DataFrame(
        {
            "engine_id": [3, 3],
            "cycle": [1, 2],
            "op_1": [0.0, 0.0],
            "s_1": [99.0, 100.0],
            "s_2": [999.0, 998.0],
            "s_3": [60.0, 61.0],
        }
    )
    dropper = SensorDropper()
    dropper.fit(train)
    result = dropper.transform(test)
    assert "s_2" not in result.columns
    assert "s_1" in result.columns


def test_keep_parameter_prevents_drop() -> None:
    """Force-keeping a constant sensor via the keep parameter."""
    df = _make_df_with_constant()
    dropper = SensorDropper(keep=["s_2"])
    dropper.fit(df)
    result = dropper.transform(df)
    assert "s_2" in result.columns


def test_empty_drop_list_when_all_vary() -> None:
    """No sensors dropped when all have nonzero variance."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 2],
            "cycle": [1, 1],
            "s_1": [1.0, 2.0],
            "s_2": [3.0, 4.0],
        }
    )
    dropper = SensorDropper()
    dropper.fit(df)
    assert dropper.columns_to_drop_ == []
    result = dropper.transform(df)
    assert "s_1" in result.columns
    assert "s_2" in result.columns
