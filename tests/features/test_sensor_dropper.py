"""Tests for turbofan.features.sensor_dropper."""
from __future__ import annotations

import pandas as pd

from turbofan.features.sensor_dropper import SensorDropper


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "engine_id": [1, 2],
            "cycle": [1, 1],
            "op_1": [0.0, 0.0],
            "s_1": [1.0, 2.0],
            "s_2": [3.0, 4.0],
            "s_3": [5.0, 6.0],
        }
    )


def test_drops_listed_sensors() -> None:
    """Sensors named in drop list are removed."""
    df = _make_df()
    dropper = SensorDropper(drop=["s_1", "s_2"])
    result = dropper.fit_transform(df)
    assert "s_1" not in result.columns
    assert "s_2" not in result.columns
    assert "s_3" in result.columns


def test_empty_drop_list_preserves_all_columns() -> None:
    """No sensors dropped when drop list is empty."""
    df = _make_df()
    dropper = SensorDropper()
    result = dropper.fit_transform(df)
    assert list(result.columns) == list(df.columns)


def test_keeps_non_sensor_columns() -> None:
    """engine_id, cycle, and op_* columns are never dropped."""
    df = _make_df()
    dropper = SensorDropper(drop=["s_1", "s_2", "s_3"])
    result = dropper.fit_transform(df)
    assert "engine_id" in result.columns
    assert "cycle" in result.columns
    assert "op_1" in result.columns


def test_fit_on_train_transform_on_test_drops_same_columns() -> None:
    """Dropper fitted on train removes the same columns on test data."""
    train = _make_df()
    test = pd.DataFrame(
        {
            "engine_id": [3, 3],
            "cycle": [1, 2],
            "op_1": [0.0, 0.0],
            "s_1": [9.0, 10.0],
            "s_2": [11.0, 12.0],
            "s_3": [13.0, 14.0],
        }
    )
    dropper = SensorDropper(drop=["s_1"])
    dropper.fit(train)
    result = dropper.transform(test)
    assert "s_1" not in result.columns
    assert "s_2" in result.columns


def test_columns_to_drop_attribute_reflects_config() -> None:
    """Fitted attribute mirrors the configured drop list."""
    dropper = SensorDropper(drop=["s_1", "s_5"])
    dropper.fit(_make_df())
    assert dropper.columns_to_drop_ == ["s_1", "s_5"]


def test_missing_drop_column_is_silently_ignored() -> None:
    """Column in drop list absent from DataFrame does not raise."""
    df = pd.DataFrame({"engine_id": [1], "s_2": [3.0]})
    dropper = SensorDropper(drop=["s_1"])  # s_1 not present
    result = dropper.fit_transform(df)
    assert list(result.columns) == list(df.columns)
