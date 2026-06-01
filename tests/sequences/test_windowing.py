"""Tests for fixed-length sequence windowing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.sequences.windowing import (
    WindowedSequences,
    build_final_windows,
    build_sliding_windows,
)

FEATURE_COLS = ["s_1", "s_2"]


def _df() -> pd.DataFrame:
    """Build intentionally unsorted multi-engine sequence rows.

    Returns:
        Test rows with two eligible engines and one short engine.
    """
    return pd.DataFrame(
        {
            "engine_id": [2, 1, 3, 1, 2, 1, 2],
            "cycle": [2, 3, 1, 1, 1, 2, 3],
            "s_1": [22.0, 13.0, 31.0, 11.0, 21.0, 12.0, 23.0],
            "s_2": [220.0, 130.0, 310.0, 110.0, 210.0, 120.0, 230.0],
            "rul": [1.0, 0.0, 9.0, 2.0, 2.0, 1.0, 0.0],
        }
    )


def test_sliding_windows_are_sorted_and_labeled_by_final_timestep() -> None:
    """Sliding windows sort rows, pad short engines, and use final-timestep labels."""
    windows = build_sliding_windows(_df(), FEATURE_COLS, window_size=2)

    assert isinstance(windows, WindowedSequences)
    assert windows.X.dtype == np.float32
    assert windows.y.dtype == np.float32
    # engine 1: 2 windows, engine 2: 2 windows, engine 3: 1 padded window
    assert windows.X.shape == (5, 2, 2)
    assert windows.y.tolist() == [1.0, 0.0, 1.0, 0.0, 9.0]
    assert windows.metadata[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 2},
        {"engine_id": 1, "cycle": 3},
        {"engine_id": 2, "cycle": 2},
        {"engine_id": 2, "cycle": 3},
        {"engine_id": 3, "cycle": 1},
    ]
    np.testing.assert_array_equal(
        windows.X[0],
        np.array([[11.0, 110.0], [12.0, 120.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        windows.X[2],
        np.array([[21.0, 210.0], [22.0, 220.0]], dtype=np.float32),
    )
    # engine 3 is right-zero-padded
    np.testing.assert_array_equal(
        windows.X[4],
        np.array([[31.0, 310.0], [0.0, 0.0]], dtype=np.float32),
    )
    assert windows.lengths[4] == 1
    assert bool(windows.metadata["padded"].iloc[4]) is True


def test_final_windows_return_one_window_per_engine() -> None:
    """Final windows return one window per engine; short engines are padded."""
    windows = build_final_windows(_df(), FEATURE_COLS, window_size=2)

    # engine 1: final window; engine 2: final window; engine 3: padded window
    assert windows.X.shape == (3, 2, 2)
    assert windows.y.tolist() == [0.0, 0.0, 9.0]
    assert windows.metadata[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 3},
        {"engine_id": 2, "cycle": 3},
        {"engine_id": 3, "cycle": 1},
    ]
    np.testing.assert_array_equal(
        windows.X[0],
        np.array([[12.0, 120.0], [13.0, 130.0]], dtype=np.float32),
    )
    # engine 3 is right-zero-padded
    np.testing.assert_array_equal(
        windows.X[2],
        np.array([[31.0, 310.0], [0.0, 0.0]], dtype=np.float32),
    )
    assert windows.lengths[2] == 1


def test_final_windows_without_target_returns_nan_labels() -> None:
    """Unlabeled final windows are supported for prediction."""
    df = _df().drop(columns=["rul"])

    windows = build_final_windows(
        df,
        FEATURE_COLS,
        window_size=2,
        target_col=None,
    )

    # engine 1, 2, and 3 (padded) all produce one window each
    assert windows.X.shape == (3, 2, 2)
    assert windows.y.dtype == np.float32
    assert np.isnan(windows.y).all()
    assert windows.metadata["engine_id"].tolist() == [1, 2, 3]


def test_short_engines_are_padded_not_skipped() -> None:
    """Engines shorter than the window size are right-zero-padded, not skipped."""
    windows = build_sliding_windows(_df(), FEATURE_COLS, window_size=3)

    # engine 3 has 1 cycle, engines 1 and 2 have 3 cycles each
    # engine 1: 1 full window; engine 2: 1 full window; engine 3: 1 padded window
    assert windows.X.shape == (3, 3, 2)
    assert set(windows.metadata["engine_id"].tolist()) == {1, 2, 3}
    short_rows = windows.metadata.loc[windows.metadata["engine_id"] == 3]
    assert bool(short_rows["padded"].iloc[0]) is True
    short_idx = short_rows.index[0]
    assert windows.lengths[short_idx] == 1


def test_no_eligible_windows_raises() -> None:
    """No eligible windows raises ValueError."""
    empty = pd.DataFrame(
        {"engine_id": [], "cycle": [], "s_1": [], "s_2": [], "rul": []}
    )
    with pytest.raises(ValueError, match="No eligible sequence windows"):
        build_sliding_windows(empty, FEATURE_COLS, window_size=1)


def test_missing_required_feature_or_target_columns_raise_key_error() -> None:
    """Missing required feature or target columns raise KeyError."""
    with pytest.raises(KeyError, match="s_1"):
        build_sliding_windows(_df().drop(columns=["s_1"]), FEATURE_COLS, 2)

    with pytest.raises(KeyError, match="rul"):
        build_sliding_windows(_df().drop(columns=["rul"]), FEATURE_COLS, 2)


def test_missing_engine_id_column_raises_key_error() -> None:
    """Missing engine identifier column raises KeyError."""
    with pytest.raises(KeyError, match="engine_id"):
        build_sliding_windows(_df().drop(columns=["engine_id"]), FEATURE_COLS, 2)


def test_missing_cycle_column_raises_key_error() -> None:
    """Missing cycle column raises KeyError."""
    with pytest.raises(KeyError, match="cycle"):
        build_sliding_windows(_df().drop(columns=["cycle"]), FEATURE_COLS, 2)


def test_window_size_must_be_positive() -> None:
    """Non-positive window sizes raise ValueError."""
    with pytest.raises(ValueError, match="window_size"):
        build_final_windows(_df(), FEATURE_COLS, window_size=0)

    with pytest.raises(ValueError, match="window_size"):
        build_final_windows(_df(), FEATURE_COLS, window_size=-1)


def _toy_frame(engine_id: int, n_cycles: int, n_features: int = 3) -> pd.DataFrame:
    """Build a minimal engine dataframe for padding tests.

    Args:
        engine_id: Identifier for the engine.
        n_cycles: Number of cycles to generate.
        n_features: Number of float feature columns to include.

    Returns:
        DataFrame with ``engine_id``, ``cycle``, ``rul``, and feature columns.
    """
    data: dict[str, list[object]] = {
        "engine_id": [engine_id] * n_cycles,
        "cycle": list(range(1, n_cycles + 1)),
        "rul": list(range(n_cycles, 0, -1)),
    }
    for i in range(n_features):
        data[f"f_{i}"] = list(np.arange(n_cycles, dtype=np.float32) + i)
    return pd.DataFrame(data)


def test_sliding_window_pads_short_engine_right_zero() -> None:
    """Short engines produce one right-zero-padded window in sliding mode."""
    short = _toy_frame(engine_id=1, n_cycles=5)  # shorter than window
    long = _toy_frame(engine_id=2, n_cycles=12)
    frame = pd.concat([short, long], ignore_index=True)

    windows = build_sliding_windows(
        frame, window_size=10, feature_cols=["f_0", "f_1", "f_2"]
    )

    assert windows.lengths.dtype == np.int64
    # short engine: one padded window of actual length 5
    short_rows = windows.metadata.loc[windows.metadata["engine_id"] == 1]
    assert len(short_rows) == 1
    short_idx = short_rows.index[0]
    assert windows.lengths[short_idx] == 5
    assert bool(short_rows["padded"].iloc[0]) is True
    # right-zero-pad: last 5 timesteps must be zero (window_size=10, n_rows=5)
    np.testing.assert_array_equal(
        windows.X[short_idx, 5:, :], np.zeros((5, 3), dtype=np.float32)
    )
    # real data is left-aligned
    assert windows.X[short_idx, 0, 0] == 0.0   # first real value (feature 0)
    assert windows.X[short_idx, 4, 0] == 4.0   # last real value

    # long engine windows are full length, not padded
    long_rows = windows.metadata.loc[windows.metadata["engine_id"] == 2]
    long_idx = long_rows.index
    assert (windows.lengths[long_idx] == 10).all()
    assert (long_rows["padded"] == False).all()  # noqa: E712


def test_final_window_pads_short_engine() -> None:
    """Short engines produce one right-zero-padded window in final mode."""
    short = _toy_frame(engine_id=1, n_cycles=4)
    windows = build_final_windows(
        short, window_size=8, feature_cols=["f_0", "f_1", "f_2"]
    )
    assert len(windows.metadata) == 1
    assert windows.lengths[0] == 4
    assert bool(windows.metadata["padded"].iloc[0]) is True
    # right-zero-pad: last 4 timesteps must be zero (window_size=8, n_rows=4)
    np.testing.assert_array_equal(
        windows.X[0, 4:, :], np.zeros((4, 3), dtype=np.float32)
    )
