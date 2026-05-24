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
    """Sliding windows sort rows and use final-timestep labels and metadata."""
    windows = build_sliding_windows(_df(), FEATURE_COLS, window_size=2)

    assert isinstance(windows, WindowedSequences)
    assert windows.X.dtype == np.float32
    assert windows.y.dtype == np.float32
    assert windows.X.shape == (4, 2, 2)
    assert windows.y.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert windows.metadata[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 2},
        {"engine_id": 1, "cycle": 3},
        {"engine_id": 2, "cycle": 2},
        {"engine_id": 2, "cycle": 3},
    ]
    np.testing.assert_array_equal(
        windows.X[0],
        np.array([[11.0, 110.0], [12.0, 120.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        windows.X[2],
        np.array([[21.0, 210.0], [22.0, 220.0]], dtype=np.float32),
    )


def test_final_windows_return_one_window_per_eligible_engine() -> None:
    """Final windows use only the last full window per eligible engine."""
    windows = build_final_windows(_df(), FEATURE_COLS, window_size=2)

    assert windows.X.shape == (2, 2, 2)
    assert windows.y.tolist() == [0.0, 0.0]
    assert windows.metadata[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 3},
        {"engine_id": 2, "cycle": 3},
    ]
    np.testing.assert_array_equal(
        windows.X[0],
        np.array([[12.0, 120.0], [13.0, 130.0]], dtype=np.float32),
    )


def test_final_windows_without_target_returns_nan_labels() -> None:
    """Unlabeled final windows are supported for prediction."""
    df = _df().drop(columns=["rul"])

    windows = build_final_windows(
        df,
        FEATURE_COLS,
        window_size=2,
        target_col=None,
    )

    assert windows.X.shape == (2, 2, 2)
    assert windows.y.dtype == np.float32
    assert np.isnan(windows.y).all()
    assert windows.metadata["engine_id"].tolist() == [1, 2]


def test_short_engines_are_skipped() -> None:
    """Engines shorter than the window size are skipped."""
    windows = build_sliding_windows(_df(), FEATURE_COLS, window_size=3)

    assert windows.X.shape == (2, 3, 2)
    assert windows.metadata["engine_id"].tolist() == [1, 2]


def test_no_eligible_windows_raises() -> None:
    """No eligible windows raises ValueError."""
    with pytest.raises(ValueError, match="No eligible sequence windows"):
        build_sliding_windows(_df(), FEATURE_COLS, window_size=10)


def test_missing_required_feature_or_target_columns_raise_key_error() -> None:
    """Missing required feature or target columns raise KeyError."""
    with pytest.raises(KeyError, match="s_1"):
        build_sliding_windows(_df().drop(columns=["s_1"]), FEATURE_COLS, 2)

    with pytest.raises(KeyError, match="rul"):
        build_sliding_windows(_df().drop(columns=["rul"]), FEATURE_COLS, 2)


def test_window_size_must_be_positive() -> None:
    """Non-positive window sizes raise ValueError."""
    with pytest.raises(ValueError, match="window_size"):
        build_final_windows(_df(), FEATURE_COLS, window_size=0)
