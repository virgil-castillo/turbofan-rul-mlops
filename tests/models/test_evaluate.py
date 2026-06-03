"""Tests for turbofan.models.evaluate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from turbofan.models.evaluate import (
    add_rul_column,
    align_official_test_labels,
    evaluate_rows,
    select_last_cycle_per_engine,
    split_features_target,
)


class FixedPredictor:
    """Predictor returning preconfigured values."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return predictions matching the requested row count.

        Args:
            X: Feature rows.

        Returns:
            Configured predictions as a float array.
        """
        return np.asarray(self.values[: len(X)], dtype=np.float64)


def test_add_rul_column_uses_capped_rul() -> None:
    """RUL column follows existing capped label semantics."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "s_1": [1.0, 2.0, 3.0],
        }
    )
    result = add_rul_column(df, max_rul=1)
    assert list(result["rul"]) == [1, 1, 0]
    assert "rul" not in df.columns


def test_split_features_target_removes_only_target() -> None:
    """Feature/target split preserves non-target columns."""
    df = pd.DataFrame(
        {"engine_id": [1], "cycle": [1], "s_1": [2.0], "rul": [3.0]}
    )
    X, y = split_features_target(df)
    assert list(X.columns) == ["engine_id", "cycle", "s_1"]
    assert list(y) == [3.0]


def test_split_features_target_missing_target_raises() -> None:
    """Missing target column raises KeyError."""
    df = pd.DataFrame({"engine_id": [1], "cycle": [1], "s_1": [2.0]})
    with pytest.raises(KeyError):
        split_features_target(df)


def test_select_last_cycle_per_engine() -> None:
    """Final row per engine is selected and sorted by engine_id."""
    df = pd.DataFrame(
        {
            "engine_id": [2, 1, 1, 2],
            "cycle": [1, 1, 3, 4],
            "s_1": [20.0, 10.0, 30.0, 40.0],
        }
    )
    result = select_last_cycle_per_engine(df)
    assert list(result["engine_id"]) == [1, 2]
    assert list(result["cycle"]) == [3, 4]


def test_align_official_test_labels() -> None:
    """Official labels align to one final-cycle row per engine."""
    last_rows = pd.DataFrame({"engine_id": [1, 2], "cycle": [3, 4]})
    labels = pd.Series([50, 60], name="rul")
    aligned = align_official_test_labels(last_rows, labels)
    assert list(aligned) == [50.0, 60.0]
    assert aligned.name == "rul"


def test_align_official_test_labels_count_mismatch_raises() -> None:
    """Official label count must match final-cycle row count."""
    last_rows = pd.DataFrame({"engine_id": [1, 2], "cycle": [3, 4]})
    labels = pd.Series([50], name="rul")
    with pytest.raises(ValueError, match="label count"):
        align_official_test_labels(last_rows, labels)


def test_official_labels_align_after_last_cycle_selection() -> None:
    """Labels align after selecting sorted final-cycle test rows."""
    test_rows = pd.DataFrame(
        {
            "engine_id": [2, 1, 1, 2],
            "cycle": [1, 2, 5, 4],
        }
    )
    last_rows = select_last_cycle_per_engine(test_rows)
    aligned = align_official_test_labels(last_rows, pd.Series([100, 200]))
    assert list(last_rows["engine_id"]) == [1, 2]
    assert list(aligned) == [100.0, 200.0]


def test_evaluate_rows_clips_negative_predictions() -> None:
    """Negative predictions are clipped to zero before metrics."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            "s_1": [1.0, 2.0],
            "rul": [0.0, 10.0],
        }
    )
    metrics = evaluate_rows(FixedPredictor([-5.0, 10.0]), df)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0


def test_evaluate_rows_rejects_wrong_prediction_length() -> None:
    """Prediction count must match target row count."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            "s_1": [1.0, 2.0],
            "rul": [0.0, 10.0],
        }
    )
    with pytest.raises(ValueError, match="same length"):
        evaluate_rows(FixedPredictor([1.0]), df)
