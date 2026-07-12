"""Tests for turbofan.evaluation.metrics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbofan.evaluation.metrics import (
    mae,
    official_test_metrics,
    phm08_score,
    regression_metrics,
    rmse,
)


def test_rmse_matches_hand_computation() -> None:
    """RMSE equals sqrt(mean squared error)."""
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([12.0, 18.0, 33.0])
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(17.0 / 3.0))


def test_mae_matches_hand_computation() -> None:
    """MAE equals mean absolute error."""
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([12.0, 18.0, 33.0])
    assert mae(y_true, y_pred) == pytest.approx(7.0 / 3.0)


def test_phm08_score_matches_early_and_late_errors() -> None:
    """PHM08 uses asymmetric penalties for early and late predictions."""
    y_true = pd.Series([100.0, 100.0])
    y_pred = pd.Series([90.0, 110.0])
    expected = (math.exp(10.0 / 13.0) - 1.0) + (math.exp(10.0 / 10.0) - 1.0)
    assert phm08_score(y_true, y_pred) == pytest.approx(expected)


def test_phm08_score_does_not_overflow_for_large_residuals() -> None:
    """PHM08 clips exponent inputs while preserving finite penalties."""
    score = phm08_score([0.0, 10_000.0], [10_000.0, 0.0])

    assert math.isfinite(score)
    assert score > 0.0


def test_regression_metrics_excludes_phm08() -> None:
    """Validation metrics contain only RMSE and MAE, never the PHM08 score."""
    metrics = regression_metrics(
        pd.Series([10.0, 20.0]),
        pd.Series([13.0, 18.0]),
    )
    assert set(metrics) == {"rmse", "mae"}
    assert metrics["rmse"] == pytest.approx(math.sqrt(13.0 / 2.0))
    assert metrics["mae"] == pytest.approx(2.5)


def test_official_test_metrics_includes_phm08() -> None:
    """Official-test metrics extend RMSE/MAE with the asymmetric PHM08 score."""
    metrics = official_test_metrics(
        pd.Series([10.0, 20.0]),
        pd.Series([13.0, 18.0]),
    )
    assert set(metrics) == {"rmse", "mae", "phm08_score"}
    assert metrics["rmse"] == pytest.approx(math.sqrt(13.0 / 2.0))
    assert metrics["mae"] == pytest.approx(2.5)
    expected_score = (math.exp(3.0 / 10.0) - 1.0) + (math.exp(2.0 / 13.0) - 1.0)
    assert metrics["phm08_score"] == pytest.approx(expected_score)


def test_metrics_reject_unequal_lengths() -> None:
    """Mismatched input lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        rmse(pd.Series([1.0]), pd.Series([1.0, 2.0]))


def test_metrics_reject_nan_inputs() -> None:
    """NaN values raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        mae(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0]))


def test_metrics_reject_nan_predictions() -> None:
    """NaN predictions raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        rmse(pd.Series([1.0, 2.0]), pd.Series([1.0, np.nan]))


def test_metrics_reject_pandas_missing_values() -> None:
    """Pandas missing values raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        mae(pd.Series([1.0, pd.NA]), pd.Series([1.0, 2.0]))


def test_metrics_reject_infinite_inputs() -> None:
    """Infinite values raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        rmse([1.0, float("inf")], [1.0, 2.0])


def test_metrics_reject_infinite_predictions() -> None:
    """Infinite predictions raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        mae([1.0, 2.0], [1.0, float("-inf")])


def test_metrics_reject_non_numeric_inputs() -> None:
    """Non-numeric inputs raise ValueError."""
    with pytest.raises(ValueError, match="numeric"):
        phm08_score(["bad"], [1.0])


def test_metrics_reject_empty_inputs() -> None:
    """Empty inputs raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        phm08_score([], [])


def test_metrics_reject_non_1d_inputs() -> None:
    """Multi-dimensional inputs raise ValueError."""
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[1.0, 3.0]])
    with pytest.raises(ValueError, match="one-dimensional"):
        rmse(y_true, y_pred)


def test_metrics_accept_plain_sequences() -> None:
    """Plain Python sequences are accepted."""
    assert mae([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


def test_metrics_accept_numpy_arrays() -> None:
    """NumPy arrays are accepted."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 3, 5])
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(5.0 / 3.0))
