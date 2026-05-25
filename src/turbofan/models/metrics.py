"""Regression metrics for RUL prediction."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

type MetricInput = pd.Series | npt.ArrayLike | Sequence[float]


def _as_arrays(
    y_true: MetricInput,
    y_pred: MetricInput,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate and convert metric inputs to float arrays.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Tuple of one-dimensional float arrays.

    Raises:
        ValueError: If lengths differ or inputs contain NaN values.
    """
    true_arr = _to_float_array(y_true)
    pred_arr = _to_float_array(y_pred)
    if true_arr.ndim != 1 or pred_arr.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional.")
    if true_arr.shape[0] == 0:
        raise ValueError("Metric inputs must not be empty.")
    if true_arr.shape != pred_arr.shape:
        raise ValueError("Metric inputs must have the same length.")
    if not np.isfinite(true_arr).all() or not np.isfinite(pred_arr).all():
        raise ValueError("Metric inputs must contain only finite values.")
    return true_arr, pred_arr


def _to_float_array(values: MetricInput) -> npt.NDArray[np.float64]:
    """Convert metric inputs to float arrays.

    Args:
        values: Values to convert.

    Returns:
        One-dimensional or multi-dimensional float array.

    Raises:
        ValueError: If values cannot be converted to float.
    """
    try:
        if isinstance(values, pd.Series):
            return values.to_numpy(dtype=np.float64, na_value=np.nan)
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Metric inputs must be numeric.") from exc


def rmse(y_true: MetricInput, y_pred: MetricInput) -> float:
    """Compute root mean squared error.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Root mean squared error.

    Raises:
        ValueError: If inputs differ in shape, are empty, are not
            one-dimensional, contain non-finite values, or cannot be converted to
            numeric arrays.
    """
    true_arr, pred_arr = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))


def mae(y_true: MetricInput, y_pred: MetricInput) -> float:
    """Compute mean absolute error.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Mean absolute error.

    Raises:
        ValueError: If inputs differ in shape, are empty, are not
            one-dimensional, contain non-finite values, or cannot be converted to
            numeric arrays.
    """
    true_arr, pred_arr = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(pred_arr - true_arr)))


def phm08_score(y_true: MetricInput, y_pred: MetricInput) -> float:
    """Compute the asymmetric PHM08 RUL score.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Sum of PHM08 asymmetric penalties.

    Raises:
        ValueError: If inputs differ in shape, are empty, are not
            one-dimensional, contain non-finite values, or cannot be converted to
            numeric arrays.
    """
    true_arr, pred_arr = _as_arrays(y_true, y_pred)
    diff = pred_arr - true_arr
    max_exponent = 700.0
    penalties = np.where(
        diff < 0.0,
        np.exp(np.clip(-diff / 13.0, a_min=None, a_max=max_exponent)) - 1.0,
        np.exp(np.clip(diff / 10.0, a_min=None, a_max=max_exponent)) - 1.0,
    )
    return float(np.sum(penalties))


def regression_metrics(
    y_true: MetricInput,
    y_pred: MetricInput,
) -> dict[str, float]:
    """Compute all baseline regression metrics.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Mapping with ``rmse``, ``mae``, and ``phm08_score``.

    Raises:
        ValueError: If inputs differ in shape, are empty, are not
            one-dimensional, contain non-finite values, or cannot be converted to
            numeric arrays.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "phm08_score": phm08_score(y_true, y_pred),
    }
