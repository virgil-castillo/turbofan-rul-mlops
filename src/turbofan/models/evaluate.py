"""Evaluation helpers for baseline RUL models."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.data import labels
from turbofan.models import metrics


class Predictor(Protocol):
    """Protocol for fitted estimators that can predict from a DataFrame."""

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Predict target values for feature rows.

        Args:
            X: Feature rows.

        Returns:
            Predicted RUL values.
        """
        ...


def add_rul_column(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """Return a copy of training data with a computed ``rul`` column.

    Args:
        df: Training DataFrame with ``engine_id`` and ``cycle``.
        max_rul: Maximum RUL cap.

    Returns:
        Copy of ``df`` with a ``rul`` column.
    """
    result = df.copy()
    result["rul"] = labels.compute_rul_labels(result, max_rul=max_rul)
    return result


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "rul",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from target labels.

    Args:
        df: Labeled DataFrame.
        target_col: Target column name.

    Returns:
        Tuple of feature DataFrame and target Series.

    Raises:
        KeyError: If ``target_col`` is missing.
    """
    return df.drop(columns=[target_col]), df[target_col].astype(float)


def evaluate_rows(
    estimator: Predictor,
    df: pd.DataFrame,
    target_col: str = "rul",
) -> dict[str, float]:
    """Evaluate predictions for labeled rows.

    Args:
        estimator: Fitted estimator with a ``predict`` method.
        df: Labeled rows to evaluate.
        target_col: Target column name.

    Returns:
        Regression metrics with non-negative clipped predictions.

    Raises:
        KeyError: If ``target_col`` is missing.
        ValueError: If metric inputs are invalid.
    """
    X, y = split_features_target(df, target_col=target_col)
    preds = np.clip(np.asarray(estimator.predict(X), dtype=np.float64), 0.0, None)
    return metrics.regression_metrics(y, preds)


def select_last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Select the final available row for each engine.

    Args:
        df: Test DataFrame with ``engine_id`` and ``cycle`` columns.

    Returns:
        One row per engine, sorted by ``engine_id`` and reset index.
    """
    sorted_df = df.sort_values(["engine_id", "cycle"])
    idx = sorted_df.groupby("engine_id")["cycle"].idxmax()
    return sorted_df.loc[idx].sort_values("engine_id").reset_index(drop=True)


def align_official_test_labels(
    last_cycle_df: pd.DataFrame,
    rul_labels: pd.Series,
) -> pd.Series:
    """Align official RUL labels to final-cycle test rows.

    Args:
        last_cycle_df: One final-cycle row per test engine.
        rul_labels: Official RUL labels in engine order.

    Returns:
        Float RUL Series aligned to ``last_cycle_df``.

    Raises:
        ValueError: If the number of labels does not match the number of rows.
    """
    if len(last_cycle_df) != len(rul_labels):
        raise ValueError("Official RUL label count must match test engine count.")
    return pd.Series(
        rul_labels.to_numpy(dtype=np.float64),
        index=last_cycle_df.index,
        name="rul",
    )
