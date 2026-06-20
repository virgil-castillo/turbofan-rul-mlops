"""Evaluation helpers for baseline RUL models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from turbofan.config.schema import DataConfig
from turbofan.data import labels
from turbofan.data import loader as data_loader
from turbofan.models import metrics
from turbofan.utils import logging as turbofan_logging

logger = turbofan_logging.get_logger(__name__)


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


def clip_rul_predictions(
    values: npt.ArrayLike,
    max_rul: int,
) -> npt.NDArray[np.float64]:
    """Clip raw predictions into ``[0, max_rul]`` as float64.

    Args:
        values: Raw model predictions.
        max_rul: Maximum allowed RUL value.

    Returns:
        Float64 predictions clipped to ``[0, max_rul]``.
    """
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, float(max_rul))


def predict_with_clipping(
    estimator: Pipeline,
    rows: pd.DataFrame,
    *,
    max_rul: int,
    label: str,
) -> npt.NDArray[np.float64]:
    """Predict rows, log the raw prediction range, and clip to valid RUL bounds.

    Args:
        estimator: Fitted sklearn estimator.
        rows: Feature rows to predict.
        max_rul: Maximum allowed RUL value.
        label: Human-readable prediction-set label for the debug log line.

    Returns:
        Float64 predictions clipped to ``[0, max_rul]``.
    """
    raw = np.asarray(estimator.predict(rows), dtype=np.float64)
    logger.debug(
        "%s raw prediction min/max: %.6f/%.6f", label, raw.min(), raw.max()
    )
    return clip_rul_predictions(raw, max_rul=max_rul)


@dataclass(frozen=True)
class OfficialRidgePredictions:
    """Aligned official-test predictions for a Ridge model.

    Args:
        last_rows: One final-cycle row per test engine.
        y_true: Official RUL labels aligned to ``last_rows``.
        y_pred: Final-cycle predicted RUL values, clipped to ``[0, max_rul]``.
    """

    last_rows: pd.DataFrame
    y_true: pd.Series
    y_pred: npt.NDArray[np.float64]


def predict_ridge_official(
    data_cfg: DataConfig,
    *,
    estimator: Pipeline,
    max_rul: int,
) -> OfficialRidgePredictions:
    """Evaluate a fitted Ridge estimator on the official C-MAPSS test set.

    Predicts over each engine's full trajectory (so rolling/lag features keep
    their context), clips to ``[0, max_rul]``, then selects the final cycle per
    engine to compare against the official labels.

    Args:
        data_cfg: Data layer config locating the official test files.
        estimator: Fitted Ridge pipeline.
        max_rul: Maximum-RUL ceiling for clipping predictions.

    Returns:
        The final-cycle rows with aligned labels and clipped predictions.

    Raises:
        FileNotFoundError: If the official test or RUL files are missing.
    """
    test_raw = data_loader.load_raw_test(data_cfg)
    rul_labels = data_loader.load_rul_labels(data_cfg)
    last_rows = select_last_cycle_per_engine(test_raw)
    y_true = align_official_test_labels(last_rows, rul_labels)
    all_pred = predict_with_clipping(
        estimator, test_raw, max_rul=max_rul, label="official_test"
    )
    pred_rows = test_raw[["engine_id", "cycle"]].copy()
    pred_rows["prediction"] = all_pred
    last_pred = select_last_cycle_per_engine(pred_rows)
    y_pred = last_pred["prediction"].to_numpy(dtype=np.float64)
    return OfficialRidgePredictions(
        last_rows=last_rows, y_true=y_true, y_pred=y_pred
    )
