"""Thin MLflow wrapper for unified Ridge and GRU run logging.

This module holds no training logic. It points MLflow at a local SQLite store
and exposes small, fully typed helpers for logging params, metrics, tags, and
per-epoch training curves. On-disk artifacts remain the source of truth; MLflow
only records run metadata.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

import mlflow
import pandas as pd

TRAINING_EXPERIMENT: str = "turbofan-training"
SWEEP_EXPERIMENT: str = "turbofan-sweeps"

_DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
_HISTORY_STEP_COLUMN = "epoch"
_REQUIRED_METRIC_SUFFIXES = ("rmse", "mae")


def configure_mlflow(tracking_uri: str | None = None) -> None:
    """Point MLflow at the local SQLite run store.

    Resolution order: the explicit ``tracking_uri`` argument, then the
    ``MLFLOW_TRACKING_URI`` environment variable, then the default local
    ``sqlite:///mlflow.db`` store created in the working directory. Idempotent.

    Args:
        tracking_uri: Optional explicit MLflow tracking URI. When ``None``, the
            environment variable or local SQLite default is used.

    Returns:
        None.
    """
    resolved = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI
    )
    mlflow.set_tracking_uri(resolved)


def log_params(params: Mapping[str, object]) -> None:
    """Stringify and log a flat parameter mapping to the active run.

    Args:
        params: Flat mapping of parameter names to values. Values are converted
            to strings before logging.

    Returns:
        None.
    """
    mlflow.log_params({key: str(value) for key, value in params.items()})


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log final or stepped metrics to the active run.

    Validates that the mapping carries both an RMSE and an MAE metric (any
    ``*rmse``/``*mae`` key, e.g. ``val_rmse``), mirroring the guard previously
    enforced by the removed training-log writer.

    Args:
        metrics: Mapping of metric names to float values.
        step: Optional metric step. When ``None``, metrics are logged without a
            step (final values).

    Returns:
        None.

    Raises:
        ValueError: If no RMSE-type or MAE-type metric is present.
    """
    _validate_required_metrics(metrics)
    values = {key: float(value) for key, value in metrics.items()}
    if step is None:
        mlflow.log_metrics(values)
    else:
        mlflow.log_metrics(values, step=step)


def set_tags(tags: Mapping[str, object]) -> None:
    """Stringify and set run tags on the active run.

    Args:
        tags: Mapping of tag names to values. Values are converted to strings.

    Returns:
        None.
    """
    mlflow.set_tags({key: str(value) for key, value in tags.items()})


def log_history(history: pd.DataFrame) -> None:
    """Replay a per-epoch training history as stepped MLflow metrics.

    Every column except ``epoch`` is logged as a stepped metric keyed by its
    column name, using the ``epoch`` value as the MLflow step. This replays the
    history returned by ``train_gru_model`` post-hoc without modifying the
    training loop.

    Args:
        history: Per-epoch history with an ``epoch`` column plus one column per
            recorded metric (e.g. ``train_loss``, ``validation_windows_rmse``).

    Returns:
        None.
    """
    metric_columns = [
        column for column in history.columns if column != _HISTORY_STEP_COLUMN
    ]
    for record in history.to_dict(orient="records"):
        step = int(record[_HISTORY_STEP_COLUMN])
        mlflow.log_metrics(
            {column: float(record[column]) for column in metric_columns},
            step=step,
        )


def _validate_required_metrics(metrics: Mapping[str, float]) -> None:
    """Ensure both an RMSE-type and an MAE-type metric are present.

    Args:
        metrics: Mapping of metric names to values.

    Raises:
        ValueError: If no key matches an ``rmse`` or ``mae`` metric.
    """
    missing = [
        suffix
        for suffix in _REQUIRED_METRIC_SUFFIXES
        if not any(
            key == suffix or key.endswith(f"_{suffix}") for key in metrics
        )
    ]
    if missing:
        raise ValueError(
            f"Missing required metric(s): {', '.join(missing)}"
        )
