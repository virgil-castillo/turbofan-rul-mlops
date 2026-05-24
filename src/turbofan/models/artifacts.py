"""Local artifact persistence for model training runs."""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator


def create_run_dir(
    artifact_dir: Path,
    run_name: str,
    timestamp: datetime | None = None,
) -> Path:
    """Create and return a timestamped local run directory.

    Args:
        artifact_dir: Root artifact directory.
        run_name: Human-readable run group name.
        timestamp: Optional timestamp for deterministic tests.

    Returns:
        Created run directory path.
    """
    ts = timestamp if timestamp is not None else datetime.now(tz=UTC)
    run_dir = artifact_dir / run_name / ts.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_model(estimator: BaseEstimator, path: Path) -> Path:
    """Persist a fitted sklearn estimator with joblib.

    Args:
        estimator: Fitted sklearn estimator.
        path: Destination file path.

    Returns:
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
    return path


def save_json(payload: Mapping[str, object], path: Path) -> Path:
    """Write a JSON artifact.

    Args:
        payload: JSON-serializable mapping. Path values are stringified.
        path: Destination file path.

    Returns:
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def save_predictions(df: pd.DataFrame, path: Path) -> Path:
    """Write prediction rows to CSV.

    Args:
        df: Prediction DataFrame.
        path: Destination file path.

    Returns:
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
