"""Tests for turbofan.models.artifacts."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge

from turbofan.models.artifacts import (
    create_run_dir,
    save_json,
    save_model,
    save_predictions,
)


def test_create_run_dir_uses_timestamp_and_run_name(tmp_path: Path) -> None:
    """Run directories are timestamped under the configured artifact dir."""
    timestamp = datetime(2026, 5, 24, 12, 30, 5, tzinfo=UTC)
    path = create_run_dir(tmp_path, "baseline", timestamp=timestamp)
    assert path == tmp_path / "baseline" / "20260524-123005"
    assert path.exists()


def test_save_model_round_trip(tmp_path: Path) -> None:
    """Saved joblib model can be loaded again."""
    model = Ridge().fit([[0.0], [1.0]], [0.0, 1.0])
    path = save_model(model, tmp_path / "model.joblib")
    loaded = joblib.load(path)
    assert loaded.predict([[2.0]])[0] == model.predict([[2.0]])[0]


def test_save_json_writes_valid_json(tmp_path: Path) -> None:
    """JSON payload is written with stringified Path values."""
    path = save_json(
        {"alpha": 1.0, "artifact_dir": Path("artifacts")},
        tmp_path / "x.json",
    )
    loaded = json.loads(path.read_text())
    assert loaded == {"alpha": 1.0, "artifact_dir": "artifacts"}


def test_save_predictions_writes_csv(tmp_path: Path) -> None:
    """Prediction rows are written to CSV."""
    df = pd.DataFrame({"engine_id": [1], "prediction": [10.0]})
    path = save_predictions(df, tmp_path / "predictions.csv")
    loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(loaded, df)
