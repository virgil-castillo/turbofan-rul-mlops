"""Tests for turbofan.models.training_log."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from turbofan.models.training_log import append_training_log, build_log_entry


def _metrics() -> dict[str, float]:
    """Return a complete metrics mapping for training log tests.

    Returns:
        Metrics containing all required training log keys.
    """
    return {"rmse": 1.5, "mae": 1.0, "phm08_score": 2.5}


def test_build_log_entry_required_fields() -> None:
    """Log entries contain required fields with timestamp and defaults."""
    entry = build_log_entry(
        model_type="baseline",
        dataset="FD001",
        random_seed=42,
        hyperparameters={"alpha": 0.1},
        metrics=_metrics(),
        training_duration_seconds=12.5,
        device="cpu",
    )

    assert set(entry) == {
        "timestamp",
        "model_type",
        "dataset",
        "random_seed",
        "run_dir",
        "hyperparameters",
        "metrics",
        "best_epoch",
        "training_duration_seconds",
        "device",
        "extra",
    }
    timestamp = entry["timestamp"]
    assert isinstance(timestamp, str)
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0
    assert entry["run_dir"] is None
    assert entry["best_epoch"] is None
    assert entry["extra"] == {}


@pytest.mark.parametrize("missing_metric", ["rmse", "mae", "phm08_score"])
def test_build_log_entry_validates_metrics(missing_metric: str) -> None:
    """Missing required metrics raise ValueError."""
    metrics = _metrics()
    del metrics[missing_metric]

    with pytest.raises(ValueError, match=missing_metric):
        build_log_entry(
            model_type="baseline",
            dataset="FD001",
            random_seed=42,
            hyperparameters={},
            metrics=metrics,
            training_duration_seconds=12.5,
            device="cpu",
        )


def test_append_training_log_creates_file(tmp_path: Path) -> None:
    """Appending a log entry creates missing parents and file."""
    log_path = tmp_path / "nested" / "training_log.jsonl"
    entry = build_log_entry(
        model_type="baseline",
        dataset="FD001",
        random_seed=42,
        hyperparameters={},
        metrics=_metrics(),
        training_duration_seconds=12.5,
        device="cpu",
    )

    append_training_log(entry, log_path)

    assert log_path.exists()
    assert json.loads(log_path.read_text()) == entry


def test_append_training_log_appends_multiple(tmp_path: Path) -> None:
    """Multiple appended entries produce one JSON object per line."""
    log_path = tmp_path / "training_log.jsonl"
    first = build_log_entry(
        model_type="baseline",
        dataset="FD001",
        random_seed=42,
        hyperparameters={},
        metrics=_metrics(),
        training_duration_seconds=12.5,
        device="cpu",
    )
    second = build_log_entry(
        model_type="gru",
        dataset="FD002",
        random_seed=7,
        hyperparameters={"hidden_size": 16},
        metrics={"rmse": 2.0, "mae": 1.5, "phm08_score": 3.0},
        training_duration_seconds=30.0,
        device="cuda",
        best_epoch=4,
    )

    append_training_log(first, log_path)
    append_training_log(second, log_path)

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == [first, second]


def test_append_training_log_serializes_paths(tmp_path: Path) -> None:
    """Path values in entries are serialized as strings."""
    log_path = tmp_path / "training_log.jsonl"
    run_dir = str(tmp_path / "runs" / "baseline")
    artifact_path = tmp_path / "artifacts" / "model.joblib"
    entry = build_log_entry(
        model_type="baseline",
        dataset="FD001",
        random_seed=42,
        hyperparameters={},
        metrics=_metrics(),
        training_duration_seconds=12.5,
        device="cpu",
        run_dir=run_dir,
        extra={"model_path": artifact_path},
    )

    append_training_log(entry, log_path)

    loaded = json.loads(log_path.read_text())
    assert loaded["run_dir"] == run_dir
    assert loaded["extra"]["model_path"] == str(artifact_path)
