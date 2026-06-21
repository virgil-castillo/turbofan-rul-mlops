"""Tests for turbofan.registry.tracking."""
from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from turbofan.registry import tracking


def _sqlite_uri(tmp_path: Path, name: str = "mlflow.db") -> str:
    """Build a SQLite tracking URI under ``tmp_path``.

    Args:
        tmp_path: Pytest temporary directory.
        name: Database file name.

    Returns:
        SQLite tracking URI string.
    """
    return f"sqlite:///{(tmp_path / name).as_posix()}"


def test_configure_mlflow_sets_explicit_tracking_uri(tmp_path: Path) -> None:
    """An explicit tracking URI is applied to MLflow."""
    uri = _sqlite_uri(tmp_path)
    tracking.configure_mlflow(uri)
    assert mlflow.get_tracking_uri() == uri


def test_configure_mlflow_honors_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A None URI falls back to the MLFLOW_TRACKING_URI env var."""
    uri = _sqlite_uri(tmp_path, "from_env.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    tracking.configure_mlflow()
    assert mlflow.get_tracking_uri() == uri


def test_configure_mlflow_defaults_to_local_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no URI and no env var, the local mlflow.db default is used."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tracking.configure_mlflow()
    assert mlflow.get_tracking_uri() == "sqlite:///mlflow.db"


def test_experiment_name_constants() -> None:
    """Experiment name constants match the agreed conventions."""
    assert tracking.TRAINING_EXPERIMENT == "turbofan-training"
    assert tracking.SWEEP_EXPERIMENT == "turbofan-sweeps"


def test_log_params_stringifies_values(tmp_path: Path) -> None:
    """Params of mixed types are stringified before logging."""
    tracking.configure_mlflow(_sqlite_uri(tmp_path))
    mlflow.set_experiment(tracking.TRAINING_EXPERIMENT)
    with mlflow.start_run():
        tracking.log_params(
            {"alpha": 0.5, "feature_families": ["raw"], "windows": (5, 10)}
        )
    runs = mlflow.search_runs(experiment_names=[tracking.TRAINING_EXPERIMENT])
    assert runs.loc[0, "params.alpha"] == "0.5"
    assert runs.loc[0, "params.feature_families"] == "['raw']"
    assert runs.loc[0, "params.windows"] == "(5, 10)"


def test_log_metrics_records_values(tmp_path: Path) -> None:
    """Final metrics are recorded as floats on the active run."""
    tracking.configure_mlflow(_sqlite_uri(tmp_path))
    mlflow.set_experiment(tracking.TRAINING_EXPERIMENT)
    with mlflow.start_run():
        tracking.log_metrics({"val_rmse": 1.5, "val_mae": 1.0})
    runs = mlflow.search_runs(experiment_names=[tracking.TRAINING_EXPERIMENT])
    assert runs.loc[0, "metrics.val_rmse"] == 1.5
    assert runs.loc[0, "metrics.val_mae"] == 1.0


def test_log_metrics_records_official_metrics(tmp_path: Path) -> None:
    """Official-test metrics are logged alongside validation metrics."""
    tracking.configure_mlflow(_sqlite_uri(tmp_path))
    mlflow.set_experiment(tracking.TRAINING_EXPERIMENT)
    with mlflow.start_run():
        tracking.log_metrics(
            {
                "val_rmse": 1.5,
                "val_mae": 1.0,
                "official_rmse": 2.0,
                "official_mae": 1.7,
                "official_phm08": 3.0,
            }
        )
    runs = mlflow.search_runs(experiment_names=[tracking.TRAINING_EXPERIMENT])
    assert runs.loc[0, "metrics.official_phm08"] == 3.0


@pytest.mark.parametrize("missing", ["val_rmse", "val_mae"])
def test_log_metrics_validates_required_metrics(
    tmp_path: Path,
    missing: str,
) -> None:
    """Metrics without an rmse or mae entry raise ValueError."""
    tracking.configure_mlflow(_sqlite_uri(tmp_path))
    metrics = {"val_rmse": 1.5, "val_mae": 1.0}
    del metrics[missing]
    expected = missing.rsplit("_", 1)[-1]
    mlflow.set_experiment(tracking.TRAINING_EXPERIMENT)
    with mlflow.start_run(), pytest.raises(ValueError, match=expected):
        tracking.log_metrics(metrics)


def test_set_tags_stringifies_values(tmp_path: Path) -> None:
    """Tags of mixed types are stringified before being set."""
    tracking.configure_mlflow(_sqlite_uri(tmp_path))
    mlflow.set_experiment(tracking.TRAINING_EXPERIMENT)
    with mlflow.start_run():
        tracking.set_tags(
            {
                "model_type": "ridge",
                "run_type": "production",
                "best_epoch": 7,
            }
        )
    runs = mlflow.search_runs(experiment_names=[tracking.TRAINING_EXPERIMENT])
    assert runs.loc[0, "tags.model_type"] == "ridge"
    assert runs.loc[0, "tags.run_type"] == "production"
    assert runs.loc[0, "tags.best_epoch"] == "7"


def test_log_history_replays_per_epoch_metrics(tmp_path: Path) -> None:
    """Every non-epoch history column is replayed as a stepped metric."""
    tracking.configure_mlflow(_sqlite_uri(tmp_path))
    mlflow.set_experiment(tracking.TRAINING_EXPERIMENT)
    history = pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": 5.0,
                "validation_windows_rmse": 4.0,
                "validation_windows_mae": 3.0,
            },
            {
                "epoch": 2,
                "train_loss": 3.0,
                "validation_windows_rmse": 2.0,
                "validation_windows_mae": 1.0,
            },
        ]
    )
    with mlflow.start_run() as run:
        tracking.log_history(history)
        run_id = run.info.run_id

    client = MlflowClient()
    train_hist = client.get_metric_history(run_id, "train_loss")
    assert [(m.step, m.value) for m in train_hist] == [(1, 5.0), (2, 3.0)]
    rmse_hist = client.get_metric_history(run_id, "validation_windows_rmse")
    assert [(m.step, m.value) for m in rmse_hist] == [(1, 4.0), (2, 2.0)]
