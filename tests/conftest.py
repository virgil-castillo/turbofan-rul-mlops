"""Shared pytest fixtures for the turbofan test suite."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from turbofan.config.schema import DataConfig


@pytest.fixture(autouse=True)
def _redirect_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Redirect MLflow to a per-test SQLite store so runs never touch the repo.

    Sets ``MLFLOW_TRACKING_URI`` so both in-process code and subprocess CLIs
    (which copy ``os.environ``) log to an isolated ``tmp_path`` database, and
    pins MLflow's process-global tracking URI to the same store. Pinning is
    required because any prior test calling ``mlflow.set_tracking_uri`` makes
    the global URI override ``MLFLOW_TRACKING_URI`` for ``MlflowClient`` and
    ``mlflow.start_run`` for the rest of the session; without this, registry
    reads and writes would leak across tests. The previous global URI is
    restored afterward.
    """
    import mlflow.tracking.fluent as fluent

    db_path = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    # MLflow writes MLFLOW_EXPERIMENT_ID into os.environ during a run so child
    # processes inherit it; that id refers to a previous test's store and does
    # not exist in this test's fresh store, so clear it (and the name var).
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    # Reset the cached active-experiment id so runs resolve the default
    # experiment in this test's fresh store rather than reusing an id created
    # in a previous test's store (which would not exist here).
    monkeypatch.setattr(fluent, "_active_experiment_id", None, raising=False)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous_uri)


def _write_cmapss_file(path: Path, n_engines: int, n_cycles: int) -> None:
    """Write a small C-MAPSS-style whitespace-delimited file.

    Args:
        path: Destination file path.
        n_engines: Number of synthetic engines.
        n_cycles: Number of cycles per engine.
    """
    lines = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            op_cols = [0.0, 0.0, 0.0]
            sensors = [
                float(cycle + sensor_idx + engine_id)
                for sensor_idx in range(1, 22)
            ]
            values = [engine_id, cycle, *op_cols, *sensors]
            lines.append(" ".join(str(v) for v in values))
    path.write_text("\n".join(lines))

SENSOR_COLS = {f"s_{i}": float(i) for i in range(1, 22)}


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Minimal valid train DataFrame: 3 engines with varied cycle lengths.

    Engine 1: 10 cycles, Engine 2: 7 cycles, Engine 3: 15 cycles.
    """
    rows = []
    for engine_id, n_cycles in [(1, 10), (2, 7), (3, 15)]:
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    **SENSOR_COLS,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_data_dir(tmp_path: Path, sample_train_df: pd.DataFrame) -> Path:
    """Temp directory with correctly-named stub C-MAPSS .txt files for FD001."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    train_path = raw_dir / "train_FD001.txt"
    sample_train_df.to_csv(train_path, sep=" ", header=False, index=False)

    test_path = raw_dir / "test_FD001.txt"
    sample_train_df.to_csv(test_path, sep=" ", header=False, index=False)

    rul_path = raw_dir / "RUL_FD001.txt"
    rul_path.write_text("112\n98\n42\n")

    return raw_dir


@pytest.fixture
def data_cfg(tmp_data_dir: Path) -> DataConfig:
    """DataConfig pointing at the tmp_data_dir stub files."""
    return DataConfig(
        raw_dir=tmp_data_dir,
        processed_dir=tmp_data_dir / "processed",
        interim_dir=tmp_data_dir / "interim",
        fd_subset="FD001",
    )


@pytest.fixture
def tiny_config_path(tmp_path: Path) -> Path:
    """Path to a minimal YAML config suitable for fast GRU integration tests.

    Writes a 5-engine, 25-cycle C-MAPSS stub and a config with tiny GRU
    hyperparameters so tests complete quickly on CPU.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the written config YAML.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapss_file(raw_dir / "train_FD001.txt", n_engines=5, n_cycles=25)
    _write_cmapss_file(raw_dir / "test_FD001.txt", n_engines=2, n_cycles=10)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "project_name: tiny_test",
                "data:",
                f"  raw_dir: {raw_dir.as_posix()}",
                f"  processed_dir: {(tmp_path / 'processed').as_posix()}",
                f"  interim_dir: {(tmp_path / 'interim').as_posix()}",
                "  fd_subset: FD001",
                "  max_rul: 30",
                "  test_size: 0.4",
                "  random_seed: 42",
                "sequence:",
                "  architecture: gru",
                "  window_size: 10",
                "  batch_size: 8",
                "  hidden_size: 4",
                "  num_layers: 1",
                "  dropout: 0.0",
                "  learning_rate: 0.001",
                "  epochs: 2",
                "  patience: 2",
                "  device: cpu",
            ]
        )
    )
    return cfg_path
