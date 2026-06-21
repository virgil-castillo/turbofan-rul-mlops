"""Smoke tests for turbofan.cli.train_baseline."""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from turbofan.cli.train_baseline import main as baseline_main
from turbofan.config.schema import DataConfig, ModelConfig, ProjectConfig
from turbofan.data import loader
from turbofan.utils.logging import setup_logging


class _CliResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


def _load_train_baseline_module() -> ModuleType:
    """Load the train_baseline command module for helper testing."""
    from turbofan.cli import train_baseline

    return train_baseline


class RecordingEstimator:
    """Estimator that records prediction input length."""

    def __init__(self) -> None:
        self.seen_rows = 0

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return row-position predictions for inspection.

        Args:
            X: Feature rows.

        Returns:
            Increasing float predictions.
        """
        self.seen_rows = len(X)
        return np.arange(len(X), dtype=np.float64)


def _write_cmapps_file(path: Path, n_engines: int, n_cycles: int) -> None:
    """Write a small C-MAPSS-style whitespace-delimited file."""
    lines = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            op_cols = [0.0, 0.0, 0.0]
            sensors = [float(cycle + i + engine_id) for i in range(1, 22)]
            values = [engine_id, cycle, *op_cols, *sensors]
            lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines))


def test_train_baseline_cli_writes_records_logs_run_and_registers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One CLI run: disk run records, MLflow run + registration, and run.log.

    Consolidates the run-dir artifact, MLflow params/metrics/tags +
    registration, and run.log-artifact facets into a single training run
    instead of spawning a separate ~10s training subprocess per facet, all of
    which exercised the same standard run.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from turbofan import registry

    cfg_path = _write_minimal_baseline_config(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    result = _run_baseline_cli(cfg_path, monkeypatch, capsys)

    # --- run-dir records (model bytes + manifest retired; MLflow is the store) ---
    assert "validation rmse" in result.stdout
    run_dirs = list((artifact_dir / "baseline").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert not (run_dir / "model.joblib").exists()
    assert not (run_dir / "model_manifest.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "validation_predictions.csv").exists()
    assert (run_dir / "official_test_predictions.csv").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"validation", "official_test"}
    assert set(metrics["validation"]) == {"rmse", "mae"}
    assert set(metrics["official_test"]) == {"rmse", "mae", "phm08_score"}

    # --- the production MLflow run: params, metrics, tags ---
    registry.tracking.configure_mlflow()
    runs = mlflow.search_runs(
        experiment_names=[registry.tracking.TRAINING_EXPERIMENT]
    )
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["tags.model_type"] == "ridge"
    assert row["tags.run_type"] == "production"
    assert "tags.run_dir" in row
    assert row["params.alpha"] == "1.0"
    assert row["params.feature_families"] == "['raw']"
    assert row["metrics.val_rmse"] >= 0.0
    assert row["metrics.val_mae"] >= 0.0
    assert row["metrics.training_duration_seconds"] >= 0.0
    assert row["metrics.official_rmse"] >= 0.0

    # --- a registered model version linked to the run + prediction artifacts ---
    run_id = row["run_id"]
    client = MlflowClient()
    name = registry.model_name("ridge", "FD001")
    assert name == "turbofan-ridge-fd001"
    versions = client.search_model_versions(f"name = '{name}'")
    assert len(versions) >= 1
    assert any(version.run_id == run_id for version in versions)

    artifact_paths = {
        artifact.path for artifact in client.list_artifacts(run_id, "predictions")
    }
    assert "predictions/validation_predictions.csv" in artifact_paths

    # --- the run.log diagnostic artifact with training narration ---
    log_paths = {artifact.path for artifact in client.list_artifacts(run_id, "logs")}
    assert "logs/run.log" in log_paths
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="logs/run.log",
    )
    contents = Path(local_path).read_text()
    assert "loading training data for FD001" in contents
    assert "saved baseline run to" in contents


def test_train_baseline_cli_skips_missing_official_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI trains validation model when official test files are absent."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=8)

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "project_name: test",
                "data:",
                f"  raw_dir: {raw_dir.as_posix()}",
                f"  processed_dir: {(tmp_path / 'processed').as_posix()}",
                f"  interim_dir: {(tmp_path / 'interim').as_posix()}",
                "  fd_subset: FD001",
                "  max_rul: 30",
                "  test_size: 0.25",
                "  random_seed: 42",
                "model:",
                "  name: ridge",
                "  alpha: 1.0",
                f"  artifact_dir: {artifact_dir.as_posix()}",
            ]
        )
    )

    result = _run_baseline_cli(cfg_path, monkeypatch, capsys)

    assert "official test evaluation skipped" in result.stderr
    run_dir = next((artifact_dir / "baseline").iterdir())
    assert not (run_dir / "official_test_predictions.csv").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"validation"}


def _write_minimal_baseline_config(tmp_path: Path) -> Path:
    """Write synthetic data and a minimal baseline config; return the config path.

    Args:
        tmp_path: Temporary test directory.

    Returns:
        Path to the written YAML config.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=8)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=2, n_cycles=5)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n")

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "project_name: test",
                "data:",
                f"  raw_dir: {raw_dir.as_posix()}",
                f"  processed_dir: {(tmp_path / 'processed').as_posix()}",
                f"  interim_dir: {(tmp_path / 'interim').as_posix()}",
                "  fd_subset: FD001",
                "  max_rul: 30",
                "  test_size: 0.25",
                "  random_seed: 42",
                "model:",
                "  name: ridge",
                "  alpha: 1.0",
                f"  artifact_dir: {artifact_dir.as_posix()}",
                "features:",
                "  feature_families: [raw]",
            ]
        )
    )
    return cfg_path


def _run_baseline_cli(
    cfg_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *extra_args: str,
) -> _CliResult:
    """Run the baseline training CLI in-process and return captured output.

    Args:
        cfg_path: Path to the YAML config.
        monkeypatch: pytest monkeypatch fixture for sys.argv injection.
        capsys: pytest capsys fixture for stdout/stderr capture.
        *extra_args: Additional CLI arguments.

    Returns:
        CLI result with returncode, stdout, and stderr.
    """
    returncode = baseline_main(["--config", str(cfg_path), *extra_args])
    captured = capsys.readouterr()
    return _CliResult(returncode=returncode, stdout=captured.out, stderr=captured.err)


def test_predict_with_clipping_debug_line_respects_log_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The raw prediction min/max line is DEBUG-only and goes to stderr.

    Exercises the verbosity wiring in-process by driving the helper that emits
    the line and toggling the configured level, instead of running the full
    training CLI in a subprocess twice.
    """
    module = _load_train_baseline_module()
    estimator = RecordingEstimator()
    rows = pd.DataFrame({"s_1": [1.0, 2.0, 3.0]})

    setup_logging("DEBUG")
    module._predict_with_clipping(estimator, rows, rul_cap=125, label="validation")
    assert "raw prediction min/max" in capsys.readouterr().err

    setup_logging("WARNING")
    module._predict_with_clipping(estimator, rows, rul_cap=125, label="validation")
    assert "raw prediction min/max" not in capsys.readouterr().err


def test_official_eval_predicts_full_trajectory_before_final_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Official evaluation preserves full trajectory context for rolling features."""
    module = _load_train_baseline_module()
    test_raw = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "op_1": [0.0] * 5,
            "op_2": [0.0] * 5,
            "op_3": [0.0] * 5,
            "s_1": [1.0, 2.0, 3.0, 10.0, 20.0],
        }
    )
    monkeypatch.setattr(loader, "load_raw_test", lambda cfg: test_raw)
    monkeypatch.setattr(
        loader,
        "load_rul_labels",
        lambda cfg: pd.Series([10.0, 20.0], name="rul"),
    )
    cfg = ProjectConfig(
        project_name="test",
        data=DataConfig(
            raw_dir=tmp_path,
            processed_dir=tmp_path,
            interim_dir=tmp_path,
        ),
        model=ModelConfig(),
    )
    estimator = RecordingEstimator()

    _, predictions = module._evaluate_official_test(cfg, estimator)

    assert estimator.seen_rows == len(test_raw)
    assert list(predictions["engine_id"]) == [1, 2]
    assert list(predictions["cycle"]) == [3, 2]
    assert list(predictions["prediction"]) == [2.0, 4.0]
