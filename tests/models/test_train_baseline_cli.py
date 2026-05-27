"""Smoke tests for turbofan.cli.train_baseline."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from turbofan.config.schema import DataConfig, ModelConfig, ProjectConfig


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


def test_train_baseline_cli_writes_artifacts(tmp_path: Path) -> None:
    """CLI trains on synthetic files and writes model artifacts."""
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
                "  feature_set: raw_plus_rolling",
                "  windows:",
                "    - 5",
                f"  artifact_dir: {artifact_dir.as_posix()}",
                "features:",
                "  sensor_std_threshold: 0.02",
                "  sensor_keep:",
                "    - s_2",
            ]
        )
    )

    project_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "turbofan.cli.train_baseline",
            "--config",
            str(cfg_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "validation rmse" in result.stdout
    run_dirs = list((artifact_dir / "baseline").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "model_manifest.json").exists()
    assert (run_dir / "validation_predictions.csv").exists()
    assert (run_dir / "official_test_predictions.csv").exists()

    manifest = json.loads((run_dir / "model_manifest.json").read_text())
    assert manifest == {
        "schema_version": 1,
        "model_type": "ridge",
        "artifact_id": f"baseline/{run_dir.name}",
        "prediction_scope": "row",
        "model_path": "model.joblib",
        "config_path": "config.json",
        "metrics_path": "metrics.json",
    }

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "validation" in metrics
    assert "official_test" in metrics
    assert set(metrics["validation"]) == {"rmse", "mae", "phm08_score"}

    estimator = joblib.load(run_dir / "model.joblib")
    dropper = estimator.named_steps["features"].named_steps["sensor_dropper"]
    assert dropper.std_threshold == 0.02
    assert dropper.keep == ["s_2"]
    rolling = estimator.named_steps["features"].named_steps["rolling_features"]
    selector = estimator.named_steps["select_model_features"]
    assert rolling.windows == [5]
    assert selector.feature_set == "raw_plus_rolling"


def test_train_baseline_cli_skips_missing_official_test(
    tmp_path: Path,
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

    project_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "turbofan.cli.train_baseline",
            "--config",
            str(cfg_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "official test evaluation skipped" in result.stdout
    run_dir = next((artifact_dir / "baseline").iterdir())
    assert not (run_dir / "official_test_predictions.csv").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"validation"}


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
    monkeypatch.setattr(module, "load_raw_test", lambda cfg: test_raw)
    monkeypatch.setattr(
        module,
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


def test_clip_rul_predictions_bounds_values_to_rul_cap(tmp_path: Path) -> None:
    """Prediction post-processing clips values into the configured RUL range."""
    module = _load_train_baseline_module()

    clipped = module._clip_rul_predictions(
        np.array([-5.0, 10.0, 200.0], dtype=np.float64),
        rul_cap=125,
    )

    assert clipped.tolist() == [0.0, 10.0, 125.0]
