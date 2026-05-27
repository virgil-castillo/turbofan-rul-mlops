"""Tests for turbofan.experiments.feature_gru_sweep."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _load_module(project_root: Path) -> ModuleType:
    """Load the feature GRU sweep module for helper testing.

    Args:
        project_root: Repository root path.

    Returns:
        Imported script module.

    """
    del project_root
    from turbofan.experiments import feature_gru_sweep

    return feature_gru_sweep


def _write_cmapps_file(path: Path, n_engines: int, n_cycles: int) -> None:
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
            lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines))


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal project config for feature GRU sweep tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Created config path.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=4, n_cycles=6)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n30\n40\n")
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
                "sequence:",
                "  architecture: gru",
                "  window_size: 3",
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


def test_feature_sweep_validates_feature_set_names(tmp_path: Path) -> None:
    """Invalid feature set names raise ValueError before training."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)

    with pytest.raises(ValueError, match="Invalid feature sets"):
        module._validate_inputs(["invalid_set"], [0.5], 10)


def test_feature_sweep_validates_corr_thresholds(tmp_path: Path) -> None:
    """Out-of-range correlation thresholds raise ValueError."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)

    with pytest.raises(ValueError, match="[Cc]orrelation"):
        module._validate_inputs(["raw"], [0.0], 10)

    with pytest.raises(ValueError, match="[Cc]orrelation"):
        module._validate_inputs(["raw"], [1.0], 10)

    with pytest.raises(ValueError, match="[Cc]orrelation"):
        module._validate_inputs(["raw"], [], 10)


def test_feature_sweep_validates_rolling_window(tmp_path: Path) -> None:
    """Non-positive rolling window raises ValueError."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)

    with pytest.raises(ValueError, match="[Rr]olling"):
        module._validate_inputs(["raw"], [0.5], 0)


def test_feature_sweep_grid_produces_correct_number_of_runs() -> None:
    """Sweep grid yields 8 entries for default feature sets and 3 thresholds."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)

    grid = module._build_sweep_grid(
        ["raw", "raw_plus_rolling", "top_corr", "top_corr_rolling"],
        [0.3, 0.5, 0.7],
    )

    assert len(grid) == 8
    # raw and raw_plus_rolling each contribute one entry with threshold None
    assert grid[0] == ("raw", None)
    assert grid[1] == ("raw_plus_rolling", None)
    # top_corr contributes 3 entries
    assert grid[2] == ("top_corr", 0.3)
    assert grid[3] == ("top_corr", 0.5)
    assert grid[4] == ("top_corr", 0.7)
    # top_corr_rolling contributes 3 entries
    assert grid[5] == ("top_corr_rolling", 0.3)
    assert grid[6] == ("top_corr_rolling", 0.5)
    assert grid[7] == ("top_corr_rolling", 0.7)


def test_feature_sweep_returns_expected_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Feature sweep returns a DataFrame with required columns."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(module, "append_training_log", lambda entry: None)

    results = module.run_feature_sweep(
        config_path=cfg_path,
        feature_sets=["raw"],
        corr_thresholds=[0.5],
        rolling_window=3,
        device="cpu",
    )

    assert len(results) == 1
    assert list(results.columns) == [
        "feature_set",
        "corr_threshold",
        "n_features",
        "best_epoch",
        "rmse",
        "mae",
        "phm08_score",
        "test_rmse",
        "test_mae",
        "test_phm08_score",
    ]


def test_feature_sweep_cli_writes_csv(tmp_path: Path) -> None:
    """CLI writes a sorted feature sweep CSV when output is supplied."""
    project_root = Path(__file__).parent.parent.parent
    cfg_path = _write_config(tmp_path)
    output_path = tmp_path / "feature_sweep.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "turbofan.experiments.feature_gru_sweep",
            "--config",
            str(cfg_path),
            "--feature-sets",
            "raw",
            "--corr-thresholds",
            "0.5",
            "--rolling-window",
            "3",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "run 1/1" in result.stdout
    assert output_path.exists()
    results = pd.read_csv(output_path)
    assert list(results.columns) == [
        "feature_set",
        "corr_threshold",
        "n_features",
        "best_epoch",
        "rmse",
        "mae",
        "phm08_score",
        "test_rmse",
        "test_mae",
        "test_phm08_score",
    ]
    assert len(results) == 1


def test_feature_sweep_includes_test_metric_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Feature sweep result includes test_rmse, test_mae, test_phm08_score."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(module, "append_training_log", lambda entry: None)

    results = module.run_feature_sweep(
        config_path=cfg_path,
        feature_sets=["raw"],
        corr_thresholds=[0.5],
        rolling_window=3,
        device="cpu",
    )

    assert "test_rmse" in results.columns
    assert "test_mae" in results.columns
    assert "test_phm08_score" in results.columns
