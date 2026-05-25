"""Tests for scripts/sweep_sequence_gru.py."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _load_module(project_root: Path) -> ModuleType:
    """Load the GRU sweep script as a module for helper testing.

    Args:
        project_root: Repository root path.

    Returns:
        Imported script module.

    Raises:
        RuntimeError: If the script cannot be imported.
    """
    script_path = project_root / "scripts" / "sweep_sequence_gru.py"
    spec = importlib.util.spec_from_file_location("sweep_sequence_gru", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    """Write a minimal project config for GRU sweep tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Created config path.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
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


def test_gru_sweep_returns_expected_rows(tmp_path: Path) -> None:
    """GRU sweep evaluates the Cartesian product of requested specs."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)

    results = module.run_gru_sweep(
        config_path=cfg_path,
        window_sizes=[3, 4],
        hidden_sizes=[2, 3],
        learning_rates=[1e-3],
        device="cpu",
    )

    assert len(results) == 4
    observed_specs = {
        (row.window_size, row.hidden_size, row.learning_rate)
        for row in results.itertuples(index=False)
    }
    assert observed_specs == {
        (3, 2, 1e-3),
        (3, 3, 1e-3),
        (4, 2, 1e-3),
        (4, 3, 1e-3),
    }
    assert list(results.columns) == [
        "window_size",
        "hidden_size",
        "learning_rate",
        "best_epoch",
        "rmse",
        "mae",
        "phm08_score",
    ]
    assert results["phm08_score"].is_monotonic_increasing


def test_gru_sweep_validates_inputs(tmp_path: Path) -> None:
    """Invalid GRU sweep grids fail before training."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="window"):
        module.run_gru_sweep(cfg_path, [0], [2], [1e-3], device="cpu")
    with pytest.raises(ValueError, match="hidden"):
        module.run_gru_sweep(cfg_path, [3], [], [1e-3], device="cpu")
    with pytest.raises(ValueError, match="learning"):
        module.run_gru_sweep(cfg_path, [3], [2], [0.0], device="cpu")


def test_sweep_sequence_gru_cli_writes_csv(tmp_path: Path) -> None:
    """CLI writes a sorted GRU sweep CSV when output is supplied."""
    project_root = Path(__file__).parent.parent.parent
    cfg_path = _write_config(tmp_path)
    output_path = tmp_path / "gru_sweep.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_sequence_gru.py",
            "--config",
            str(cfg_path),
            "--window-sizes",
            "3",
            "--hidden-sizes",
            "2",
            "--learning-rates",
            "1e-3",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "window_size" in result.stdout
    assert "run 1/1" in result.stdout
    results = pd.read_csv(output_path)
    assert len(results) == 1
    assert results["phm08_score"].is_monotonic_increasing
