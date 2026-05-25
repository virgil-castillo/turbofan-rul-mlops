"""Tests for scripts/compare_baseline_features.py."""
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
    """Load the comparison script as a module for helper testing.

    Args:
        project_root: Repository root path.

    Returns:
        Imported script module.

    Raises:
        RuntimeError: If the script cannot be imported.
    """
    script_path = project_root / "scripts" / "compare_baseline_features.py"
    spec = importlib.util.spec_from_file_location(
        "compare_baseline_features",
        script_path,
    )
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
            sensors = [float(cycle + i + engine_id) for i in range(1, 22)]
            values = [engine_id, cycle, *op_cols, *sensors]
            lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines))


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal project config for comparison tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Created config path.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=8)
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
            ]
        )
    )
    return cfg_path


def test_feature_comparison_returns_expected_rows(tmp_path: Path) -> None:
    """Comparison helper evaluates raw once and rolling variants per window."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)

    results = module.run_feature_comparison(
        config_path=cfg_path,
        feature_sets=["raw", "raw_plus_rolling", "rolling"],
        windows=[3, 5],
        n_jobs=1,
    )

    assert len(results) == 5
    observed_specs = {
        (row.feature_set, row.windows) for row in results.itertuples(index=False)
    }
    assert observed_specs == {
        ("raw", ""),
        ("raw_plus_rolling", "3"),
        ("rolling", "3"),
        ("raw_plus_rolling", "5"),
        ("rolling", "5"),
    }
    assert set(results.columns) == {
        "feature_set",
        "windows",
        "alpha",
        "n_features",
        "raw_prediction_min",
        "raw_prediction_max",
        "rmse",
        "mae",
        "phm08_score",
    }


def test_feature_comparison_validates_inputs(tmp_path: Path) -> None:
    """Invalid comparison inputs fail before training."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        module.run_feature_comparison(cfg_path, ["raw"], [0], n_jobs=1)
    with pytest.raises(ValueError, match="n_jobs"):
        module.run_feature_comparison(cfg_path, ["raw"], [3], n_jobs=0)
    with pytest.raises(ValueError, match="Unsupported feature_set"):
        module.run_feature_comparison(cfg_path, ["bad"], [3], n_jobs=1)


def test_compare_baseline_features_cli_writes_csv(tmp_path: Path) -> None:
    """CLI writes a sorted comparison CSV when output is supplied."""
    project_root = Path(__file__).parent.parent.parent
    cfg_path = _write_config(tmp_path)
    output_path = tmp_path / "comparison.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_baseline_features.py",
            "--config",
            str(cfg_path),
            "--feature-sets",
            "raw",
            "raw_plus_rolling",
            "rolling",
            "--windows",
            "3",
            "--n-jobs",
            "1",
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "feature_set" in result.stdout
    results = pd.read_csv(output_path)
    assert len(results) == 3
    assert results["phm08_score"].is_monotonic_increasing
