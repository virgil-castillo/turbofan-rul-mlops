"""Tests for turbofan.experiments.feature_sweep."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _load_module() -> ModuleType:
    """Import the feature_sweep module under test.

    Returns:
        Imported feature_sweep module.
    """
    from turbofan.experiments import feature_sweep

    return feature_sweep


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
            lines.append(" ".join(str(v) for v in values))
    path.write_text("\n".join(lines))


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal project config usable for both ridge and gru sweeps.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Created config path.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=8)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=2, n_cycles=6)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n")
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


# ---------------------------------------------------------------------------
# Unit tests: _build_experiment_specs
# ---------------------------------------------------------------------------


def test_build_experiment_specs_raw_yields_one_spec() -> None:
    """raw feature set produces exactly one spec with no windows or lag."""
    module = _load_module()
    specs = module._build_experiment_specs(["raw"], windows=[5, 10], lag_steps=[1, 2])
    assert len(specs) == 1
    assert specs[0].feature_set == "raw"
    assert specs[0].windows == ()
    assert specs[0].lag_steps == ()


def test_build_experiment_specs_rolling_yields_one_spec_per_window() -> None:
    """Rolling feature set produces one spec per window value."""
    module = _load_module()
    specs = module._build_experiment_specs(
        ["rolling_mean"], windows=[5, 10], lag_steps=[1]
    )
    assert len(specs) == 2
    assert all(s.feature_set == "rolling_mean" for s in specs)
    assert {s.windows for s in specs} == {(5,), (10,)}


def test_build_experiment_specs_lag_yields_one_spec_per_lag_step() -> None:
    """lag feature set produces one spec per lag_step value."""
    module = _load_module()
    specs = module._build_experiment_specs(["lag"], windows=[5], lag_steps=[1, 3])
    assert len(specs) == 2
    assert all(s.feature_set == "lag" for s in specs)
    assert {s.lag_steps for s in specs} == {(1,), (3,)}


def test_build_experiment_specs_mixed_sets_total_count() -> None:
    """Mixed feature sets produce correct total spec count."""
    module = _load_module()
    specs = module._build_experiment_specs(
        ["raw", "rolling_mean", "lag"],
        windows=[5, 10],
        lag_steps=[1, 2],
    )
    # raw=1, rolling_mean=2 (one per window), lag=2 (one per lag_step)
    assert len(specs) == 5


def test_build_experiment_specs_raw_plus_lag_yields_one_spec_per_lag_step() -> None:
    """raw_plus_lag produces one spec per lag_step, like lag."""
    module = _load_module()
    specs = module._build_experiment_specs(
        ["raw_plus_lag"], windows=[5], lag_steps=[1, 3]
    )
    assert len(specs) == 2
    assert all(s.feature_set == "raw_plus_lag" for s in specs)
    assert {s.lag_steps for s in specs} == {(1,), (3,)}


def test_validate_inputs_accepts_raw_plus_lag() -> None:
    """raw_plus_lag is a valid feature set."""
    module = _load_module()
    result = module._validate_inputs("ridge", ["raw_plus_lag"], [5], [1], n_jobs=1)
    assert "raw_plus_lag" in result


# ---------------------------------------------------------------------------
# Unit tests: _validate_inputs
# ---------------------------------------------------------------------------


def test_validate_inputs_rejects_unknown_feature_set() -> None:
    """Unknown feature set name raises ValueError."""
    module = _load_module()
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        module._validate_inputs("ridge", ["not_a_feature_set"], [5], [1], n_jobs=1)


def test_validate_inputs_rejects_nonpositive_window() -> None:
    """Non-positive window size raises ValueError."""
    module = _load_module()
    with pytest.raises(ValueError, match="[Pp]ositive"):
        module._validate_inputs("ridge", ["rolling_mean"], [0], [1], n_jobs=1)


def test_validate_inputs_rejects_nonpositive_lag_step() -> None:
    """Non-positive lag step raises ValueError."""
    module = _load_module()
    with pytest.raises(ValueError, match="[Pp]ositive"):
        module._validate_inputs("ridge", ["lag"], [5], [0], n_jobs=1)


def test_validate_inputs_rejects_zero_n_jobs() -> None:
    """n_jobs=0 raises ValueError."""
    module = _load_module()
    with pytest.raises(ValueError, match="n_jobs"):
        module._validate_inputs("ridge", ["raw"], [5], [1], n_jobs=0)


def test_validate_inputs_rejects_invalid_model() -> None:
    """Unknown model name raises ValueError."""
    module = _load_module()
    with pytest.raises(ValueError, match="[Mm]odel"):
        module._validate_inputs("xgboost", ["raw"], [5], [1], n_jobs=1)


# ---------------------------------------------------------------------------
# Integration tests: run_feature_sweep
# ---------------------------------------------------------------------------


def test_run_feature_sweep_ridge_returns_expected_columns(tmp_path: Path) -> None:
    """Ridge sweep DataFrame has all required columns."""
    module = _load_module()
    cfg_path = _write_config(tmp_path)
    results = module.run_feature_sweep(
        config_path=cfg_path,
        model="ridge",
        feature_sets=["raw"],
        windows=[5],
        lag_steps=[1],
        n_jobs=1,
    )
    assert set(results.columns) == {
        "model",
        "feature_set",
        "windows",
        "lag_steps",
        "n_features",
        "alpha",
        "rmse",
        "mae",
        "phm08_score",
    }
    assert (results["model"] == "ridge").all()


def test_run_feature_sweep_gru_returns_expected_columns(tmp_path: Path) -> None:
    """GRU sweep DataFrame has all required columns including best_epoch."""
    module = _load_module()
    cfg_path = _write_config(tmp_path)
    results = module.run_feature_sweep(
        config_path=cfg_path,
        model="gru",
        feature_sets=["raw"],
        windows=[5],
        lag_steps=[1],
        device="cpu",
    )
    assert set(results.columns) == {
        "model",
        "feature_set",
        "windows",
        "lag_steps",
        "n_features",
        "best_epoch",
        "rmse",
        "mae",
        "phm08_score",
    }
    assert (results["model"] == "gru").all()


def test_run_feature_sweep_raw_produces_one_row(tmp_path: Path) -> None:
    """raw feature set always produces exactly one result row."""
    module = _load_module()
    cfg_path = _write_config(tmp_path)
    results = module.run_feature_sweep(
        config_path=cfg_path,
        model="ridge",
        feature_sets=["raw"],
        windows=[5, 10],
        lag_steps=[1, 2],
        n_jobs=1,
    )
    assert len(results) == 1


def test_run_feature_sweep_rolling_mean_two_windows_produces_two_rows(
    tmp_path: Path,
) -> None:
    """rolling_mean with two windows produces two result rows."""
    module = _load_module()
    cfg_path = _write_config(tmp_path)
    results = module.run_feature_sweep(
        config_path=cfg_path,
        model="ridge",
        feature_sets=["rolling_mean"],
        windows=[3, 5],
        lag_steps=[1],
        n_jobs=1,
    )
    assert len(results) == 2


def test_run_feature_sweep_results_sorted_by_phm08_score(tmp_path: Path) -> None:
    """Results are sorted ascending by phm08_score."""
    module = _load_module()
    cfg_path = _write_config(tmp_path)
    results = module.run_feature_sweep(
        config_path=cfg_path,
        model="ridge",
        feature_sets=["raw", "rolling_mean"],
        windows=[5],
        lag_steps=[1],
        n_jobs=1,
    )
    assert results["phm08_score"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_run_feature_sweep_gru_calls_training_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GRU sweep calls append_training_log once per trained spec."""
    module = _load_module()
    cfg_path = _write_config(tmp_path)
    log_calls: list[object] = []
    monkeypatch.setattr(
        module, "append_training_log", lambda entry: log_calls.append(entry)
    )
    module.run_feature_sweep(
        config_path=cfg_path,
        model="gru",
        feature_sets=["raw"],
        windows=[5],
        lag_steps=[1],
        device="cpu",
        output_path=tmp_path / "out.csv",
    )
    assert len(log_calls) == 1


def test_run_feature_sweep_writes_default_output_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """run_feature_sweep writes a default CSV when output_path is None."""
    module = _load_module()
    cfg_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    module.run_feature_sweep(
        config_path=cfg_path,
        model="ridge",
        feature_sets=["raw"],
        windows=[5],
        lag_steps=[1],
        n_jobs=1,
    )
    assert (tmp_path / "results" / "feature_sweep_ridge_fd001.csv").exists()


def test_feature_sweep_cli_ridge_writes_csv(tmp_path: Path) -> None:
    """CLI --model ridge writes a sorted results CSV."""
    project_root = Path(__file__).parent.parent.parent
    cfg_path = _write_config(tmp_path)
    output_path = tmp_path / "sweep_results.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "turbofan.experiments.feature_sweep",
            "--config",
            str(cfg_path),
            "--model",
            "ridge",
            "--feature-sets",
            "raw",
            "--windows",
            "5",
            "--lag-steps",
            "1",
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
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert "model" in df.columns
    assert df["model"].iloc[0] == "ridge"
    assert len(df) == 1


def test_feature_sweep_cli_gru_writes_csv(tmp_path: Path) -> None:
    """CLI --model gru writes a CSV with best_epoch column."""
    project_root = Path(__file__).parent.parent.parent
    cfg_path = _write_config(tmp_path)
    output_path = tmp_path / "sweep_results_gru.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "turbofan.experiments.feature_sweep",
            "--config",
            str(cfg_path),
            "--model",
            "gru",
            "--feature-sets",
            "raw",
            "--windows",
            "5",
            "--lag-steps",
            "1",
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

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert "best_epoch" in df.columns
    assert df["model"].iloc[0] == "gru"
    assert len(df) == 1
