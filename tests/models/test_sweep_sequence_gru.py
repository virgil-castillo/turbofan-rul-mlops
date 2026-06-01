"""Tests for turbofan.experiments.sequence_gru_sweep."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from turbofan.config.schema import DataConfig, ProjectConfig, SequenceConfig


class _FakePipeline:
    """Minimal pipeline test double returning input unchanged."""

    def __init__(self, feature_cols: list[str] | None = None, **kwargs: object) -> None:
        """Initialize with optional feature_cols list.

        Args:
            feature_cols: Columns the pipeline exposes via feature_engineer.
            **kwargs: Absorbed for compatibility with build_feature_pipeline kwargs.
        """
        _fc = feature_cols or ["s_1"]
        _fake_fe = type("FakeFeatureEngineer", (), {"feature_cols_": _fc})()
        _fake_norm = type("FakeNorm", (), {"to_payload": lambda self: {}})()
        self.named_steps: dict[str, object] = {
            "normalizer": _fake_norm,
            "feature_engineer": _fake_fe,
        }

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return input unchanged.

        Args:
            frame: Input DataFrame.

        Returns:
            The same DataFrame, unmodified.
        """
        return frame

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return input unchanged.

        Args:
            frame: Input DataFrame.

        Returns:
            The same DataFrame, unmodified.
        """
        return frame


def _load_module(project_root: Path) -> ModuleType:
    """Load the GRU sweep module for helper testing.

    Args:
        project_root: Repository root path.

    Returns:
        Imported script module.

    """
    del project_root
    from turbofan.experiments import sequence_gru_sweep

    return sequence_gru_sweep


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


def test_gru_sweep_returns_expected_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GRU sweep evaluates the Cartesian product of requested specs."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(module, "append_training_log", lambda entry: None)

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
        "training_duration_seconds",
    ]
    assert results["rmse"].is_monotonic_increasing


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


def test_gru_sweep_reports_validation_window_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GRU sweep reports metrics from sliding validation windows."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg = ProjectConfig(
        project_name="test",
        data=DataConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
        ),
        sequence=SequenceConfig(
            architecture="gru",
            window_size=3,
            batch_size=4,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            epochs=1,
        ),
    )

    class FakeWindows:
        """Minimal window container with metric targets."""

        def __init__(self, y: list[float]) -> None:
            self.y = np.asarray(y, dtype=np.float64)

    train_windows = FakeWindows([0.0])
    validation_windows = FakeWindows([10.0, 20.0])

    _fake_train_df = pd.DataFrame(
        {"engine_id": [1, 1], "cycle": [1, 2], "rul": [2, 1], "s_1": [0.1, 0.2]}
    )
    _fake_val_df = pd.DataFrame(
        {"engine_id": [2, 2], "cycle": [1, 2], "rul": [2, 1], "s_1": [0.3, 0.4]}
    )

    _window_call_count: list[int] = [0]

    def fake_build_sliding_windows(frame: object, **kwargs: object) -> FakeWindows:
        """Return train windows on odd calls and validation windows on even calls.

        Args:
            frame: Normalized DataFrame passed to windowing.
            **kwargs: Ignored windowing parameters.

        Returns:
            Train or validation FakeWindows alternating by call order.
        """
        del frame, kwargs
        _window_call_count[0] += 1
        if _window_call_count[0] % 2 == 1:
            return train_windows
        return validation_windows

    def fake_predict_windows(
        model: object,
        loader: FakeWindows,
        device: object,
        max_rul: int = 1,
    ) -> np.ndarray:
        """Return predictions matching validation targets when loader is validation.

        Args:
            model: Ignored model placeholder.
            loader: Window loader to identify train vs validation.
            device: Ignored device placeholder.
            max_rul: Ignored RUL cap.

        Returns:
            Perfect predictions for validation_windows, zeros otherwise.
        """
        del model
        del device
        if loader is validation_windows:
            return np.asarray([10.0, 20.0], dtype=np.float64)
        return np.asarray([0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(module, "load_config", lambda path: cfg)
    monkeypatch.setattr(module, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(module, "load_raw_train", lambda data_config: object())
    monkeypatch.setattr(module, "add_rul_column", lambda frame, max_rul: frame)
    monkeypatch.setattr(
        module,
        "split_by_engine",
        lambda frame, test_size, random_seed: (_fake_train_df, _fake_val_df),
    )
    monkeypatch.setattr(
        module, "build_feature_pipeline", lambda **kw: _FakePipeline(["s_1"])
    )
    monkeypatch.setattr(module, "build_sliding_windows", fake_build_sliding_windows)
    monkeypatch.setattr(
        module,
        "build_sequence_loader",
        lambda windows, batch_size, shuffle: windows,
    )
    monkeypatch.setattr(module, "seed_everything", lambda random_seed: None)
    monkeypatch.setattr(module, "GRURULRegressor", lambda **kwargs: object())
    monkeypatch.setattr(
        module,
        "train_gru_model",
        lambda **kwargs: type("Result", (), {"model": object(), "best_epoch": 1})(),
    )
    monkeypatch.setattr(module, "predict_windows", fake_predict_windows)
    monkeypatch.setattr(module, "append_training_log", lambda entry: None)

    results = module.run_gru_sweep(
        config_path=tmp_path / "config.yaml",
        window_sizes=[3],
        hidden_sizes=[4],
        learning_rates=[1e-3],
        device="cpu",
    )

    row = results.iloc[0]
    assert row["rmse"] == 0.0
    assert row["mae"] == 0.0
    assert "phm08_score" not in results.columns


def test_gru_sweep_appends_training_log_entry_per_completed_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GRU sweep logs one entry per completed config using sliding metrics."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg = ProjectConfig(
        project_name="test",
        data=DataConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
            fd_subset="FD003",
            random_seed=321,
        ),
        sequence=SequenceConfig(
            architecture="gru",
            window_size=3,
            batch_size=4,
            hidden_size=4,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            epochs=5,
            patience=3,
        ),
    )

    class FakeWindows:
        """Minimal window container with metric targets."""

        def __init__(self, y: list[float]) -> None:
            self.y = np.asarray(y, dtype=np.float64)

    validation_windows = FakeWindows([10.0, 20.0])
    build_calls: list[dict[str, object]] = []
    appended_entries: list[dict[str, object]] = []
    timer_values = iter([1.0, 1.5, 4.0, 6.25])

    _fake_train_df = pd.DataFrame(
        {"engine_id": [1, 1], "cycle": [1, 2], "rul": [2, 1], "s_1": [0.1, 0.2]}
    )
    _fake_val_df = pd.DataFrame(
        {"engine_id": [2, 2], "cycle": [1, 2], "rul": [2, 1], "s_1": [0.3, 0.4]}
    )

    _window_call_count: list[int] = [0]

    def fake_build_sliding_windows(frame: object, **kwargs: object) -> FakeWindows:
        """Return train windows on odd calls and validation windows on even calls.

        Args:
            frame: Normalized DataFrame passed to windowing.
            **kwargs: Ignored windowing parameters.

        Returns:
            Fresh FakeWindows for train (odd calls); validation_windows for
            val (even calls).
        """
        del frame, kwargs
        _window_call_count[0] += 1
        if _window_call_count[0] % 2 == 1:
            return FakeWindows([0.0])
        return validation_windows

    def fake_predict_windows(
        model: object,
        loader: FakeWindows,
        device: object,
        max_rul: int = 1,
    ) -> np.ndarray:
        """Return predictions matching validation targets when loader is validation.

        Args:
            model: Ignored model placeholder.
            loader: Window loader to identify train vs validation.
            device: Ignored device placeholder.
            max_rul: Ignored RUL cap.

        Returns:
            Perfect predictions for validation_windows, zeros otherwise.
        """
        del model
        del device
        if loader is validation_windows:
            return np.asarray([10.0, 20.0], dtype=np.float64)
        return np.asarray([0.0], dtype=np.float64)

    def fake_build_log_entry(**kwargs: object) -> dict[str, object]:
        build_calls.append(kwargs)
        return {"entry": kwargs}

    monkeypatch.setattr(module, "load_config", lambda path: cfg)
    monkeypatch.setattr(module, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(module, "load_raw_train", lambda data_config: object())
    monkeypatch.setattr(module, "add_rul_column", lambda frame, max_rul: frame)
    monkeypatch.setattr(
        module,
        "split_by_engine",
        lambda frame, test_size, random_seed: (_fake_train_df, _fake_val_df),
    )
    monkeypatch.setattr(
        module, "build_feature_pipeline", lambda **kw: _FakePipeline(["s_1"])
    )
    monkeypatch.setattr(module, "build_sliding_windows", fake_build_sliding_windows)
    monkeypatch.setattr(
        module,
        "build_sequence_loader",
        lambda windows, batch_size, shuffle: windows,
    )
    monkeypatch.setattr(module, "seed_everything", lambda random_seed: None)
    monkeypatch.setattr(module, "GRURULRegressor", lambda **kwargs: object())
    monkeypatch.setattr(
        module,
        "train_gru_model",
        lambda **kwargs: type("Result", (), {"model": object(), "best_epoch": 9})(),
    )
    monkeypatch.setattr(module, "predict_windows", fake_predict_windows)
    monkeypatch.setattr(module, "build_log_entry", fake_build_log_entry, raising=False)
    monkeypatch.setattr(
        module,
        "append_training_log",
        lambda entry: appended_entries.append(entry),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "perf_counter",
        lambda: next(timer_values),
        raising=False,
    )

    results = module.run_gru_sweep(
        config_path=tmp_path / "config.yaml",
        window_sizes=[3, 4],
        hidden_sizes=[6],
        learning_rates=[0.002],
        device="cpu",
    )

    assert build_calls == [
        {
            "model_type": "gru",
            "dataset": "FD003",
            "random_seed": 321,
            "hyperparameters": {
                "window_size": 3,
                "hidden_size": 6,
                "learning_rate": 0.002,
                "num_layers": 2,
                "dropout": 0.2,
                "batch_size": 4,
                "epochs": 5,
                "patience": 3,
            },
            "metrics": {"rmse": 0.0, "mae": 0.0},
            "training_duration_seconds": 0.5,
            "device": "cpu",
            "run_dir": None,
            "best_epoch": 9,
        },
        {
            "model_type": "gru",
            "dataset": "FD003",
            "random_seed": 321,
            "hyperparameters": {
                "window_size": 4,
                "hidden_size": 6,
                "learning_rate": 0.002,
                "num_layers": 2,
                "dropout": 0.2,
                "batch_size": 4,
                "epochs": 5,
                "patience": 3,
            },
            "metrics": {"rmse": 0.0, "mae": 0.0},
            "training_duration_seconds": 2.25,
            "device": "cpu",
            "run_dir": None,
            "best_epoch": 9,
        },
    ]
    assert appended_entries == [{"entry": call} for call in build_calls]
    assert list(results["training_duration_seconds"]) == [0.5, 2.25]


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
            "-m",
            "turbofan.experiments.sequence_gru_sweep",
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
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "window_size" in result.stdout
    assert "run 1/1" in result.stdout
    results = pd.read_csv(output_path)
    assert len(results) == 1
    assert "training_duration_seconds" in results.columns
    assert results.loc[0, "training_duration_seconds"] > 0.0
    assert results["rmse"].is_monotonic_increasing


def test_sweep_sequence_gru_cli_writes_default_csv(tmp_path: Path) -> None:
    """CLI writes a default GRU sweep CSV when output is omitted."""
    project_root = Path(__file__).parent.parent.parent
    cfg_path = _write_config(tmp_path)
    output_path = tmp_path / "results" / "gru_sweep.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "turbofan.experiments.sequence_gru_sweep",
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
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "window_size" in result.stdout
    assert "run 1/1" in result.stdout
    results = pd.read_csv(output_path)
    assert len(results) == 1
    assert "training_duration_seconds" in results.columns
    assert results.loc[0, "training_duration_seconds"] > 0.0
    assert results["rmse"].is_monotonic_increasing
