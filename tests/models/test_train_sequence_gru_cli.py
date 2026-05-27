"""Smoke tests for turbofan.cli.train_sequence_gru."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from csv import DictReader
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import torch

from turbofan.config.schema import DataConfig, ProjectConfig, SequenceConfig
from turbofan.models.gru import GRURULRegressor
from turbofan.models.sequence_training import TrainingResult


class _FakeNormalizer:
    """Minimal sequence normalizer test double."""

    def __init__(self, feature_cols: list[str]) -> None:
        self.feature_cols = feature_cols
        self.means_ = pd.Series({feature: 0.0 for feature in feature_cols})
        self.stds_ = pd.Series({feature: 1.0 for feature in feature_cols})

    def fit_transform(self, frame: object) -> object:
        """Return the input training frame unchanged.

        Args:
            frame: Training frame placeholder.

        Returns:
            Unchanged frame placeholder.
        """
        return frame

    def transform(self, frame: object) -> object:
        """Return the input frame unchanged.

        Args:
            frame: Frame placeholder.

        Returns:
            Unchanged frame placeholder.
        """
        return frame


def _write_cmapps_file(path: Path, n_engines: int, n_cycles: int) -> None:
    """Write a small C-MAPSS-style whitespace-delimited file.

    Args:
        path: Destination file path.
        n_engines: Number of synthetic engines to write.
        n_cycles: Number of cycles per engine.
    """
    lines = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            op_cols = [0.0, 0.0, 0.0]
            sensors = [
                float((engine_id * 0.1) + (cycle * 0.2) + sensor_idx)
                for sensor_idx in range(1, 22)
            ]
            values = [engine_id, cycle, *op_cols, *sensors]
            lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines))


def _write_cmapps_file_by_cycles(path: Path, cycles_by_engine: dict[int, int]) -> None:
    """Write a small C-MAPSS-style file with per-engine cycle counts.

    Args:
        path: Destination file path.
        cycles_by_engine: Mapping from engine ID to number of cycles.
    """
    lines = []
    for engine_id, n_cycles in cycles_by_engine.items():
        for cycle in range(1, n_cycles + 1):
            op_cols = [0.0, 0.0, 0.0]
            sensors = [
                float((engine_id * 0.1) + (cycle * 0.2) + sensor_idx)
                for sensor_idx in range(1, 22)
            ]
            values = [engine_id, cycle, *op_cols, *sensors]
            lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines))


def _write_config(
    path: Path,
    raw_dir: Path,
    artifact_dir: Path,
    tmp_path: Path,
) -> None:
    """Write a tiny GRU training config.

    Args:
        path: Destination YAML path.
        raw_dir: Raw C-MAPSS data directory.
        artifact_dir: Artifact root.
        tmp_path: Temporary test directory for other configured paths.
    """
    path.write_text(
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
                "  batch_size: 4",
                "  hidden_size: 4",
                "  num_layers: 1",
                "  dropout: 0.0",
                "  learning_rate: 0.001",
                "  epochs: 2",
                "  patience: 2",
                "  device: cpu",
                f"  artifact_dir: {artifact_dir.as_posix()}",
            ]
        )
    )


def _assert_metric_keys(metrics: dict[str, object], section: str) -> None:
    """Assert a metrics section contains the expected regression metrics.

    Args:
        metrics: Metrics payload loaded from JSON.
        section: Top-level metric section name.
    """
    assert set(metrics[section]) == {
        "rmse",
        "mae",
        "phm08_score",
    }


def _run_cli(cfg_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the sequence GRU CLI with the worktree source path.

    Args:
        cfg_path: YAML config path.

    Returns:
        Completed subprocess result.
    """
    project_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "turbofan.cli.train_sequence_gru",
            "--config",
            str(cfg_path),
        ],
        cwd=cfg_path.parent,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _load_train_sequence_gru_module() -> ModuleType:
    """Load the GRU training CLI module.

    Returns:
        Imported CLI module.
    """
    from turbofan.cli import train_sequence_gru

    return train_sequence_gru


def test_train_sequence_gru_cli_seeds_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI seeds torch before constructing the GRU model."""
    module = _load_train_sequence_gru_module()
    seed = 123
    cfg = ProjectConfig(
        project_name="test",
        data=DataConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
            random_seed=seed,
        ),
        sequence=SequenceConfig(
            architecture="gru",
            window_size=3,
            batch_size=4,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            epochs=1,
            artifact_dir=tmp_path / "artifacts",
        ),
    )
    captured_state: dict[str, torch.Tensor] = {}

    def fake_train_gru_model(
        *,
        model: GRURULRegressor,
        train_loader: object,
        validation_windows_loader: object,
        config: SequenceConfig,
        device: torch.device,
        random_seed: int,
        max_rul: int,
    ) -> TrainingResult:
        del train_loader
        del validation_windows_loader
        del config
        del device
        del random_seed
        del max_rul
        captured_state.update(
            {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        )
        return TrainingResult(
            model=model,
            history=pd.DataFrame([{"epoch": 1}]),
            best_epoch=1,
            best_metric=0.0,
        )

    def fake_torch_save(payload: object, path: Path) -> None:
        del payload
        del path

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: argparse.Namespace(config=tmp_path / "config.yaml"),
    )
    monkeypatch.setattr(module, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        module,
        "resolve_device",
        lambda requested: torch.device("cpu"),
    )
    monkeypatch.setattr(module, "default_feature_cols", lambda: ["s1", "s2"])
    monkeypatch.setattr(module, "load_raw_train", lambda data_config: object())
    monkeypatch.setattr(module, "add_rul_column", lambda frame, max_rul: frame)
    monkeypatch.setattr(
        module,
        "split_by_engine",
        lambda frame, test_size, random_seed: (frame, frame),
    )
    monkeypatch.setattr(module, "SequenceNormalizer", _FakeNormalizer)
    monkeypatch.setattr(
        module,
        "build_sliding_windows",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "build_sequence_loader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(module, "train_gru_model", fake_train_gru_model)
    monkeypatch.setattr(
        module,
        "_evaluate_windows",
        lambda *args, **kwargs: (
            {"rmse": 0.0, "mae": 0.0, "phm08_score": 0.0},
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(module, "_evaluate_official_test", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "create_run_dir", lambda artifact_dir, name: tmp_path)
    monkeypatch.setattr(module, "save_json", lambda payload, path: None)
    monkeypatch.setattr(module, "save_predictions", lambda frame, path: None)
    monkeypatch.setattr(module.torch, "save", fake_torch_save)
    monkeypatch.setattr(module, "append_training_log", lambda entry: None)

    torch.manual_seed(999)
    module.main()

    torch.manual_seed(seed)
    expected = GRURULRegressor(
        input_size=2,
        hidden_size=cfg.sequence.hidden_size,
        num_layers=cfg.sequence.num_layers,
        dropout=cfg.sequence.dropout,
    ).state_dict()
    assert captured_state
    for name, value in expected.items():
        assert torch.equal(captured_state[name], value)


def test_train_sequence_gru_cli_appends_training_log_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI appends a GRU training log entry after artifacts are saved."""
    module = _load_train_sequence_gru_module()
    run_dir = tmp_path / "run"
    cfg = ProjectConfig(
        project_name="test",
        data=DataConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
            fd_subset="FD002",
            random_seed=123,
        ),
        sequence=SequenceConfig(
            architecture="gru",
            window_size=5,
            batch_size=8,
            hidden_size=6,
            num_layers=2,
            dropout=0.1,
            learning_rate=0.002,
            epochs=3,
            patience=2,
            artifact_dir=tmp_path / "artifacts",
        ),
    )
    window_metrics = {"rmse": 1.0, "mae": 2.0, "phm08_score": 3.0}
    build_calls: list[dict[str, object]] = []
    appended_entries: list[dict[str, object]] = []
    timer_values = iter([10.0, 12.5])

    def fake_train_gru_model(**kwargs: object) -> TrainingResult:
        del kwargs
        return TrainingResult(
            model=GRURULRegressor(
                input_size=2,
                hidden_size=6,
                num_layers=2,
                dropout=0.1,
            ),
            history=pd.DataFrame([{"epoch": 1}]),
            best_epoch=7,
            best_metric=0.0,
        )

    def fake_evaluate_windows(
        *args: object,
        **kwargs: object,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        del args
        del kwargs
        return window_metrics, pd.DataFrame()

    def fake_build_log_entry(**kwargs: object) -> dict[str, object]:
        build_calls.append(kwargs)
        return {"entry": kwargs}

    def fake_create_run_dir(artifact_dir: Path, name: str) -> Path:
        del artifact_dir
        del name
        run_dir.mkdir()
        return run_dir

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: argparse.Namespace(config=tmp_path / "config.yaml"),
    )
    monkeypatch.setattr(module, "load_config", lambda path: cfg)
    monkeypatch.setattr(module, "resolve_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr(module, "default_feature_cols", lambda: ["s1", "s2"])
    monkeypatch.setattr(module, "load_raw_train", lambda data_config: object())
    monkeypatch.setattr(module, "add_rul_column", lambda frame, max_rul: frame)
    monkeypatch.setattr(
        module,
        "split_by_engine",
        lambda frame, test_size, random_seed: (frame, frame),
    )
    monkeypatch.setattr(module, "SequenceNormalizer", _FakeNormalizer)
    monkeypatch.setattr(
        module,
        "build_sliding_windows",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "build_sequence_loader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(module, "train_gru_model", fake_train_gru_model)
    monkeypatch.setattr(module, "_evaluate_windows", fake_evaluate_windows)
    monkeypatch.setattr(module, "_evaluate_official_test", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "create_run_dir", fake_create_run_dir)
    monkeypatch.setattr(module, "save_json", lambda payload, path: None)
    monkeypatch.setattr(module, "save_predictions", lambda frame, path: None)
    monkeypatch.setattr(module.torch, "save", lambda payload, path: None)
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

    module.main()

    assert build_calls == [
        {
            "model_type": "gru",
            "dataset": "FD002",
            "random_seed": 123,
            "hyperparameters": {
                "window_size": 5,
                "hidden_size": 6,
                "learning_rate": 0.002,
                "num_layers": 2,
                "dropout": 0.1,
                "batch_size": 8,
                "epochs": 3,
                "patience": 2,
            },
            "metrics": window_metrics,
            "training_duration_seconds": 2.5,
            "device": "cpu",
            "run_dir": str(run_dir),
            "best_epoch": 7,
        }
    ]
    assert appended_entries == [{"entry": build_calls[0]}]


def test_train_sequence_gru_cli_writes_artifacts_with_official_test(
    tmp_path: Path,
) -> None:
    """CLI trains a tiny GRU and writes validation plus official artifacts."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=2, n_cycles=5)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n")

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path)

    result = _run_cli(cfg_path)

    assert "validation_windows rmse" in result.stdout
    run_dirs = list((artifact_dir / "sequence_gru").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "model_manifest.json").exists()
    assert (run_dir / "training_history.csv").exists()
    assert (run_dir / "validation_window_predictions.csv").exists()
    assert (run_dir / "official_test_predictions.csv").exists()

    manifest = json.loads((run_dir / "model_manifest.json").read_text())
    assert manifest == {
        "schema_version": 1,
        "model_type": "gru",
        "artifact_id": f"sequence_gru/{run_dir.name}",
        "prediction_scope": "final_window",
        "model_path": "model.pt",
        "config_path": "config.json",
        "metrics_path": "metrics.json",
    }

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {
        "validation_windows",
        "official_test",
    }
    _assert_metric_keys(metrics, "validation_windows")
    _assert_metric_keys(metrics, "official_test")


def test_train_sequence_gru_cli_aligns_official_labels_to_eligible_test_engines(
    tmp_path: Path,
) -> None:
    """Official test labels align to eligible final-window test engines only."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file_by_cycles(raw_dir / "test_FD001.txt", {1: 2, 2: 5})
    (raw_dir / "RUL_FD001.txt").write_text("11\n22\n")

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path)

    _run_cli(cfg_path)

    run_dir = next((artifact_dir / "sequence_gru").iterdir())
    with (run_dir / "official_test_predictions.csv").open(newline="") as csv_file:
        rows = list(DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["engine_id"] == "2"
    assert rows[0]["rul"] == "22.0"

    metrics = json.loads((run_dir / "metrics.json").read_text())
    _assert_metric_keys(metrics, "validation_windows")
    _assert_metric_keys(metrics, "official_test")


def test_train_sequence_gru_cli_skips_missing_official_test(
    tmp_path: Path,
) -> None:
    """CLI skips official evaluation when test or RUL files are absent."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path)

    result = _run_cli(cfg_path)

    assert "validation_windows rmse" in result.stdout
    assert "official test evaluation skipped" in result.stdout
    run_dir = next((artifact_dir / "sequence_gru").iterdir())
    assert not (run_dir / "official_test_predictions.csv").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"validation_windows"}
    _assert_metric_keys(metrics, "validation_windows")
