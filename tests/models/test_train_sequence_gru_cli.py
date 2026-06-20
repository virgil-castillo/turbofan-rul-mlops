"""Smoke tests for turbofan.cli.train_sequence_gru."""
from __future__ import annotations

import argparse
import json
from csv import DictReader
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pandas as pd
import pytest
import torch

from turbofan.cli.train_sequence_gru import main as gru_main
from turbofan.config import schema
from turbofan.config.schema import DataConfig, ProjectConfig, SequenceConfig
from turbofan.data import loader
from turbofan.features import pipeline as feature_pipeline
from turbofan.models import (
    artifacts,
    evaluate,
    sequence_training,
    split,
)
from turbofan.models.gru import GRURULRegressor
from turbofan.models.sequence_training import TrainingResult
from turbofan.sequences import dataset, windowing


class _CliResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


class _FakePipeline:
    """Minimal pipeline test double returning input unchanged."""

    def __init__(self, feature_cols: list[str] | None = None, **kwargs: object) -> None:
        """Initialize with optional feature_cols list.

        Args:
            feature_cols: Columns the pipeline exposes via feature_engineer.
            **kwargs: Absorbed for compatibility with build_feature_pipeline kwargs.
        """
        _fc = feature_cols or ["s_1", "s_2"]
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

    Validation sections carry only RMSE and MAE; the PHM08 score is reserved
    for the official test section.

    Args:
        metrics: Metrics payload loaded from JSON.
        section: Top-level metric section name.
    """
    expected = {"rmse", "mae"}
    if section == "official_test":
        expected = {"rmse", "mae", "phm08_score"}
    assert set(metrics[section]) == expected


def _run_cli(
    cfg_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> _CliResult:
    """Run the sequence GRU CLI in-process and return captured output.

    Args:
        cfg_path: YAML config path.
        monkeypatch: pytest monkeypatch fixture.
        capsys: pytest capsys fixture for stdout/stderr capture.

    Returns:
        CLI result with returncode, stdout, and stderr.
    """
    returncode = gru_main(["--config", str(cfg_path)])
    captured = capsys.readouterr()
    return _CliResult(returncode=returncode, stdout=captured.out, stderr=captured.err)


def _load_train_sequence_gru_module() -> ModuleType:
    """Load the generalized sequence training CLI module that ``main`` lives in.

    The ``turbofan-train-sequence-gru`` console script now aliases the
    generalized :mod:`turbofan.cli.train_sequence` entrypoint, so the
    monkeypatch targets resolve against that module.

    Returns:
        Imported CLI module.
    """
    from turbofan.cli import train_sequence

    return train_sequence


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
        lambda argv=None: argparse.Namespace(
            config=tmp_path / "config.yaml", log_level="INFO"
        ),
    )
    monkeypatch.setattr(schema, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        sequence_training,
        "resolve_device",
        lambda requested: torch.device("cpu"),
    )
    _fake_df = pd.DataFrame(
        {"engine_id": [1, 1, 1], "cycle": [1, 2, 3], "rul": [3, 2, 1]}
    )
    monkeypatch.setattr(
        feature_pipeline,
        "build_feature_pipeline",
        lambda **kw: _FakePipeline(["s1", "s2"]),
    )
    monkeypatch.setattr(loader, "load_raw_train", lambda data_config: _fake_df)
    monkeypatch.setattr(evaluate, "add_rul_column", lambda frame, max_rul: frame)
    monkeypatch.setattr(
        split,
        "split_by_engine",
        lambda frame, test_size, random_seed: (frame, frame),
    )
    monkeypatch.setattr(
        windowing,
        "build_sliding_windows",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        dataset,
        "build_sequence_loader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(sequence_training, "train_sequence_model", fake_train_gru_model)
    monkeypatch.setattr(
        module,
        "_evaluate_windows",
        lambda *args, **kwargs: (
            {"rmse": 0.0, "mae": 0.0},
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(module, "_evaluate_official_test", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        artifacts, "create_run_dir", lambda artifact_dir, name: tmp_path
    )
    monkeypatch.setattr(artifacts, "save_json", lambda payload, path: None)
    monkeypatch.setattr(artifacts, "save_predictions", lambda frame, path: None)
    monkeypatch.setattr(module.torch, "save", fake_torch_save)
    monkeypatch.setattr(module, "_model_payload", lambda *a, **k: {})
    # Disk artifacts are stubbed out above, so skip registry/artifact logging
    # (exercised by the dedicated registration test and tests/test_registry.py).
    monkeypatch.setattr(
        module.registry, "log_and_register", lambda *a, **k: 1
    )
    monkeypatch.setattr(module.mlflow, "log_artifact", lambda *a, **k: None)

    torch.manual_seed(999)
    assert module.main() == 0

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


def test_train_sequence_gru_cli_logs_mlflow_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI logs one production GRU MLflow run with params, metrics, and tags."""
    import mlflow

    from turbofan import tracking

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
    window_metrics = {"rmse": 1.0, "mae": 2.0}

    def fake_train_gru_model(**kwargs: object) -> TrainingResult:
        del kwargs
        return TrainingResult(
            model=GRURULRegressor(
                input_size=2,
                hidden_size=6,
                num_layers=2,
                dropout=0.1,
            ),
            history=pd.DataFrame(
                [
                    {"epoch": 1, "train_loss": 5.0, "validation_windows_rmse": 4.0},
                    {"epoch": 2, "train_loss": 3.0, "validation_windows_rmse": 2.0},
                ]
            ),
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

    def fake_create_run_dir(artifact_dir: Path, name: str) -> Path:
        del artifact_dir
        del name
        run_dir.mkdir()
        return run_dir

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda argv=None: argparse.Namespace(
            config=tmp_path / "config.yaml", log_level="INFO"
        ),
    )
    _fake_df = pd.DataFrame(
        {"engine_id": [1, 1, 1], "cycle": [1, 2, 3], "rul": [3, 2, 1]}
    )
    monkeypatch.setattr(schema, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        sequence_training, "resolve_device", lambda requested: torch.device("cpu")
    )
    monkeypatch.setattr(
        feature_pipeline,
        "build_feature_pipeline",
        lambda **kw: _FakePipeline(["s1", "s2"]),
    )
    monkeypatch.setattr(loader, "load_raw_train", lambda data_config: _fake_df)
    monkeypatch.setattr(evaluate, "add_rul_column", lambda frame, max_rul: frame)
    monkeypatch.setattr(
        split,
        "split_by_engine",
        lambda frame, test_size, random_seed: (frame, frame),
    )
    monkeypatch.setattr(
        windowing,
        "build_sliding_windows",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        dataset,
        "build_sequence_loader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(sequence_training, "train_sequence_model", fake_train_gru_model)
    monkeypatch.setattr(module, "_evaluate_windows", fake_evaluate_windows)
    monkeypatch.setattr(module, "_evaluate_official_test", lambda *args, **kwargs: None)
    monkeypatch.setattr(artifacts, "create_run_dir", fake_create_run_dir)
    monkeypatch.setattr(artifacts, "save_json", lambda payload, path: None)
    monkeypatch.setattr(artifacts, "save_predictions", lambda frame, path: None)
    monkeypatch.setattr(module.torch, "save", lambda payload, path: None)
    monkeypatch.setattr(module, "_model_payload", lambda *a, **k: {})
    # Disk artifacts are stubbed out above, so skip registry/artifact logging
    # (exercised by the dedicated registration test and tests/test_registry.py).
    monkeypatch.setattr(
        module.registry, "log_and_register", lambda *a, **k: 1
    )
    monkeypatch.setattr(module.mlflow, "log_artifact", lambda *a, **k: None)

    assert module.main() == 0

    tracking.configure_mlflow()
    runs = mlflow.search_runs(experiment_names=[tracking.TRAINING_EXPERIMENT])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["tags.model_type"] == "gru"
    assert row["tags.run_type"] == "production"
    assert row["tags.best_epoch"] == "7"
    assert row["tags.run_dir"] == str(run_dir)
    assert row["params.window_size"] == "5"
    assert row["params.hidden_size"] == "6"
    assert row["params.learning_rate"] == "0.002"
    assert row["params.seed"] == "123"
    assert row["params.feature_families"] == "['raw']"
    assert row["metrics.val_rmse"] == 1.0
    assert row["metrics.val_mae"] == 2.0
    assert row["metrics.training_duration_seconds"] >= 0.0


def test_train_sequence_gru_cli_writes_artifacts_registers_and_logs_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One CLI run: disk run records, a registered checkpoint, and run artifacts.

    Consolidates the run-dir artifact, model-registration, prediction-artifact,
    and checkpoint-payload facets into a single training run instead of spawning
    a separate ~12s training subprocess per facet, all of which exercised the
    same standard-data run.
    """
    import mlflow
    import torch as _torch
    from mlflow.tracking import MlflowClient

    from turbofan import registry, tracking

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=2, n_cycles=5)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n")

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path)

    result = _run_cli(cfg_path, monkeypatch, capsys)

    # --- run-dir records (model bytes + manifest retired; MLflow is the store) ---
    assert "validation_windows rmse" in result.stdout
    run_dirs = list((artifact_dir / "sequence_gru").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert not (run_dir / "model.pt").exists()
    assert not (run_dir / "model_manifest.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "training_history.csv").exists()
    assert (run_dir / "validation_window_predictions.csv").exists()
    assert (run_dir / "official_test_predictions.csv").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"validation_windows", "official_test"}
    _assert_metric_keys(metrics, "validation_windows")
    _assert_metric_keys(metrics, "official_test")

    # --- registered model version linked to the run + prediction artifacts ---
    tracking.configure_mlflow()
    runs = mlflow.search_runs(experiment_names=[tracking.TRAINING_EXPERIMENT])
    assert len(runs) == 1
    run_id = runs.iloc[0]["run_id"]

    client = MlflowClient()
    name = registry.model_name("gru", "FD001")
    assert name == "turbofan-gru-fd001"
    versions = client.search_model_versions(f"name = '{name}'")
    assert len(versions) >= 1
    assert any(version.run_id == run_id for version in versions)

    artifact_paths = {
        artifact.path for artifact in client.list_artifacts(run_id, "predictions")
    }
    assert "predictions/validation_window_predictions.csv" in artifact_paths
    assert "predictions/official_test_predictions.csv" in artifact_paths

    # --- the registered checkpoint carries the operating-mode normalizer payload ---
    local_dir = Path(mlflow.artifacts.download_artifacts(f"models:/{name}/1"))
    checkpoint = next(local_dir.rglob("model.pt"))
    payload = _torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["normalizer_type"] == "operating_mode"
    assert "feature_pipeline" in payload
    assert "normalizer_payload" in payload
    assert payload["normalizer_payload"]["schema_version"] == 1
    assert "normalizer_means" not in payload
    assert "normalizer_stds" not in payload


def test_train_sequence_gru_cli_aligns_official_labels_to_all_test_engines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Official test labels align to all test engines, including short padded ones."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file_by_cycles(raw_dir / "test_FD001.txt", {1: 2, 2: 5})
    (raw_dir / "RUL_FD001.txt").write_text("11\n22\n")

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path)

    _run_cli(cfg_path, monkeypatch, capsys)

    run_dir = next((artifact_dir / "sequence_gru").iterdir())
    with (run_dir / "official_test_predictions.csv").open(newline="") as csv_file:
        rows = list(DictReader(csv_file))

    # Both engines are now included: engine 1 is left-zero-padded (2 cycles < window 3)
    assert len(rows) == 2
    engine_ids = {row["engine_id"] for row in rows}
    assert engine_ids == {"1", "2"}
    rul_by_engine = {row["engine_id"]: row["rul"] for row in rows}
    assert rul_by_engine["1"] == "11.0"
    assert rul_by_engine["2"] == "22.0"

    metrics = json.loads((run_dir / "metrics.json").read_text())
    _assert_metric_keys(metrics, "validation_windows")
    _assert_metric_keys(metrics, "official_test")


def test_train_sequence_gru_cli_skips_missing_official_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI skips official evaluation when test or RUL files are absent."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path)

    result = _run_cli(cfg_path, monkeypatch, capsys)

    assert "validation_windows rmse" in result.stdout
    assert "official test evaluation skipped" in result.stderr
    run_dir = next((artifact_dir / "sequence_gru").iterdir())
    assert not (run_dir / "official_test_predictions.csv").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"validation_windows"}
    _assert_metric_keys(metrics, "validation_windows")


def test_train_sequence_gru_cli_uses_subset_derived_mode_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GRU training calls build_feature_pipeline with subset mode count."""
    module = _load_train_sequence_gru_module()
    captured: list[dict[str, object]] = []

    def _capturing_pipeline(**kwargs: object) -> _FakePipeline:
        captured.append(kwargs)
        return _FakePipeline(["s1", "s2"])

    cfg = ProjectConfig(
        project_name="test",
        data=DataConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
            fd_subset="FD001",
            random_seed=77,
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

    def fake_train(**kwargs: object) -> TrainingResult:
        return TrainingResult(
            model=GRURULRegressor(
                input_size=2, hidden_size=4, num_layers=1, dropout=0.0
            ),
            history=pd.DataFrame([{"epoch": 1}]),
            best_epoch=1,
            best_metric=0.0,
        )

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda argv=None: argparse.Namespace(
            config=tmp_path / "c.yaml", log_level="INFO"
        ),
    )
    monkeypatch.setattr(schema, "load_config", lambda p: cfg)
    monkeypatch.setattr(
        sequence_training, "resolve_device", lambda r: torch.device("cpu")
    )
    _fake_df = pd.DataFrame(
        {"engine_id": [1, 1, 1], "cycle": [1, 2, 3], "rul": [3, 2, 1]}
    )
    monkeypatch.setattr(feature_pipeline, "build_feature_pipeline", _capturing_pipeline)
    monkeypatch.setattr(loader, "load_raw_train", lambda c: _fake_df)
    monkeypatch.setattr(evaluate, "add_rul_column", lambda f, max_rul: f)
    monkeypatch.setattr(
        split,
        "split_by_engine",
        lambda f, test_size, random_seed: (f, f),
    )
    monkeypatch.setattr(
        windowing, "build_sliding_windows", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        dataset, "build_sequence_loader", lambda *a, **k: object()
    )
    monkeypatch.setattr(sequence_training, "train_sequence_model", fake_train)
    monkeypatch.setattr(
        module,
        "_evaluate_windows",
        lambda *a, **k: (
            {"rmse": 0.0, "mae": 0.0},
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        module, "_evaluate_official_test", lambda *a, **k: None
    )
    monkeypatch.setattr(artifacts, "create_run_dir", lambda a, n: tmp_path)
    monkeypatch.setattr(artifacts, "save_json", lambda p, pa: None)
    monkeypatch.setattr(artifacts, "save_predictions", lambda f, p: None)
    monkeypatch.setattr(module.torch, "save", lambda p, pa: None)
    monkeypatch.setattr(module, "_model_payload", lambda *a, **k: {})
    # Disk artifacts are stubbed out above, so skip registry/artifact logging
    # (exercised by the dedicated registration test and tests/test_registry.py).
    monkeypatch.setattr(
        module.registry, "log_and_register", lambda *a, **k: 1
    )
    monkeypatch.setattr(module.mlflow, "log_artifact", lambda *a, **k: None)

    assert module.main() == 0

    assert captured, "build_feature_pipeline was never called"
    assert captured[0]["n_modes"] == 1  # FD001 → 1 mode
    assert captured[0]["random_state"] == 77
