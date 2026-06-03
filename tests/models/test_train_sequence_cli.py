"""Smoke tests for the generalized turbofan.cli.train_sequence entrypoint."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pandas as pd
import pytest
import torch

from turbofan.config.schema import DataConfig, ProjectConfig, SequenceConfig
from turbofan.models.sequence_models import build_sequence_model
from turbofan.models.sequence_training import TrainingResult


class _CliResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


class _FakePipeline:
    """Minimal pipeline test double returning input unchanged."""

    def __init__(
        self, feature_cols: list[str] | None = None, **kwargs: object
    ) -> None:
        """Initialize with optional feature_cols list.

        Args:
            feature_cols: Columns the pipeline exposes via feature_engineer.
            **kwargs: Absorbed for compatibility with build_feature_pipeline.
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


def _write_config(
    path: Path,
    raw_dir: Path,
    artifact_dir: Path,
    tmp_path: Path,
    architecture: str,
) -> None:
    """Write a tiny sequence training config for a given architecture.

    Args:
        path: Destination YAML path.
        raw_dir: Raw C-MAPSS data directory.
        artifact_dir: Artifact root.
        tmp_path: Temporary test directory for other configured paths.
        architecture: Sequence architecture (``gru`` or ``lstm``).
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
                f"  architecture: {architecture}",
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


def _load_module() -> ModuleType:
    """Load the generalized sequence training CLI module.

    Returns:
        Imported CLI module.
    """
    from turbofan.cli import train_sequence

    return train_sequence


def test_train_sequence_cli_trains_and_registers_lstm(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A full LSTM run writes run records and registers a turbofan-lstm model."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from turbofan import registry, tracking

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=2, n_cycles=5)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n")

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path, architecture="lstm")

    module = _load_module()
    import sys

    monkeypatch_argv = ["turbofan.cli.train_sequence", "--config", str(cfg_path)]
    old_argv = sys.argv
    sys.argv = monkeypatch_argv
    try:
        module.main()
    finally:
        sys.argv = old_argv
    out = capsys.readouterr().out

    assert "validation_windows rmse" in out

    tracking.configure_mlflow()
    runs = mlflow.search_runs(experiment_names=[tracking.TRAINING_EXPERIMENT])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.model_type"] == "lstm"
    run_id = runs.iloc[0]["run_id"]

    client = MlflowClient()
    name = registry.model_name("lstm", "FD001")
    assert name == "turbofan-lstm-fd001"
    versions = client.search_model_versions(f"name = '{name}'")
    assert any(version.run_id == run_id for version in versions)


def test_train_sequence_cli_constructs_model_via_registry_architecture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI builds the model for the configured architecture and registers it.

    Confirms the architecture flows from config through model construction to the
    ``model_type`` used for registration — an LSTM config registers an ``lstm``
    model, not a hardwired ``gru``.
    """
    module = _load_module()
    cfg = ProjectConfig(
        project_name="test",
        data=DataConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
            fd_subset="FD001",
            random_seed=7,
        ),
        sequence=SequenceConfig(
            architecture="lstm",
            window_size=3,
            batch_size=4,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            epochs=1,
            artifact_dir=tmp_path / "artifacts",
        ),
    )
    captured_model_types: list[str] = []
    built_architectures: list[str] = []

    def fake_build_sequence_model(
        architecture: str, **kwargs: object
    ) -> torch.nn.Module:
        built_architectures.append(architecture)
        return build_sequence_model(architecture, **kwargs)  # type: ignore[arg-type]

    def fake_train(**kwargs: object) -> TrainingResult:
        model = kwargs["model"]
        assert isinstance(model, torch.nn.Module)
        return TrainingResult(
            model=model,
            history=pd.DataFrame([{"epoch": 1}]),
            best_epoch=1,
            best_metric=0.0,
        )

    def fake_log_and_register(
        payload: object, *, model_type: str, subset: str
    ) -> int:
        captured_model_types.append(model_type)
        return 1

    _fake_df = pd.DataFrame(
        {"engine_id": [1, 1, 1], "cycle": [1, 2, 3], "rul": [3, 2, 1]}
    )
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: argparse.Namespace(config=tmp_path / "c.yaml", log_level="INFO"),
    )
    monkeypatch.setattr(module, "load_config", lambda p: cfg)
    monkeypatch.setattr(module, "resolve_device", lambda r: torch.device("cpu"))
    monkeypatch.setattr(
        module, "build_feature_pipeline", lambda **kw: _FakePipeline(["s1", "s2"])
    )
    monkeypatch.setattr(module, "load_raw_train", lambda c: _fake_df)
    monkeypatch.setattr(module, "add_rul_column", lambda f, max_rul: f)
    monkeypatch.setattr(
        module, "split_by_engine", lambda f, test_size, random_seed: (f, f)
    )
    monkeypatch.setattr(module, "build_sliding_windows", lambda *a, **k: object())
    monkeypatch.setattr(module, "build_sequence_loader", lambda *a, **k: object())
    monkeypatch.setattr(module, "build_sequence_model", fake_build_sequence_model)
    monkeypatch.setattr(module, "train_sequence_model", fake_train)
    monkeypatch.setattr(
        module,
        "_evaluate_windows",
        lambda *a, **k: ({"rmse": 0.0, "mae": 0.0}, pd.DataFrame()),
    )
    monkeypatch.setattr(module, "_evaluate_official_test", lambda *a, **k: None)
    monkeypatch.setattr(module, "create_run_dir", lambda a, n: tmp_path)
    monkeypatch.setattr(module, "save_json", lambda p, pa: None)
    monkeypatch.setattr(module, "save_predictions", lambda f, p: None)
    monkeypatch.setattr(module.torch, "save", lambda p, pa: None)
    monkeypatch.setattr(module, "_model_payload", lambda *a, **k: {})
    monkeypatch.setattr(module.registry, "log_and_register", fake_log_and_register)
    monkeypatch.setattr(module.mlflow, "log_artifact", lambda *a, **k: None)

    module.main()

    assert built_architectures == ["lstm"]
    assert captured_model_types == ["lstm"]


def test_train_sequence_gru_alias_main_is_same_callable() -> None:
    """The backward-compatible GRU entrypoint reuses the generalized main."""
    from turbofan.cli.train_sequence import main as sequence_main
    from turbofan.cli.train_sequence_gru import main as gru_main

    assert gru_main is sequence_main
