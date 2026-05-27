"""Tests for turbofan.inference.manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofan.inference.manifest import ManifestError, load_model_metadata


def _write_manifest(
    directory: Path,
    payload: dict[str, object],
) -> Path:
    """Write a model manifest file.

    Args:
        directory: Directory that receives the manifest.
        payload: Manifest payload.

    Returns:
        Path to the manifest file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model_manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_model_metadata_reads_valid_ridge_manifest(tmp_path: Path) -> None:
    """A schema-version-1 Ridge manifest is loaded as engine metadata."""
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "model_type": "ridge",
            "artifact_id": "ridge-001",
            "prediction_scope": "engine",
            "model_path": "model.joblib",
        },
    )

    metadata = load_model_metadata(manifest_path)

    assert metadata.model_type == "ridge"
    assert metadata.artifact_id == "ridge-001"
    assert metadata.prediction_scope == "engine"
    assert metadata.model_path == model_path
    assert metadata.config_path is None
    assert metadata.metrics_path is None


def test_load_model_metadata_reads_valid_gru_manifest(tmp_path: Path) -> None:
    """A schema-version-1 GRU manifest is loaded as final-window metadata."""
    model_path = tmp_path / "model.pt"
    config_path = tmp_path / "config.yaml"
    metrics_path = tmp_path / "metrics.json"
    model_path.write_bytes(b"model")
    config_path.write_text("model: gru")
    metrics_path.write_text("{}")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "model_type": "gru",
            "artifact_id": "gru-001",
            "prediction_scope": "final_window",
            "model_path": "model.pt",
            "config_path": "config.yaml",
            "metrics_path": "metrics.json",
        },
    )

    metadata = load_model_metadata(manifest_path)

    assert metadata.model_type == "gru"
    assert metadata.artifact_id == "gru-001"
    assert metadata.prediction_scope == "final_window"
    assert metadata.model_path == model_path
    assert metadata.config_path == config_path
    assert metadata.metrics_path == metrics_path


@pytest.mark.parametrize(
    ("model_type", "prediction_scope"),
    [("ridge", "final_window"), ("gru", "row")],
)
def test_load_model_metadata_rejects_inconsistent_model_scope_pairs(
    tmp_path: Path,
    model_type: str,
    prediction_scope: str,
) -> None:
    """Manifest model type and prediction scope must be a supported pair."""
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"model")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "model_type": model_type,
            "artifact_id": "bad-pair",
            "prediction_scope": prediction_scope,
            "model_path": "model.bin",
        },
    )

    with pytest.raises(ManifestError, match="inconsistent"):
        load_model_metadata(manifest_path)


def test_load_model_metadata_resolves_relative_paths_from_manifest_directory(
    tmp_path: Path,
) -> None:
    """Relative manifest paths are resolved against the manifest directory."""
    run_dir = tmp_path / "runs" / "ridge"
    (run_dir / "artifacts").mkdir(parents=True)
    model_path = run_dir / "artifacts" / "model.joblib"
    model_path.write_bytes(b"model")
    manifest_path = _write_manifest(
        run_dir,
        {
            "schema_version": 1,
            "model_type": "ridge",
            "artifact_id": "ridge-002",
            "prediction_scope": "engine",
            "model_path": "artifacts/model.joblib",
        },
    )

    metadata = load_model_metadata(manifest_path)

    assert metadata.model_path == model_path


def test_load_model_metadata_supports_directory_with_only_joblib(
    tmp_path: Path,
) -> None:
    """A directory containing only model.joblib loads as Ridge engine metadata."""
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")

    metadata = load_model_metadata(tmp_path)

    assert metadata.model_type == "ridge"
    assert metadata.artifact_id == tmp_path.name
    assert metadata.prediction_scope == "engine"
    assert metadata.model_path == model_path


def test_load_model_metadata_supports_directory_with_only_pt(tmp_path: Path) -> None:
    """A directory containing only model.pt loads as GRU final-window metadata."""
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")

    metadata = load_model_metadata(tmp_path)

    assert metadata.model_type == "gru"
    assert metadata.artifact_id == tmp_path.name
    assert metadata.prediction_scope == "final_window"
    assert metadata.model_path == model_path


def test_load_model_metadata_rejects_ambiguous_compatibility_artifacts(
    tmp_path: Path,
) -> None:
    """Compatibility loading fails when Ridge and GRU artifacts are present."""
    (tmp_path / "model.joblib").write_bytes(b"model")
    (tmp_path / "model.pt").write_bytes(b"model")

    with pytest.raises(ManifestError, match="ambiguous"):
        load_model_metadata(tmp_path)


def test_load_model_metadata_rejects_missing_compatibility_artifacts(
    tmp_path: Path,
) -> None:
    """Compatibility loading fails when no known model artifact is present."""
    with pytest.raises(ManifestError, match="missing"):
        load_model_metadata(tmp_path)


def test_load_model_metadata_rejects_missing_manifest_file(tmp_path: Path) -> None:
    """Loading an explicit missing manifest path fails clearly."""
    with pytest.raises(ManifestError, match="does not exist"):
        load_model_metadata(tmp_path / "model_manifest.json")
