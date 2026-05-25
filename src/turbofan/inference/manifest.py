"""Model manifest loading for local inference artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from turbofan.inference.schemas import ModelType, PredictionScope

MANIFEST_FILENAME = "model_manifest.json"


class ManifestError(ValueError):
    """Raised when a model artifact manifest cannot be loaded."""


@dataclass(frozen=True)
class ModelMetadata:
    """Runtime model artifact metadata.

    Args:
        schema_version: Manifest schema version.
        model_type: Model family.
        artifact_id: Stable model artifact identifier.
        prediction_scope: Scope supported by the model.
        model_path: Resolved model artifact path.
        config_path: Optional resolved training/configuration path.
        metrics_path: Optional resolved metrics path.
    """

    schema_version: int
    model_type: ModelType
    artifact_id: str
    prediction_scope: PredictionScope
    model_path: Path
    config_path: Path | None = None
    metrics_path: Path | None = None


def load_model_metadata(path: Path) -> ModelMetadata:
    """Load model metadata from a manifest file or compatibility directory.

    Args:
        path: Path to a model_manifest.json file or artifact directory.

    Returns:
        Validated model metadata with paths resolved to the artifact directory.

    Raises:
        ManifestError: If metadata cannot be resolved unambiguously.
    """
    if path.is_dir():
        manifest_path = path / MANIFEST_FILENAME
        if manifest_path.exists():
            return _load_manifest_file(manifest_path)
        return _load_compatibility_directory(path)

    if not path.exists():
        raise ManifestError(f"Manifest path does not exist: {path}")
    return _load_manifest_file(path)


def _load_manifest_file(path: Path) -> ModelMetadata:
    """Load and validate an explicit JSON manifest file.

    Args:
        path: Manifest file path.

    Returns:
        Validated model metadata.

    Raises:
        ManifestError: If the manifest is invalid.
    """
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("Manifest must be a JSON object.")
    return _metadata_from_payload(payload, path.parent)


def _metadata_from_payload(
    payload: dict[str, object],
    manifest_dir: Path,
) -> ModelMetadata:
    """Build model metadata from a manifest payload.

    Args:
        payload: Decoded JSON object.
        manifest_dir: Directory containing the manifest file.

    Returns:
        Validated model metadata.

    Raises:
        ManifestError: If required fields are missing or invalid.
    """
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ManifestError("Manifest schema_version must be 1.")

    model_type = _require_model_type(payload)
    prediction_scope = _require_prediction_scope(payload)
    _validate_model_scope_pair(model_type, prediction_scope)
    artifact_id = _require_string(payload, "artifact_id")
    model_path = _resolve_manifest_path(
        manifest_dir,
        _require_string(payload, "model_path"),
    )
    config_path = _resolve_optional_manifest_path(manifest_dir, payload, "config_path")
    metrics_path = _resolve_optional_manifest_path(
        manifest_dir,
        payload,
        "metrics_path",
    )

    return ModelMetadata(
        schema_version=1,
        model_type=model_type,
        artifact_id=artifact_id,
        prediction_scope=prediction_scope,
        model_path=model_path,
        config_path=config_path,
        metrics_path=metrics_path,
    )


def _load_compatibility_directory(directory: Path) -> ModelMetadata:
    """Load metadata from legacy single-artifact directory conventions.

    Args:
        directory: Artifact directory.

    Returns:
        Compatibility metadata.

    Raises:
        ManifestError: If artifacts are ambiguous or missing.
    """
    joblib_path = directory / "model.joblib"
    pt_path = directory / "model.pt"
    has_joblib = joblib_path.exists()
    has_pt = pt_path.exists()

    if has_joblib and has_pt:
        raise ManifestError(
            "Compatibility artifact directory is ambiguous: both model.joblib "
            "and model.pt exist."
        )
    if not has_joblib and not has_pt:
        raise ManifestError(
            "Compatibility artifact directory is missing model.joblib or model.pt."
        )
    if has_joblib:
        return ModelMetadata(
            schema_version=1,
            model_type="ridge",
            artifact_id=directory.name,
            prediction_scope="row",
            model_path=joblib_path,
        )
    return ModelMetadata(
        schema_version=1,
        model_type="gru",
        artifact_id=directory.name,
        prediction_scope="final_window",
        model_path=pt_path,
    )


def _require_string(payload: dict[str, object], key: str) -> str:
    """Read a required string field.

    Args:
        payload: Manifest payload.
        key: Field name.

    Returns:
        Field value.

    Raises:
        ManifestError: If the field is missing or not a string.
    """
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"Manifest field {key!r} must be a non-empty string.")
    return value


def _require_model_type(payload: dict[str, object]) -> ModelType:
    """Read the required model type field.

    Args:
        payload: Manifest payload.

    Returns:
        Validated model type.

    Raises:
        ManifestError: If the model type is missing or unsupported.
    """
    value = _require_string(payload, "model_type")
    if value == "ridge":
        return "ridge"
    if value == "gru":
        return "gru"
    raise ManifestError("Manifest field 'model_type' must be one of: gru, ridge.")


def _require_prediction_scope(payload: dict[str, object]) -> PredictionScope:
    """Read the required prediction scope field.

    Args:
        payload: Manifest payload.

    Returns:
        Validated prediction scope.

    Raises:
        ManifestError: If the prediction scope is missing or unsupported.
    """
    value = _require_string(payload, "prediction_scope")
    if value == "row":
        return "row"
    if value == "final_window":
        return "final_window"
    raise ManifestError(
        "Manifest field 'prediction_scope' must be one of: final_window, row."
    )


def _validate_model_scope_pair(
    model_type: ModelType,
    prediction_scope: PredictionScope,
) -> None:
    """Validate that model type and prediction scope are compatible.

    Args:
        model_type: Validated model family.
        prediction_scope: Validated prediction scope.

    Raises:
        ManifestError: If the pair is not supported.
    """
    if (model_type, prediction_scope) in {
        ("ridge", "row"),
        ("gru", "final_window"),
    }:
        return
    raise ManifestError(
        "Manifest model_type and prediction_scope are inconsistent; expected "
        "ridge with row or gru with final_window."
    )


def _resolve_optional_manifest_path(
    manifest_dir: Path,
    payload: dict[str, object],
    key: str,
) -> Path | None:
    """Resolve an optional manifest path field.

    Args:
        manifest_dir: Directory containing the manifest.
        payload: Manifest payload.
        key: Optional path field name.

    Returns:
        Resolved path or None.

    Raises:
        ManifestError: If the optional path is not a string.
    """
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestError(f"Manifest field {key!r} must be a non-empty string.")
    return _resolve_manifest_path(manifest_dir, value)


def _resolve_manifest_path(manifest_dir: Path, raw_path: str) -> Path:
    """Resolve an absolute or manifest-relative path.

    Args:
        manifest_dir: Directory containing the manifest.
        raw_path: Raw path string from the manifest.

    Returns:
        Resolved path.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_dir / path
