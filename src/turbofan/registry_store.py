"""Lookup, loading, and listing helpers for MLflow registered models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from turbofan.inference.pyfunc_adapter import PyfuncPredictor
from turbofan.inference.schemas import ModelType

_VAL_RMSE_METRIC = "val_rmse"


def model_name(model_type: str, subset: str) -> str:
    """Build the canonical registered-model name for a model family and subset.

    Args:
        model_type: Model family identifier, e.g. ``"ridge"`` or ``"gru"``.
        subset: C-MAPSS subset identifier, e.g. ``"FD001"``.

    Returns:
        Canonical registered-model name.
    """
    return f"turbofan-{model_type}-{subset.lower()}"


def latest_version(name: str) -> int:
    """Return the highest registered version number for a model.

    Args:
        name: Registered-model name.

    Returns:
        The maximum version number registered under ``name``.

    Raises:
        ValueError: If no versions are registered under ``name``.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{name}'")
    if not versions:
        raise ValueError(f"No registered versions found for {name!r}.")
    return max(int(version.version) for version in versions)


def promote(name: str, version: int, alias: str = "production") -> None:
    """Point a registered-model alias at a specific version.

    Args:
        name: Registered-model name.
        version: Model version to alias.
        alias: Alias to repoint. Defaults to ``"production"``.

    Returns:
        None.
    """
    MlflowClient().set_registered_model_alias(name, alias, str(version))


def resolve_uri(name: str, alias: str = "production") -> str:
    """Build the MLflow ``models:`` URI for an aliased registered model.

    Args:
        name: Registered-model name.
        alias: Alias to resolve. Defaults to ``"production"``.

    Returns:
        Model URI of the form ``models:/<name>@<alias>``.
    """
    return f"models:/{name}@{alias}"


def load(name: str, alias: str = "production") -> PyFuncModel:
    """Load the aliased registered model as a pyfunc model for inference.

    Args:
        name: Registered-model name.
        alias: Alias to load. Defaults to ``"production"``.

    Returns:
        Loaded pyfunc model resolved from ``models:/<name>@<alias>``.
    """
    return cast(PyFuncModel, mlflow.pyfunc.load_model(resolve_uri(name, alias)))


def model_type_from_name(name: str) -> ModelType:
    """Infer the model family from a canonical registered-model name.

    Args:
        name: Registered-model name of the form ``turbofan-{type}-{subset}``.

    Returns:
        The model family.

    Raises:
        ValueError: If the name does not encode a supported model family.
    """
    parts = name.split("-")
    if len(parts) >= 2 and parts[0] == "turbofan":
        candidate = parts[1]
        if candidate in ("ridge", "gru", "lstm"):
            return cast(ModelType, candidate)
    raise ValueError(
        f"Cannot infer model type from registered-model name {name!r}; "
        "expected the form 'turbofan-{ridge,gru,lstm}-{subset}'."
    )


def load_predictor(name: str, alias: str = "production") -> PyfuncPredictor:
    """Load an aliased registry model wrapped in the predictor adapter.

    Args:
        name: Registered-model name.
        alias: Alias to load. Defaults to ``"production"``.

    Returns:
        A predictor exposing the inference predictor contract.

    Raises:
        ValueError: If the model family cannot be inferred from ``name``.
    """
    model = load(name, alias)
    model_type = model_type_from_name(name)
    version = MlflowClient().get_model_version_by_alias(name, alias).version
    artifact_id = f"{name}/{version}"
    return PyfuncPredictor(model, model_type=model_type, artifact_id=artifact_id)


def load_predictor_from_uri(uri: str) -> PyfuncPredictor:
    """Load a ``models:`` URI model wrapped in the predictor adapter.

    Supports both alias URIs and version URIs.

    Args:
        uri: Model URI of the form ``models:/<name>@<alias>`` or
            ``models:/<name>/<version>``.

    Returns:
        A predictor exposing the inference predictor contract.

    Raises:
        ValueError: If the URI is not supported or model type inference fails.
    """
    name, reference = parse_models_uri(uri)
    model = cast(PyFuncModel, mlflow.pyfunc.load_model(uri))
    model_type = model_type_from_name(name)
    artifact_id = f"{name}/{reference}"
    return PyfuncPredictor(model, model_type=model_type, artifact_id=artifact_id)


def parse_models_uri(uri: str) -> tuple[str, str]:
    """Parse a ``models:`` URI into its name and version or alias reference.

    Args:
        uri: Model URI of the form ``models:/<name>@<alias>`` or
            ``models:/<name>/<version>``.

    Returns:
        Tuple of registered-model name and version or alias reference.

    Raises:
        ValueError: If the URI is not a ``models:`` URI.
    """
    prefix = "models:/"
    if not uri.startswith(prefix):
        raise ValueError(f"Expected a 'models:/' URI, got {uri!r}.")
    remainder = uri[len(prefix) :]
    if "@" in remainder:
        name, alias = remainder.split("@", 1)
        return name, alias
    name, _, version = remainder.rpartition("/")
    if not name:
        raise ValueError(f"Malformed 'models:/' URI: {uri!r}.")
    return name, version


@dataclass(frozen=True)
class RegisteredModelInfo:
    """Listing record for one registered model.

    Args:
        name: Registered-model name.
        versions: All registered version numbers, ascending.
        production_version: Version currently aliased ``@production``.
        val_rmse: Validation RMSE metric from the production run, if available.
        run_id: MLflow run id linked to the production version, if available.
    """

    name: str
    versions: list[int]
    production_version: int | None
    val_rmse: float | None
    run_id: str | None


def list_registered() -> list[RegisteredModelInfo]:
    """List registered models with version, production alias, and provenance.

    Returns:
        One listing record per registered model, ordered by name.
    """
    client = MlflowClient()
    infos: list[RegisteredModelInfo] = []
    for registered in client.search_registered_models():
        name = registered.name
        versions = sorted(
            int(version.version)
            for version in client.search_model_versions(f"name = '{name}'")
        )
        production_version: int | None = None
        val_rmse: float | None = None
        run_id: str | None = None
        production = _production_version(client, name)
        if production is not None:
            production_version = int(production.version)
            run_id = production.run_id or None
            val_rmse = _run_val_rmse(client, run_id)
        infos.append(
            RegisteredModelInfo(
                name=name,
                versions=versions,
                production_version=production_version,
                val_rmse=val_rmse,
                run_id=run_id,
            )
        )
    return sorted(infos, key=lambda info: info.name)


def _production_version(client: MlflowClient, name: str) -> ModelVersion | None:
    """Return the ``@production`` model version for a model, or ``None``."""
    try:
        return client.get_model_version_by_alias(name, "production")
    except mlflow.exceptions.MlflowException:
        return None


def _run_val_rmse(client: MlflowClient, run_id: str | None) -> float | None:
    """Read the validation RMSE metric from a run, or ``None`` if unavailable."""
    if not run_id:
        return None
    try:
        run = client.get_run(run_id)
    except mlflow.exceptions.MlflowException:
        return None
    value = run.data.metrics.get(_VAL_RMSE_METRIC)
    return float(value) if value is not None else None
