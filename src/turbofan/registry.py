"""MLflow Model Registry wrapper for turbofan Ridge and GRU models.

This module is a thin seam over MLflow's registry, mirroring ``tracking.py``.
It packages the two model families into MLflow pyfunc models that honor the
existing inference contracts and exposes registration, promotion, resolution,
loading, and listing helpers over the local SQLite registry.

Model-packaging contract for both pyfunc wrappers:

- **Input:** a pandas ``DataFrame`` of canonical raw inference records with
  columns ``engine_id``, ``cycle``, ``op_1``..``op_3``, and ``s_1``..``s_21``.
- **Output:** a one-dimensional ``numpy`` ``float64`` array of non-negative RUL
  predictions, one per engine. The Ridge wrapper returns the last-cycle
  prediction per engine (``engine`` scope); the GRU wrapper returns the
  final-window prediction per engine (``final_window`` scope). Rows are ordered
  by ``engine_id``.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import ModelSignature, infer_signature
from mlflow.pyfunc import PyFuncModel
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from mlflow.tracking import MlflowClient

from turbofan.inference.predictors import (
    gru_final_window_predictions,
    ridge_engine_predictions,
)
from turbofan.inference.schemas import CANONICAL_COLUMNS, validate_raw_records

_RIDGE_ARTIFACT_KEY = "pipeline"
_GRU_ARTIFACT_KEY = "checkpoint"
_GRU_CHECKPOINT_FILENAME = "model.pt"
_VAL_RMSE_METRIC = "val_rmse"


def model_name(model_type: str, subset: str) -> str:
    """Build the canonical registered-model name for a model family and subset.

    Args:
        model_type: Model family identifier, e.g. ``"ridge"`` or ``"gru"``.
        subset: C-MAPSS subset identifier, e.g. ``"FD001"`` (case-insensitive).

    Returns:
        Canonical registered-model name, e.g. ``"turbofan-gru-fd001"``.
    """
    return f"turbofan-{model_type}-{subset.lower()}"


class RidgeEngineModel(PythonModel):
    """Pyfunc wrapper running the engine-scope Ridge inference contract.

    Loads a fitted sklearn ``Pipeline`` from the logged artifact and, on
    ``predict``, validates the raw records and returns the last-cycle
    prediction per engine, clipped to be non-negative. The selection and
    clipping logic is shared with ``inference.predictors.RidgePredictor`` via
    ``ridge_engine_predictions``.
    """

    def load_context(self, context: PythonModelContext) -> None:
        """Load the fitted Ridge pipeline from the model artifacts.

        Args:
            context: Pyfunc context exposing the logged artifact paths.

        Returns:
            None.
        """
        import joblib

        self._pipeline = joblib.load(context.artifacts[_RIDGE_ARTIFACT_KEY])

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Predict one non-negative RUL value per engine (last-cycle prediction).

        Args:
            context: Pyfunc context (unused; loading happens in load_context).
            model_input: Canonical raw inference records.
            params: Optional inference params (unused).

        Returns:
            One-dimensional array of per-engine predictions, ordered by
            ``engine_id``.
        """
        del context, params
        validation = validate_raw_records(model_input)
        _, predictions = ridge_engine_predictions(self._pipeline, validation.records)
        return predictions


class GRUFinalWindowModel(PythonModel):
    """Pyfunc wrapper running the final-window-scope GRU inference contract.

    Loads a GRU checkpoint payload (the same shape as the training ``model.pt``)
    from the logged artifact and, on ``predict``, validates the raw records and
    returns the final-window prediction per engine (normalize, window, forward,
    rescale by ``max_rul``, clip). The compute is shared with
    ``inference.predictors.GRUPredictor`` via ``gru_final_window_predictions``.
    """

    def load_context(self, context: PythonModelContext) -> None:
        """Load the GRU checkpoint payload from the model artifacts.

        Args:
            context: Pyfunc context exposing the logged artifact paths.

        Returns:
            None.
        """
        self._payload = torch.load(
            context.artifacts[_GRU_ARTIFACT_KEY], map_location="cpu"
        )

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Predict one non-negative RUL value per engine final window.

        Args:
            context: Pyfunc context (unused; loading happens in load_context).
            model_input: Canonical raw inference records, each engine at least
                ``window_size`` cycles long.
            params: Optional inference params (unused).

        Returns:
            One-dimensional array of per-engine final-window predictions, ordered
            by ``engine_id``.
        """
        del context, params
        validation = validate_raw_records(model_input)
        _, predictions = gru_final_window_predictions(
            self._payload, validation.records
        )
        return predictions


def _signature(predictions: npt.NDArray[np.float64]) -> ModelSignature:
    """Infer a pyfunc signature from canonical raw records and predictions.

    Args:
        predictions: Sample prediction array used to infer the output schema.

    Returns:
        Model signature mapping canonical raw records to a prediction array.
    """
    sample_input = pd.DataFrame(
        [[0.0] * len(CANONICAL_COLUMNS)],
        columns=CANONICAL_COLUMNS,
    )
    sample_input["engine_id"] = sample_input["engine_id"].astype("int64")
    sample_input["cycle"] = sample_input["cycle"].astype("int64")
    return infer_signature(sample_input, predictions)


def _log_ridge_model(model: object, name: str) -> None:
    """Log a fitted Ridge pipeline as an engine-scope pyfunc model.

    Args:
        model: Fitted sklearn ``Pipeline``.
        name: Registered-model name to register the logged model under.

    Returns:
        None.
    """
    import joblib

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / "pipeline.joblib"
        joblib.dump(model, artifact_path)
        signature = _signature(np.zeros(1, dtype=np.float64))
        mlflow.pyfunc.log_model(
            name="model",
            python_model=RidgeEngineModel(),
            artifacts={_RIDGE_ARTIFACT_KEY: str(artifact_path)},
            signature=signature,
            registered_model_name=name,
        )


def _log_gru_model(payload: object, name: str) -> None:
    """Log a GRU checkpoint payload as a final-window-scope pyfunc model.

    Args:
        payload: GRU checkpoint payload (``model_state_dict``, ``feature_cols``,
            ``sequence_config``, ``normalizer_payload``, ``max_rul``).
        name: Registered-model name to register the logged model under.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / _GRU_CHECKPOINT_FILENAME
        torch.save(payload, artifact_path)
        signature = _signature(np.zeros(1, dtype=np.float64))
        mlflow.pyfunc.log_model(
            name="model",
            python_model=GRUFinalWindowModel(),
            artifacts={_GRU_ARTIFACT_KEY: str(artifact_path)},
            signature=signature,
            registered_model_name=name,
        )


def log_and_register(
    model: object,
    *,
    model_type: str,
    subset: str,
) -> int:
    """Log a model into the active run and register a new registry version.

    The Ridge model is packaged so the logged model honors the full
    engine-scope inference contract (last-cycle-per-engine selection and
    clipping); the GRU model is packaged as a custom pyfunc wrapper that carries
    the checkpoint payload and runs the final-window path. Both reuse the
    existing predictor inference logic.

    Args:
        model: Fitted sklearn ``Pipeline`` (Ridge) or GRU checkpoint payload
            mapping (GRU).
        model_type: Model family, ``"ridge"`` or ``"gru"``.
        subset: C-MAPSS subset identifier, e.g. ``"FD001"``.

    Returns:
        The newly registered model-version number.

    Raises:
        ValueError: If ``model_type`` is not ``"ridge"`` or ``"gru"``.
        RuntimeError: If no MLflow run is active.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("log_and_register requires an active MLflow run.")
    name = model_name(model_type, subset)
    if model_type == "ridge":
        _log_ridge_model(model, name)
    elif model_type == "gru":
        _log_gru_model(model, name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return _latest_version(name)


def _latest_version(name: str) -> int:
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


@dataclass(frozen=True)
class RegisteredModelInfo:
    """Listing record for one registered model.

    Args:
        name: Registered-model name.
        versions: All registered version numbers, ascending.
        production_version: Version currently aliased ``@production``, or
            ``None`` if the alias is unset.
        val_rmse: ``val_rmse`` metric of the run that produced the production
            version, or ``None`` when unavailable.
        run_id: MLflow run id linked to the production version, or ``None``.
    """

    name: str
    versions: list[int]
    production_version: int | None
    val_rmse: float | None
    run_id: str | None


def list_registered() -> list[RegisteredModelInfo]:
    """List registered models with version, production alias, and provenance.

    For each registered model, reports its versions, the version currently
    aliased ``@production``, and the ``val_rmse`` metric and run id of the run
    that produced the production version (when available).

    Returns:
        One :class:`RegisteredModelInfo` per registered model, ordered by name.
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
    """Return the ``@production`` model version for a model, or ``None``.

    Args:
        client: MLflow tracking client.
        name: Registered-model name.

    Returns:
        The aliased model version, or ``None`` if no production alias is set.
    """
    try:
        return client.get_model_version_by_alias(name, "production")
    except mlflow.exceptions.MlflowException:
        return None


def _run_val_rmse(client: MlflowClient, run_id: str | None) -> float | None:
    """Read the ``val_rmse`` metric from a run, or ``None`` if unavailable.

    Args:
        client: MLflow tracking client.
        run_id: Run id linked to a model version, or ``None``.

    Returns:
        The ``val_rmse`` metric value, or ``None`` when missing.
    """
    if not run_id:
        return None
    try:
        run = client.get_run(run_id)
    except mlflow.exceptions.MlflowException:
        return None
    value = run.data.metrics.get(_VAL_RMSE_METRIC)
    return float(value) if value is not None else None


def configure_from_env() -> None:
    """Point MLflow at the tracking URI from the environment, if set.

    Returns:
        None.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
