"""MLflow Model Registry wrapper for turbofan Ridge and sequence (GRU/LSTM) models.

This module is a thin seam over MLflow's registry, mirroring ``tracking.py``.
It packages the two model families into MLflow pyfunc models that honor the
existing inference contracts and exposes registration, promotion, resolution,
loading, and listing helpers over the local SQLite registry.

Model-packaging contract for both pyfunc wrappers:

- **Input:** a pandas ``DataFrame`` of canonical raw inference records with
  columns ``engine_id``, ``cycle``, ``op_1``..``op_3``, and ``s_1``..``s_21``.
- **Output:** a pandas ``DataFrame`` with columns ``engine_id`` (int64),
  ``cycle`` (int64), and ``prediction`` (float64) carrying one non-negative RUL
  prediction per engine, ordered by ``engine_id``. The Ridge wrapper returns the
  last-cycle prediction per engine (``engine`` scope); the shared sequence
  wrapper returns the final-window prediction per engine (``final_window``
  scope) for both GRU and LSTM. The metadata
  columns let downstream callers reconstruct the per-row prediction contract
  through the pyfunc boundary.
"""
from __future__ import annotations

import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
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
    PyfuncPredictor,
    ridge_engine_predictions,
    sequence_final_window_predictions,
)
from turbofan.inference.schemas import (
    CANONICAL_COLUMNS,
    ModelType,
    validate_raw_records,
)

_RIDGE_ARTIFACT_KEY = "pipeline"
_SEQUENCE_ARTIFACT_KEY = "checkpoint"
_SEQUENCE_CHECKPOINT_FILENAME = "model.pt"
_VAL_RMSE_METRIC = "val_rmse"

#: Pinned pip requirements for the logged pyfunc models. Declaring these
#: explicitly is the recommended MLflow practice for reproducible model
#: environments and, as a side effect, skips MLflow's automatic environment
#: inference at log time (which scans the active environment and costs several
#: seconds per model). Both wrappers import the shared inference compute from
#: the ``turbofan`` package, so it is pinned alongside the framework deps.
_RIDGE_PIP_REQUIREMENTS = [
    "mlflow",
    "scikit-learn",
    "joblib",
    "pandas",
    "numpy",
    "turbofan",
]
_SEQUENCE_PIP_REQUIREMENTS = [
    "mlflow",
    "torch",
    "pandas",
    "numpy",
    "turbofan",
]

#: Output columns produced by both pyfunc wrappers' ``predict`` methods.
PREDICTION_OUTPUT_COLUMNS = ["engine_id", "cycle", "prediction"]


def _prediction_frame(
    metadata: pd.DataFrame,
    predictions: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """Combine per-row metadata and predictions into the output contract frame.

    Args:
        metadata: Per-row ``engine_id``/``cycle`` metadata, ordered by
            ``engine_id``.
        predictions: Non-negative predictions aligned to ``metadata`` rows.

    Returns:
        DataFrame with ``engine_id`` (int64), ``cycle`` (int64), and
        ``prediction`` (float64) columns.
    """
    return pd.DataFrame(
        {
            "engine_id": metadata["engine_id"].to_numpy(dtype=np.int64),
            "cycle": metadata["cycle"].to_numpy(dtype=np.int64),
            "prediction": np.asarray(predictions, dtype=np.float64),
        },
        columns=PREDICTION_OUTPUT_COLUMNS,
    )


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
    clipping logic is shared with the in-process Ridge compute via
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
    ) -> pd.DataFrame:
        """Predict one non-negative RUL value per engine (last-cycle prediction).

        Args:
            context: Pyfunc context (unused; loading happens in load_context).
            model_input: Canonical raw inference records.
            params: Optional inference params (unused).

        Returns:
            DataFrame with ``engine_id``, ``cycle``, and ``prediction`` columns,
            one row per engine, ordered by ``engine_id``.
        """
        del context, params
        validation = validate_raw_records(model_input)
        metadata, predictions = ridge_engine_predictions(
            self._pipeline, validation.records
        )
        return _prediction_frame(metadata, predictions)


class SequenceFinalWindowModel(PythonModel):
    """Pyfunc wrapper running the final-window-scope sequence inference contract.

    Loads a sequence checkpoint payload (the same shape as the training
    ``model.pt``) from the logged artifact and, on ``predict``, validates the
    raw records and returns the final-window prediction per engine (normalize,
    window, forward, rescale by ``max_rul``, clip). Because the architecture is
    read from the payload's ``sequence_config``, this single wrapper serves every
    recurrent architecture (GRU and LSTM) — no per-architecture wrapper class is
    needed. The compute is shared with the in-process sequence path via
    ``sequence_final_window_predictions``.
    """

    def load_context(self, context: PythonModelContext) -> None:
        """Load the sequence checkpoint payload from the model artifacts.

        Args:
            context: Pyfunc context exposing the logged artifact paths.

        Returns:
            None.
        """
        self._payload = torch.load(
            context.artifacts[_SEQUENCE_ARTIFACT_KEY],
            map_location="cpu",
            weights_only=False,
        )

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """Predict one non-negative RUL value per engine final window.

        Args:
            context: Pyfunc context (unused; loading happens in load_context).
            model_input: Canonical raw inference records, each engine at least
                ``window_size`` cycles long.
            params: Optional inference params (unused).

        Returns:
            DataFrame with ``engine_id``, ``cycle``, and ``prediction`` columns,
            one row per engine final window, ordered by ``engine_id``.
        """
        del context, params
        validation = validate_raw_records(model_input)
        metadata, predictions = sequence_final_window_predictions(
            self._payload, validation.records
        )
        return _prediction_frame(metadata, predictions)


#: Backward-compatible alias for the shared sequence pyfunc wrapper, retained so
#: existing references to the GRU-specific name keep resolving.
GRUFinalWindowModel = SequenceFinalWindowModel


@contextmanager
def _suppress_integer_schema_hint() -> Iterator[None]:
    """Silence MLflow's benign integer-column schema hint during model logging.

    Our prediction contract genuinely carries ``engine_id`` and ``cycle`` as
    int64 on both the input records and the output frame, and they are never
    missing. MLflow's hint about integer columns being unable to represent
    missing values therefore does not apply, and casting these columns to
    float64 to silence it would corrupt the documented schema. Suppress only
    that one message so it does not leak as noise, leaving every other warning
    intact.

    Yields:
        None, with the integer-schema hint filtered for the duration.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column",
            category=UserWarning,
        )
        yield


def _sample_input() -> pd.DataFrame:
    """Build a one-row canonical-records frame for signature and input example.

    Returns:
        A single canonical raw-records row with ``engine_id``/``cycle`` typed as
        int64, matching the documented inference input contract.
    """
    sample_input = pd.DataFrame(
        [[0.0] * len(CANONICAL_COLUMNS)],
        columns=CANONICAL_COLUMNS,
    )
    # ``engine_id``/``cycle`` must be positive int64 to satisfy the model's own
    # record validation, which MLflow exercises when validating the logged
    # input example against the wrapper at log time.
    sample_input["engine_id"] = np.ones(len(sample_input), dtype=np.int64)
    sample_input["cycle"] = np.ones(len(sample_input), dtype=np.int64)
    return sample_input


def _signature() -> ModelSignature:
    """Infer the pyfunc signature for the canonical-records prediction contract.

    Returns:
        Model signature mapping canonical raw records to the
        ``engine_id``/``cycle``/``prediction`` output frame.
    """
    sample_output = pd.DataFrame(
        {
            "engine_id": np.zeros(1, dtype=np.int64),
            "cycle": np.zeros(1, dtype=np.int64),
            "prediction": np.zeros(1, dtype=np.float64),
        },
        columns=PREDICTION_OUTPUT_COLUMNS,
    )
    with _suppress_integer_schema_hint():
        return infer_signature(_sample_input(), sample_output)


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
        signature = _signature()
        with _suppress_integer_schema_hint():
            mlflow.pyfunc.log_model(
                name="model",
                python_model=RidgeEngineModel(),
                artifacts={_RIDGE_ARTIFACT_KEY: str(artifact_path)},
                signature=signature,
                input_example=_sample_input(),
                registered_model_name=name,
                pip_requirements=_RIDGE_PIP_REQUIREMENTS,
            )


def _log_sequence_model(payload: object, name: str) -> None:
    """Log a sequence checkpoint payload as a final-window-scope pyfunc model.

    Serves every recurrent architecture (GRU and LSTM): the shared
    ``SequenceFinalWindowModel`` reads the architecture from the payload's
    ``sequence_config``, so a single logger and pinned requirement set cover all
    RNNs.

    Args:
        payload: Sequence checkpoint payload (``model_state_dict``,
            ``feature_cols``, ``sequence_config``, ``normalizer_payload``,
            ``max_rul``).
        name: Registered-model name to register the logged model under.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / _SEQUENCE_CHECKPOINT_FILENAME
        torch.save(payload, artifact_path)
        signature = _signature()
        with _suppress_integer_schema_hint():
            mlflow.pyfunc.log_model(
                name="model",
                python_model=SequenceFinalWindowModel(),
                artifacts={_SEQUENCE_ARTIFACT_KEY: str(artifact_path)},
                signature=signature,
                input_example=_sample_input(),
                registered_model_name=name,
                pip_requirements=_SEQUENCE_PIP_REQUIREMENTS,
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
    clipping); sequence models (GRU and LSTM) are packaged as the shared pyfunc
    wrapper that carries the checkpoint payload and runs the final-window path,
    reading the architecture from the payload. Both reuse the existing predictor
    inference logic.

    Args:
        model: Fitted sklearn ``Pipeline`` (Ridge) or sequence checkpoint
            payload mapping (GRU/LSTM).
        model_type: Model family, ``"ridge"``, ``"gru"``, or ``"lstm"``.
        subset: C-MAPSS subset identifier, e.g. ``"FD001"``.

    Returns:
        The newly registered model-version number.

    Raises:
        ValueError: If ``model_type`` is not ``"ridge"``, ``"gru"``, or
            ``"lstm"``.
        RuntimeError: If no MLflow run is active.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("log_and_register requires an active MLflow run.")
    name = model_name(model_type, subset)
    if model_type == "ridge":
        _log_ridge_model(model, name)
    elif model_type in ("gru", "lstm"):
        _log_sequence_model(model, name)
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


def model_type_from_name(name: str) -> ModelType:
    """Infer the model family from a canonical registered-model name.

    Args:
        name: Registered-model name of the form ``turbofan-{type}-{subset}``.

    Returns:
        The model family, ``"ridge"``, ``"gru"``, or ``"lstm"``.

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
        A :class:`PyfuncPredictor` exposing the inference predictor contract.

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

    Supports both alias URIs (``models:/<name>@<alias>``) and version URIs
    (``models:/<name>/<version>``).

    Args:
        uri: Model URI of the form ``models:/<name>@<alias>`` or
            ``models:/<name>/<version>``.

    Returns:
        A :class:`PyfuncPredictor` exposing the inference predictor contract.

    Raises:
        ValueError: If the URI is not a supported ``models:`` URI or the model
            family cannot be inferred.
    """
    name, reference = _parse_models_uri(uri)
    model = cast(PyFuncModel, mlflow.pyfunc.load_model(uri))
    model_type = model_type_from_name(name)
    artifact_id = f"{name}/{reference}"
    return PyfuncPredictor(model, model_type=model_type, artifact_id=artifact_id)


def _parse_models_uri(uri: str) -> tuple[str, str]:
    """Parse a ``models:`` URI into its name and version/alias reference.

    Args:
        uri: Model URI of the form ``models:/<name>@<alias>`` or
            ``models:/<name>/<version>``.

    Returns:
        Tuple of the registered-model name and the version or alias reference.

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
