"""MLflow pyfunc packaging for turbofan Ridge and sequence models."""
from __future__ import annotations

import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from mlflow.models import ModelSignature, infer_signature
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from turbofan.data.contracts import CANONICAL_COLUMNS
from turbofan.predictions import compute
from turbofan.registry import store
from turbofan.serving import schemas

_RIDGE_ARTIFACT_KEY = "pipeline"
_SEQUENCE_ARTIFACT_KEY = "checkpoint"
_SEQUENCE_CHECKPOINT_FILENAME = "model.pt"

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

PREDICTION_OUTPUT_COLUMNS = ["engine_id", "cycle", "prediction"]
"""Output columns produced by logged pyfunc models."""


def _prediction_frame(
    metadata: pd.DataFrame,
    predictions: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """Combine per-row metadata and predictions into the pyfunc output frame."""
    return pd.DataFrame(
        {
            "engine_id": metadata["engine_id"].to_numpy(dtype=np.int64),
            "cycle": metadata["cycle"].to_numpy(dtype=np.int64),
            "prediction": np.asarray(predictions, dtype=np.float64),
        },
        columns=PREDICTION_OUTPUT_COLUMNS,
    )


class RidgeEngineModel(PythonModel):
    """Pyfunc wrapper running the engine-scope Ridge inference contract."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the fitted Ridge pipeline from model artifacts.

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
        """Predict one non-negative RUL value per engine.

        Args:
            context: Pyfunc context. Loading happens in ``load_context``.
            model_input: Canonical raw inference records.
            params: Optional inference params.

        Returns:
            DataFrame with ``engine_id``, ``cycle``, and ``prediction`` columns.
        """
        del context, params
        validation = schemas.validate_raw_records(model_input)
        metadata, predictions = compute.ridge_engine_predictions(
            self._pipeline, validation.records
        )
        return _prediction_frame(metadata, predictions)


class SequenceFinalWindowModel(PythonModel):
    """Pyfunc wrapper running final-window sequence inference."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the sequence checkpoint payload from model artifacts.

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
            context: Pyfunc context. Loading happens in ``load_context``.
            model_input: Canonical raw inference records.
            params: Optional inference params.

        Returns:
            DataFrame with ``engine_id``, ``cycle``, and ``prediction`` columns.
        """
        del context, params
        validation = schemas.validate_raw_records(model_input)
        metadata, predictions = compute.sequence_final_window_predictions(
            self._payload, validation.records
        )
        return _prediction_frame(metadata, predictions)


def log_and_register(
    model: object,
    *,
    model_type: str,
    subset: str,
) -> int:
    """Log a model into the active run and register a new model version.

    Args:
        model: Fitted Ridge pipeline or sequence checkpoint payload.
        model_type: Model family, ``"ridge"``, ``"gru"``, or ``"lstm"``.
        subset: C-MAPSS subset identifier, e.g. ``"FD001"``.

    Returns:
        The newly registered model-version number.

    Raises:
        ValueError: If ``model_type`` is unsupported.
        RuntimeError: If no MLflow run is active.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("log_and_register requires an active MLflow run.")
    name = store.model_name(model_type, subset)
    if model_type == "ridge":
        _log_ridge_model(model, name)
    elif model_type in ("gru", "lstm"):
        _log_sequence_model(model, name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return store.latest_version(name)


@contextmanager
def _suppress_integer_schema_hint() -> Iterator[None]:
    """Silence MLflow's benign integer-column schema hint during logging."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column",
            category=UserWarning,
        )
        yield


def _sample_input() -> pd.DataFrame:
    """Build a one-row canonical-records frame for signature inference."""
    sample_input = pd.DataFrame(
        [[0.0] * len(CANONICAL_COLUMNS)],
        columns=CANONICAL_COLUMNS,
    )
    sample_input["engine_id"] = np.ones(len(sample_input), dtype=np.int64)
    sample_input["cycle"] = np.ones(len(sample_input), dtype=np.int64)
    return sample_input


def _signature() -> ModelSignature:
    """Infer the pyfunc signature for the canonical prediction contract."""
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
    """Log a fitted Ridge pipeline as an engine-scope pyfunc model."""
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
    """Log a sequence checkpoint payload as a final-window pyfunc model."""
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
