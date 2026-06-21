"""FastAPI service factory for turbofan model inference."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from turbofan.serving.schemas import (
    PredictionMetadata,
    PredictionResult,
    RawRecords,
    SchemaValidationError,
)

#: Environment variable naming the registered model to resolve at startup.
MODEL_NAME_ENV = "TURBOFAN_MODEL_NAME"
#: Environment variable selecting the alias to resolve (defaults to production).
MODEL_ALIAS_ENV = "TURBOFAN_MODEL_ALIAS"


class _Predictor(Protocol):
    """Predictor interface the service routes depend on."""

    @property
    def metadata(self) -> PredictionMetadata:
        """Return descriptive metadata for the loaded model.

        Returns:
            Response-level model metadata.
        """
        ...

    def predict(
        self,
        records: RawRecords,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Predict remaining useful life for request records.

        Args:
            records: Raw canonical inference records.
            allow_partial: Whether invalid or short inputs may be skipped.

        Returns:
            Prediction response with rows and metadata.
        """
        ...


class PredictRequest(BaseModel):
    """HTTP request body for batch prediction.

    Args:
        records: Canonical turbofan inference records.
        allow_partial: Whether invalid rows may be skipped with warnings.
    """

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, object]] = Field(min_length=1)
    allow_partial: StrictBool = False


def create_app(
    *,
    model_name: str | None = None,
    alias: str | None = None,
    predictor: _Predictor | None = None,
) -> FastAPI:
    """Create a FastAPI app resolving one registered model by name.

    The model is resolved from ``models:/<name>@<alias>`` via the registry. When
    ``model_name`` is omitted, the ``TURBOFAN_MODEL_NAME`` environment variable
    is used; when ``alias`` is omitted, the ``TURBOFAN_MODEL_ALIAS`` environment
    variable is used, defaulting to ``"production"``.

    Args:
        model_name: Registered-model name to resolve. Falls back to
            ``TURBOFAN_MODEL_NAME``.
        alias: Alias to resolve. Falls back to ``TURBOFAN_MODEL_ALIAS`` then
            ``"production"``.
        predictor: Optional pre-loaded predictor, primarily for tests.

    Returns:
        Configured FastAPI application.

    Raises:
        ValueError: If no model name is configured or the model cannot be loaded.
    """
    loaded_predictor = _resolve_predictor(
        model_name=model_name,
        alias=alias,
        predictor=predictor,
    )
    app = FastAPI(title="Turbofan Inference API")

    @app.get("/health")
    def health() -> dict[str, object]:
        """Return service and loaded model health metadata.

        Returns:
            Health response payload.
        """
        metadata = loaded_predictor.metadata
        return {
            "status": "ok",
            "model": {
                "model_type": metadata.model_type,
                "artifact_id": metadata.artifact_id,
                "prediction_scope": metadata.prediction_scope,
            },
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, object]:
        """Run prediction for canonical request records.

        Args:
            request: Validated HTTP request body.

        Returns:
            Serialized prediction response.

        Raises:
            HTTPException: If prediction raises an unexpected runtime error.
        """
        try:
            result = loaded_predictor.predict(
                request.records,
                allow_partial=request.allow_partial,
            )
        except SchemaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return prediction_result_to_dict(result)

    return app


def prediction_result_to_dict(result: PredictionResult) -> dict[str, object]:
    """Serialize a prediction result into JSON-compatible primitives.

    Args:
        result: Prediction result dataclass.

    Returns:
        JSON-compatible response dictionary.
    """
    return cast(dict[str, object], _jsonable(asdict(result)))


def _resolve_predictor(
    *,
    model_name: str | None,
    alias: str | None,
    predictor: _Predictor | None,
) -> _Predictor:
    if predictor is not None:
        return predictor
    from turbofan import registry

    name = model_name or os.environ.get(MODEL_NAME_ENV)
    if not name:
        raise ValueError(
            "Registered model name is required via model_name or "
            f"the {MODEL_NAME_ENV} environment variable."
        )
    resolved_alias = alias or os.environ.get(MODEL_ALIAS_ENV) or "production"
    registry.tracking.configure_mlflow()
    return registry.load_predictor(name, resolved_alias)


def _jsonable(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
