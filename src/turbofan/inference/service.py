"""FastAPI service factory for turbofan model inference."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from turbofan.inference.predictors import Predictor, load_predictor
from turbofan.inference.schemas import PredictionResult, SchemaValidationError


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
    artifact_path: Path | str | None = None,
    predictor: Predictor | None = None,
) -> FastAPI:
    """Create a FastAPI app with one loaded predictor.

    Args:
        artifact_path: Optional artifact manifest or directory path. If omitted,
            ``TURBOFAN_MODEL_ARTIFACT`` is used.
        predictor: Optional pre-loaded predictor, primarily for tests.

    Returns:
        Configured FastAPI application.

    Raises:
        ValueError: If no artifact is configured or the artifact cannot be loaded.
    """
    loaded_predictor = _resolve_predictor(
        artifact_path=artifact_path,
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
    artifact_path: Path | str | None,
    predictor: Predictor | None,
) -> Predictor:
    if predictor is not None:
        return predictor
    raw_artifact_path = artifact_path
    if raw_artifact_path is None:
        raw_artifact_path = os.environ.get("TURBOFAN_MODEL_ARTIFACT")
    if raw_artifact_path is None:
        raise ValueError(
            "Model artifact is required via artifact_path or "
            "TURBOFAN_MODEL_ARTIFACT."
        )
    return load_predictor(Path(raw_artifact_path))


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
