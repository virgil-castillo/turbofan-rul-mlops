"""Tests for turbofan.serving.service."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from turbofan.serving.schemas import (
    FEATURE_COLUMNS,
    PredictionMetadata,
    PredictionResult,
    PredictionRow,
    RawRecords,
    SchemaValidationError,
)


class _StaticPredictor:
    """Small predictor implementation for service behavior tests."""

    def __init__(self) -> None:
        """Create a predictor with stable test metadata."""
        self._metadata = PredictionMetadata(
            model_type="ridge",
            artifact_id="turbofan-ridge-fd001/1",
            prediction_scope="engine",
            input_rows=0,
            prediction_rows=0,
            warnings=[],
        )
        self.allow_partial_seen: bool | None = None

    @property
    def metadata(self) -> PredictionMetadata:
        """Return stable metadata for health responses.

        Returns:
            Prediction metadata for the fake predictor.
        """
        return self._metadata

    def predict(
        self,
        records: RawRecords,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Return one deterministic prediction for the request.

        Args:
            records: Request records.
            allow_partial: Whether partial validation was requested.

        Returns:
            Deterministic prediction response.
        """
        self.allow_partial_seen = allow_partial
        return PredictionResult(
            predictions=[
                PredictionRow(
                    engine_id=1,
                    cycle=2,
                    prediction=12.5,
                    model_type="ridge",
                    artifact_id="turbofan-ridge-fd001/1",
                    prediction_scope="engine",
                    predicted_at=datetime(2026, 5, 25, tzinfo=UTC),
                )
            ],
            metadata=PredictionMetadata(
                model_type="ridge",
                artifact_id="turbofan-ridge-fd001/1",
                prediction_scope="engine",
                input_rows=len(records),
                prediction_rows=1,
                warnings=[],
            ),
        )


class _FailingPredictor(_StaticPredictor):
    """Predictor that raises at runtime."""

    def predict(
        self,
        records: RawRecords,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Raise a runtime prediction error.

        Args:
            records: Request records.
            allow_partial: Whether partial validation was requested.

        Raises:
            RuntimeError: Always raised for test coverage.
        """
        raise RuntimeError("model exploded")


class _SchemaFailingPredictor(_StaticPredictor):
    """Predictor that raises a canonical schema validation error."""

    def predict(
        self,
        records: RawRecords,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Raise a schema validation error.

        Args:
            records: Request records.
            allow_partial: Whether partial validation was requested.

        Raises:
            SchemaValidationError: Always raised for test coverage.
        """
        raise SchemaValidationError("Record is missing required columns: s_21.")


class _ShortWindowFailingPredictor(_StaticPredictor):
    """Predictor that raises a short-window validation error."""

    def predict(
        self,
        records: RawRecords,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Raise a short-window-style validation error.

        Args:
            records: Request records.
            allow_partial: Whether partial validation was requested.

        Raises:
            SchemaValidationError: Always raised for test coverage.
        """
        raise SchemaValidationError("Engine(s) shorter than window_size 3: 2.")


def _record() -> dict[str, object]:
    """Build one canonical request record.

    Returns:
        Canonical inference record.
    """
    return {
        "engine_id": 1,
        "cycle": 2,
        **dict.fromkeys(FEATURE_COLUMNS, 1.0),
    }


def test_health_returns_loaded_model_metadata() -> None:
    """Health endpoint reports status and loaded artifact metadata."""
    from turbofan.serving.service import create_app

    client = TestClient(create_app(predictor=_StaticPredictor()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": {
            "model_type": "ridge",
            "artifact_id": "turbofan-ridge-fd001/1",
            "prediction_scope": "engine",
        },
    }


def test_predict_returns_predictions_and_metadata() -> None:
    """Predict endpoint returns serialized predictions and response metadata."""
    from turbofan.serving.service import create_app

    predictor = _StaticPredictor()
    client = TestClient(create_app(predictor=predictor))

    response = client.post(
        "/predict",
        json={"records": [_record()], "allow_partial": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"] == {
        "model_type": "ridge",
        "artifact_id": "turbofan-ridge-fd001/1",
        "prediction_scope": "engine",
        "input_rows": 1,
        "prediction_rows": 1,
        "warnings": [],
    }
    assert payload["predictions"] == [
        {
            "engine_id": 1,
            "cycle": 2,
            "prediction": 12.5,
            "model_type": "ridge",
            "artifact_id": "turbofan-ridge-fd001/1",
            "prediction_scope": "engine",
            "predicted_at": "2026-05-25T00:00:00+00:00",
        }
    ]
    assert predictor.allow_partial_seen is True


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"records": "not-a-list"},
        {"records": [_record()], "allow_partial": "yes"},
    ],
)
def test_predict_rejects_invalid_request_payload(payload: dict[str, Any]) -> None:
    """Invalid prediction request bodies return HTTP 422 details."""
    from turbofan.serving.service import create_app

    client = TestClient(create_app(predictor=_StaticPredictor()))

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_returns_500_for_unexpected_runtime_errors() -> None:
    """Unexpected predictor errors are returned as HTTP 500 responses."""
    from turbofan.serving.service import create_app

    client = TestClient(create_app(predictor=_FailingPredictor()))

    response = client.post("/predict", json={"records": [_record()]})

    assert response.status_code == 500
    assert "model exploded" in response.json()["detail"]


def test_predict_returns_422_for_canonical_schema_validation_errors() -> None:
    """Schema validation failures from predictors return HTTP 422 details."""
    from turbofan.serving.service import create_app

    client = TestClient(create_app(predictor=_SchemaFailingPredictor()))

    response = client.post("/predict", json={"records": [_record()]})

    assert response.status_code == 422
    assert "missing required columns" in response.json()["detail"]


def test_predict_returns_422_for_short_window_validation_errors() -> None:
    """Short-window predictor validation failures return HTTP 422 details."""
    from turbofan.serving.service import create_app

    client = TestClient(create_app(predictor=_ShortWindowFailingPredictor()))

    response = client.post("/predict", json={"records": [_record()]})

    assert response.status_code == 422
    assert "shorter than window_size" in response.json()["detail"]


def test_create_app_fails_when_no_model_name_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory fails clearly when no model name is provided or configured."""
    from turbofan.serving.service import MODEL_NAME_ENV, create_app

    monkeypatch.delenv(MODEL_NAME_ENV, raising=False)

    with pytest.raises(ValueError, match="Registered model name is required"):
        create_app()


def test_create_app_fails_for_unresolvable_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory fails when the configured model name cannot be resolved."""
    from turbofan.serving.service import MODEL_NAME_ENV, create_app

    monkeypatch.delenv(MODEL_NAME_ENV, raising=False)

    with pytest.raises(Exception, match="turbofan-ridge-does-not-exist"):
        create_app(model_name="turbofan-ridge-does-not-exist")
