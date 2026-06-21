"""Inference DTOs and type aliases for turbofan model predictions.

These are inward-facing contracts shared by inference compute, validation,
the registry pyfunc adapter, the FastAPI service, and the batch CLI. They are
transport- and MLflow-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

IdentifierColumn = Literal["engine_id", "cycle"]
ModelType = Literal["ridge", "gru", "lstm"]
PredictionScope = Literal["engine", "final_window"]

RawRecords = Sequence[Mapping[str, object]] | pd.DataFrame


@dataclass(frozen=True)
class ValidationResult:
    """Validated records and non-fatal row warnings.

    Args:
        records: Canonical, sorted records ready for prediction.
        warnings: Descriptions of rows skipped in partial validation mode.
    """

    records: pd.DataFrame
    warnings: list[str]


@dataclass(frozen=True)
class PredictionRow:
    """JSON-friendly prediction row contract for inference outputs.

    Args:
        engine_id: Engine identifier for the predicted row or window.
        cycle: Cycle associated with the prediction.
        prediction: Predicted remaining useful life.
        model_type: Model family used for prediction.
        artifact_id: Loaded model artifact identifier.
        prediction_scope: Scope of the prediction.
        predicted_at: Timestamp when the prediction was produced.
    """

    engine_id: int
    cycle: int
    prediction: float
    model_type: ModelType
    artifact_id: str
    prediction_scope: PredictionScope
    predicted_at: datetime


@dataclass(frozen=True)
class PredictionMetadata:
    """Response-level prediction metadata.

    Args:
        model_type: Model family used for prediction.
        artifact_id: Loaded model artifact identifier.
        prediction_scope: Scope of the prediction.
        input_rows: Number of raw input rows received.
        prediction_rows: Number of prediction rows returned.
        warnings: Non-fatal validation or filtering warnings.
    """

    model_type: ModelType
    artifact_id: str
    prediction_scope: PredictionScope
    input_rows: int
    prediction_rows: int
    warnings: list[str]


@dataclass(frozen=True)
class PredictionResult:
    """Response envelope for inference predictions.

    Args:
        predictions: Prediction rows produced by the model.
        metadata: Response-level prediction metadata.
    """

    predictions: list[PredictionRow]
    metadata: PredictionMetadata
