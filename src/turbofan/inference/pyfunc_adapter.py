"""Adapter from MLflow pyfunc models to the inference predictor contract."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pandas as pd

from turbofan.inference import schemas
from turbofan.inference.schemas import (
    CANONICAL_COLUMNS,
    ModelType,
    PredictionMetadata,
    PredictionResult,
    PredictionRow,
    PredictionScope,
    RawRecords,
)
from turbofan.sklearn_types import DataFramePredictor

_MODEL_SCOPES: dict[ModelType, PredictionScope] = {
    "ridge": "engine",
    "gru": "final_window",
    "lstm": "final_window",
}


class PyfuncPredictor:
    """Adapt a loaded registry pyfunc model to the predictor contract.

    Args:
        model: Loaded pyfunc model returning the prediction output frame.
        model_type: Model family identifier.
        artifact_id: Stable identifier for the resolved registry version.
    """

    def __init__(
        self,
        model: DataFramePredictor,
        *,
        model_type: ModelType,
        artifact_id: str,
    ) -> None:
        """Store the loaded model and derived metadata.

        Args:
            model: Loaded pyfunc model returning the prediction output frame.
            model_type: Model family identifier.
            artifact_id: Stable identifier for the resolved registry version.
        """
        self._model = model
        self._metadata = PredictionMetadata(
            model_type=model_type,
            artifact_id=artifact_id,
            prediction_scope=_MODEL_SCOPES[model_type],
            input_rows=0,
            prediction_rows=0,
            warnings=[],
        )

    @property
    def metadata(self) -> PredictionMetadata:
        """Return descriptive metadata for the loaded registry model.

        Returns:
            Response-level metadata describing the resolved model.
        """
        return self._metadata

    def predict(
        self,
        records: RawRecords,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Predict remaining useful life through the loaded pyfunc model.

        Args:
            records: Raw canonical inference records.
            allow_partial: Accepted for call compatibility. The pyfunc wrapper
                validates strictly and does not skip rows.

        Returns:
            Prediction response with per-row predictions and response metadata.
        """
        del allow_partial
        input_rows = len(records)
        frame = _records_to_frame(records)
        output = self._model.predict(frame)
        if not isinstance(output, pd.DataFrame):
            raise ValueError("Pyfunc model predict() must return a DataFrame.")
        prediction_rows = _prediction_rows_from_output(
            output,
            model_type=self._metadata.model_type,
            artifact_id=self._metadata.artifact_id,
            prediction_scope=self._metadata.prediction_scope,
        )
        return PredictionResult(
            predictions=prediction_rows,
            metadata=PredictionMetadata(
                model_type=self._metadata.model_type,
                artifact_id=self._metadata.artifact_id,
                prediction_scope=self._metadata.prediction_scope,
                input_rows=input_rows,
                prediction_rows=len(prediction_rows),
                warnings=[],
            ),
        )


def _records_to_frame(records: RawRecords) -> pd.DataFrame:
    """Coerce raw records into a canonical-column DataFrame for pyfunc input."""
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(list(records))
    schemas.validate_raw_records(frame)
    return frame.loc[:, [c for c in CANONICAL_COLUMNS if c in frame.columns]]


def _prediction_rows_from_output(
    output: pd.DataFrame,
    *,
    model_type: ModelType,
    artifact_id: str,
    prediction_scope: PredictionScope,
) -> list[PredictionRow]:
    """Build prediction rows from the pyfunc output frame."""
    predicted_at = datetime.now(UTC)
    records = cast(
        list[dict[str, object]],
        output.loc[:, ["engine_id", "cycle", "prediction"]].to_dict("records"),
    )
    return [
        PredictionRow(
            engine_id=int(cast(int, row["engine_id"])),
            cycle=int(cast(int, row["cycle"])),
            prediction=float(cast(float, row["prediction"])),
            model_type=model_type,
            artifact_id=artifact_id,
            prediction_scope=prediction_scope,
            predicted_at=predicted_at,
        )
        for row in records
    ]
