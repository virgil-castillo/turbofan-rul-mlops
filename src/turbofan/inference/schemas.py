"""Schema validation for local turbofan inference inputs and outputs."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral, Real
from typing import Literal, cast

import pandas as pd

IdentifierColumn = Literal["engine_id", "cycle"]
ModelType = Literal["ridge", "gru"]
PredictionScope = Literal["row", "final_window"]

FEATURE_COLUMNS: list[str] = [
    "op_1",
    "op_2",
    "op_3",
    *[f"s_{index}" for index in range(1, 22)],
]
CANONICAL_COLUMNS: list[str] = ["engine_id", "cycle", *FEATURE_COLUMNS]


class SchemaValidationError(ValueError):
    """Raised when raw inference records do not satisfy the canonical schema."""


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


RawRecords = Sequence[Mapping[str, object]] | pd.DataFrame


@dataclass(frozen=True)
class _IndexedRecord:
    """Validated row with the original input index.

    Args:
        row_index: Original zero-based input row index.
        record: Validated canonical row.
    """

    row_index: int
    record: dict[str, object]


def validate_raw_records(
    records: RawRecords,
    *,
    partial: bool = False,
) -> ValidationResult:
    """Validate canonical raw C-MAPSS inference records.

    Args:
        records: Raw records as mappings or a DataFrame.
        partial: Whether invalid row-level records should be skipped.

    Returns:
        Validated canonical records sorted by engine and cycle.

    Raises:
        SchemaValidationError: If validation fails or no valid rows remain.
    """
    raw_records = _to_record_mappings(records)
    if not raw_records:
        raise SchemaValidationError("Inference input must contain at least one record.")

    valid_rows: list[_IndexedRecord] = []
    warnings: list[str] = []
    for row_index, record in enumerate(raw_records):
        try:
            valid_rows.append(
                _IndexedRecord(
                    row_index=row_index,
                    record=_validate_record(record),
                )
            )
        except SchemaValidationError as exc:
            if not partial:
                raise
            warnings.append(f"Skipped row {row_index}: {exc}")

    if not valid_rows:
        raise SchemaValidationError("Partial validation left no valid rows.")

    frame = pd.DataFrame(
        [
            {**indexed_record.record, "_row_index": indexed_record.row_index}
            for indexed_record in valid_rows
        ],
        columns=[*CANONICAL_COLUMNS, "_row_index"],
    )
    duplicate_mask = frame.duplicated(subset=["engine_id", "cycle"], keep="first")
    if duplicate_mask.any():
        if not partial:
            raise SchemaValidationError(
                "Input contains duplicate engine_id/cycle rows."
            )
        duplicate_indices = frame.index[duplicate_mask].tolist()
        for duplicate_index in duplicate_indices:
            original_row_index = int(frame.loc[duplicate_index, "_row_index"])
            warnings.append(
                "Skipped row "
                f"{original_row_index}: duplicate engine_id/cycle combination."
            )
        frame = frame.loc[~duplicate_mask].copy()

    frame = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    frame = frame.loc[:, CANONICAL_COLUMNS]
    return ValidationResult(records=frame, warnings=warnings)


def _to_record_mappings(records: RawRecords) -> list[Mapping[str, object]]:
    """Convert supported raw record containers to mapping rows.

    Args:
        records: Raw records as mappings or a DataFrame.

    Returns:
        List of record mappings.
    """
    if isinstance(records, pd.DataFrame):
        return cast(list[Mapping[str, object]], records.to_dict("records"))
    return list(records)


def _validate_record(record: Mapping[str, object]) -> dict[str, object]:
    """Validate one raw record and drop non-canonical fields.

    Args:
        record: Raw record mapping.

    Returns:
        Canonical row mapping.

    Raises:
        SchemaValidationError: If the record is invalid.
    """
    missing_columns = [column for column in CANONICAL_COLUMNS if column not in record]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise SchemaValidationError(f"Record is missing required columns: {joined}.")

    engine_id = _validate_identifier(record["engine_id"], "engine_id")
    cycle = _validate_identifier(record["cycle"], "cycle")

    row: dict[str, object] = {"engine_id": engine_id, "cycle": cycle}
    for column in FEATURE_COLUMNS:
        row[column] = _validate_feature(record[column], column)
    return row


def _validate_identifier(value: object, column: IdentifierColumn) -> int:
    """Validate a positive integer identifier.

    Args:
        value: Raw identifier value.
        column: Identifier column name.

    Returns:
        Validated identifier as an integer.

    Raises:
        SchemaValidationError: If the value is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise SchemaValidationError(f"{column} must contain positive integers.")
    return int(value)


def _validate_feature(value: object, column: str) -> float:
    """Validate a numeric, finite feature value.

    Args:
        value: Raw feature value.
        column: Feature column name.

    Returns:
        Validated feature as a float.

    Raises:
        SchemaValidationError: If the value is not numeric and finite.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SchemaValidationError(f"{column} must be numeric and finite.")
    feature = float(value)
    if not math.isfinite(feature):
        raise SchemaValidationError(f"{column} must be numeric and finite.")
    return feature
