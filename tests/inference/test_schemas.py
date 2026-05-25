"""Tests for turbofan.inference.schemas."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from turbofan.inference.schemas import (
    CANONICAL_COLUMNS,
    FEATURE_COLUMNS,
    PredictionMetadata,
    PredictionResult,
    PredictionRow,
    SchemaValidationError,
    validate_raw_records,
)


def _record(
    *,
    engine_id: object = 1,
    cycle: object = 1,
    feature_value: object = 1.0,
) -> dict[str, object]:
    """Build one canonical raw turbofan record.

    Args:
        engine_id: Engine identifier value.
        cycle: Cycle value.
        feature_value: Value used for all operating and sensor columns.

    Returns:
        Raw record mapping with all canonical columns.
    """
    record: dict[str, object] = {"engine_id": engine_id, "cycle": cycle}
    for column in FEATURE_COLUMNS:
        record[column] = feature_value
    return record


def test_validate_raw_records_accepts_canonical_required_columns() -> None:
    """Canonical input validates and preserves expected columns."""
    result = validate_raw_records([_record()])

    assert result.warnings == []
    assert list(result.records.columns) == CANONICAL_COLUMNS
    assert result.records.loc[0, "engine_id"] == 1
    assert result.records.loc[0, "cycle"] == 1


def test_validate_raw_records_rejects_empty_input() -> None:
    """At least one raw record is required."""
    with pytest.raises(SchemaValidationError, match="at least one record"):
        validate_raw_records([])


def test_validate_raw_records_rejects_missing_required_columns() -> None:
    """All canonical columns must be present."""
    record = _record()
    del record["s_21"]

    with pytest.raises(SchemaValidationError, match="missing required columns"):
        validate_raw_records([record])


@pytest.mark.parametrize(
    ("engine_id", "cycle"),
    [(0, 1), (1, 0), (-1, 1), (1, -1), (1.5, 1), (1, "two")],
)
def test_validate_raw_records_rejects_non_positive_identifiers(
    engine_id: object,
    cycle: object,
) -> None:
    """Engine and cycle identifiers must be positive integers."""
    with pytest.raises(SchemaValidationError, match="positive integers"):
        validate_raw_records([_record(engine_id=engine_id, cycle=cycle)])


@pytest.mark.parametrize("value", ["bad", None, math.nan, math.inf, -math.inf])
def test_validate_raw_records_rejects_non_numeric_or_non_finite_features(
    value: object,
) -> None:
    """Feature values must be numeric and finite."""
    with pytest.raises(SchemaValidationError, match="numeric and finite"):
        validate_raw_records([_record(feature_value=value)])


def test_validate_raw_records_rejects_duplicate_engine_cycle() -> None:
    """Duplicate engine-cycle rows are invalid."""
    with pytest.raises(SchemaValidationError, match="duplicate"):
        validate_raw_records([_record(), _record()])


def test_validate_raw_records_drops_extra_columns() -> None:
    """Non-canonical columns are ignored after validation."""
    record = _record()
    record["unused"] = "drop me"

    result = validate_raw_records([record])

    assert "unused" not in result.records.columns
    assert list(result.records.columns) == CANONICAL_COLUMNS


def test_validate_raw_records_sorts_by_engine_and_cycle() -> None:
    """Valid rows are sorted before prediction."""
    records = [
        _record(engine_id=2, cycle=2),
        _record(engine_id=1, cycle=3),
        _record(engine_id=1, cycle=1),
    ]

    result = validate_raw_records(records)

    assert result.records[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 1},
        {"engine_id": 1, "cycle": 3},
        {"engine_id": 2, "cycle": 2},
    ]


def test_validate_raw_records_partial_mode_skips_invalid_rows() -> None:
    """Partial mode returns valid rows and warnings for isolated row failures."""
    records = [
        _record(engine_id=1, cycle=2),
        _record(engine_id=1, cycle=1, feature_value="bad"),
        _record(engine_id=2, cycle=1),
    ]

    result = validate_raw_records(records, partial=True)

    assert result.records[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 2},
        {"engine_id": 2, "cycle": 1},
    ]
    assert len(result.warnings) == 1
    assert "row 1" in result.warnings[0]
    assert "numeric and finite" in result.warnings[0]


def test_validate_raw_records_partial_duplicate_warning_uses_original_row_index(
) -> None:
    """Partial duplicate warnings use the original input row index."""
    records = [
        _record(engine_id=9, cycle=1, feature_value="bad"),
        _record(engine_id=1, cycle=1),
        _record(engine_id=1, cycle=1),
    ]

    result = validate_raw_records(records, partial=True)

    assert result.records[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 1},
    ]
    assert any("row 0" in warning for warning in result.warnings)
    assert any("row 2" in warning for warning in result.warnings)
    assert all("row 1:" not in warning for warning in result.warnings)


def test_validate_raw_records_partial_mode_fails_when_no_valid_rows_remain() -> None:
    """Partial mode still fails when every row is invalid."""
    with pytest.raises(SchemaValidationError, match="no valid rows"):
        validate_raw_records([_record(feature_value="bad")], partial=True)


def test_validation_result_records_are_a_dataframe() -> None:
    """Validated records are returned as a DataFrame for downstream predictors."""
    result = validate_raw_records([_record()])

    assert isinstance(result.records, pd.DataFrame)


def test_prediction_result_groups_rows_and_response_metadata() -> None:
    """PredictionResult is a response envelope around rows and metadata."""
    row = PredictionRow(
        engine_id=1,
        cycle=2,
        prediction=3.0,
        model_type="ridge",
        artifact_id="ridge-001",
        prediction_scope="row",
        predicted_at=pd.Timestamp("2026-05-25T00:00:00Z").to_pydatetime(),
    )
    metadata = PredictionMetadata(
        model_type="ridge",
        artifact_id="ridge-001",
        prediction_scope="row",
        input_rows=2,
        prediction_rows=1,
        warnings=["Skipped row 1: invalid."],
    )

    result = PredictionResult(predictions=[row], metadata=metadata)

    assert result.predictions == [row]
    assert result.metadata == metadata
