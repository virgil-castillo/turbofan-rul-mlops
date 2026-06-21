"""Tests for the canonical raw C-MAPSS column contract."""

from __future__ import annotations

from turbofan.data import contracts


def test_canonical_columns_start_with_identifiers() -> None:
    """The canonical layout leads with the identifier columns."""
    assert contracts.CANONICAL_COLUMNS[:2] == ["engine_id", "cycle"]


def test_canonical_columns_are_identifiers_then_features() -> None:
    """Canonical columns are the identifiers followed by the feature columns."""
    expected = [*contracts.IDENTIFIER_COLUMNS, *contracts.FEATURE_COLUMNS]
    assert expected == contracts.CANONICAL_COLUMNS


def test_feature_columns_are_operating_modes_then_sensors() -> None:
    """Feature columns are the operating-condition columns then the sensors."""
    expected = [*contracts.OPERATING_CONDITION_COLUMNS, *contracts.SENSOR_COLUMNS]
    assert expected == contracts.FEATURE_COLUMNS


def test_sensor_columns_cover_all_21_sensors() -> None:
    """The contract enumerates all 21 C-MAPSS sensor channels in order."""
    expected = [f"s_{index}" for index in range(1, 22)]
    assert expected == contracts.SENSOR_COLUMNS
    assert len(contracts.SENSOR_COLUMNS) == 21


def test_canonical_columns_have_no_duplicates() -> None:
    """Every canonical column name is unique."""
    assert len(contracts.CANONICAL_COLUMNS) == len(set(contracts.CANONICAL_COLUMNS))
