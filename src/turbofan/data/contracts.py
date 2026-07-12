"""Canonical raw C-MAPSS record contract.

Single inward-facing source of truth for the ordered raw column layout of the
NASA C-MAPSS dataset. The data loader, inference validation, and MLflow model
signatures all import these names so that loading, validation, and serving
cannot drift apart when a sensor or identifier rule changes.
"""

from __future__ import annotations

#: Per-row identifier columns, in canonical order.
IDENTIFIER_COLUMNS: list[str] = ["engine_id", "cycle"]

#: Operating-condition setting columns, in canonical order.
OPERATING_CONDITION_COLUMNS: list[str] = ["op_1", "op_2", "op_3"]

#: Sensor measurement columns ``s_1``..``s_21``, in canonical order.
SENSOR_COLUMNS: list[str] = [f"s_{index}" for index in range(1, 22)]

#: Model feature columns: operating conditions followed by sensors.
FEATURE_COLUMNS: list[str] = [*OPERATING_CONDITION_COLUMNS, *SENSOR_COLUMNS]

#: Full canonical raw record: identifiers followed by feature columns.
CANONICAL_COLUMNS: list[str] = [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]
