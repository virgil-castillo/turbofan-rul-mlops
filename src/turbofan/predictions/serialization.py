"""JSON serialization for inference prediction results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

from turbofan.predictions.contracts import PredictionResult


def prediction_result_to_dict(result: PredictionResult) -> dict[str, object]:
    """Serialize a prediction result into JSON-compatible primitives.

    Args:
        result: Prediction result dataclass.

    Returns:
        JSON-compatible response dictionary.
    """
    return cast(dict[str, object], _jsonable(asdict(result)))


def _jsonable(value: object) -> object:
    """Recursively coerce a value into JSON-compatible primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
