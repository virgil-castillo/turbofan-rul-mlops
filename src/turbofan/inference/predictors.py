"""Compatibility exports for inference compute and pyfunc predictor adapters."""
from __future__ import annotations

from turbofan.inference.prediction_compute import (
    DEFAULT_MAX_RUL,
    ridge_engine_predictions,
    sequence_final_window_predictions,
)
from turbofan.inference.pyfunc_adapter import _MODEL_SCOPES, PyfuncPredictor

__all__ = [
    "DEFAULT_MAX_RUL",
    "PyfuncPredictor",
    "_MODEL_SCOPES",
    "ridge_engine_predictions",
    "sequence_final_window_predictions",
]
