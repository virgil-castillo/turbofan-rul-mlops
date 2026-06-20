"""Compatibility facade for MLflow model registry helpers.

Pyfunc model packaging lives in :mod:`turbofan.registry.pyfunc`; registry URI
resolution, loading, and listing live in :mod:`turbofan.registry.store`. This
module preserves the historical ``turbofan.registry`` import path.
"""
from __future__ import annotations

from turbofan.registry.pyfunc import (
    PREDICTION_OUTPUT_COLUMNS,
    RidgeEngineModel,
    SequenceFinalWindowModel,
    log_and_register,
)
from turbofan.registry.store import (
    RegisteredModelInfo,
    list_registered,
    load,
    load_predictor,
    load_predictor_from_uri,
    model_name,
    model_type_from_name,
    promote,
    resolve_uri,
)

__all__ = [
    "PREDICTION_OUTPUT_COLUMNS",
    "RegisteredModelInfo",
    "RidgeEngineModel",
    "SequenceFinalWindowModel",
    "list_registered",
    "load",
    "load_predictor",
    "load_predictor_from_uri",
    "log_and_register",
    "model_name",
    "model_type_from_name",
    "promote",
    "resolve_uri",
]
